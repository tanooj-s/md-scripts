# output timeseries of contact angles and hysteresis from a lammps dump file

import os
import time
import glob
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import basinhopping

parser = argparse.ArgumentParser(description="calculate contact angle from molecular dynamics data")
parser.add_argument('-i', action="store", dest="input") # .dump
parser.add_argument('-o', action="store", dest="output") # .npy
parser.add_argument('-dz', action="store", dest="dz") 
parser.add_argument('-dx', action="store", dest="dx") # bin widths in angstroms, use 10 by default
parser.add_argument('-minz', action="store", dest="minz")
parser.add_argument('-minx', action="store", dest="minx")
parser.add_argument('-maxx', action="store", dest="maxx") # angstroms
parser.add_argument('-start', action="store", dest="start") # timestep
parser.add_argument('-end', action="store", dest="end") # timestep
parser.add_argument('-window', action="store", dest="window") # window to average over (in timesteps) for each density snapshot
parser.add_argument('-collect', action="store", dest="collect") # collect data y or n
parser.add_argument('-analyze', action="store", dest="analyze") # analyze data y or n
parser.add_argument('-mode', action="store", dest="mode") # "static" or "dynamic", fitting procedure is different
parser.add_argument('-t', action="store", dest="atom_types") # atom types to use for analysis, string like '1 2'
args = parser.parse_args()

dt = 2e-6 # simulation timestep nanoseconds
dz = float(args.dz)
dx = float(args.dx)
minx = float(args.minx)
maxx = float(args.maxx)
minz = float(args.minz)
L = maxx - minx
maxz = minz + L
start = int(args.start)
end = int(args.end)
window = int(args.window)
snapshots = int(np.floor((end-start)/window))
assert args.collect in ['y','n']
assert args.analyze in ['y','n']
assert args.mode in ["static","dynamic"]
doCollect = True if args.collect == 'y' else False
doAnalyze = True if args.analyze == 'y' else False
atom_types = [int(t) for t in args.atom_types.split(' ')]
print(f"Atom types to bin: {atom_types}")
print(f"Density snapshots: {snapshots}")

# ===============================

def purge(tokens): # purge empty strings from token lists
	return [t for t in tokens if len(t) >= 1] 

def isfloat(s): # hack function to confirm tokens are numeric as floats
	try:
		float(s)
		return True
	except ValueError:
		return False

def dQdY(q):
	'''
	compute dq/dy using finite differences
	compute the derivative at point x as 0.5 * ((q(x+1)-q(x)) + (q(x)-q(x-1))) / dx
	here dy is just the width of one grid point
	assume x is axis 0, y is axis 1
	'''
	dq = np.zeros(q.shape)
	for i in range(1,q.shape[1]-1): dq[:,i] = 0.5 * ((q[:,i+1]-q[:,i]) + (q[:,i]-q[:,i-1]))
	dq[:,0] = 0.5 * ((q[:,1]-q[:,0]) + (q[:,0]-q[:,-1]))
	dq[:,-1] = 0.5 * ((q[:,0]-q[:,-1]) + (q[:,-1]-q[:,-2]))
	return dq # note this is in terms of grid spacings

def dQdX(q):
	dq = np.zeros(q.shape)
	for i in range(1,q.shape[0]-1): dq[i,:] = 0.5 * ((q[i+1,:]-q[i,:]) + (q[i,:]-q[i-1,:]))
	dq[0,:] = 0.5 * ((q[1,:]-q[0,:]) + (q[0,:]-q[-1,:]))
	dq[-1,:] = 0.5 * ((q[0,:]-q[-1,:]) + (q[-1,:]-q[-2,:]))
	return dq 

def COM(field2d):
	'''
	function to obtain center of mass coordinates of a 2d field
	assuming pixel values are scalars like mass/charge/number density
	'''
	mean_y = np.sum((np.mean(field2d,axis=0) * np.arange(0,field2d.shape[0],1))) / np.sum(np.mean(field2d,axis=0))
	mean_x = np.sum((np.mean(field2d,axis=1) * np.arange(0,field2d.shape[1],1))) / np.sum(np.mean(field2d,axis=1))
	return (mean_x,mean_y)

def arcLoss(params):
	'''
	calculate distance loss from ground truth points 
	assume params are r, h
	arc is a (npoints,2) shaped array to calculate loss from
	note that it is not an argument of this function
	also note xsplit is used as x coordinate of circle center
	'''
	r, h = params # guessed radius and center, x0 is fixed
	distances = []
	for pt in arc: distances.append(((pt[1]-h)**2 + (pt[0]-xsplit)**2) ** 0.5) # !!
	distances = np.array(distances)
	distance_loss = np.sum(((distances-r) / distances) ** 2)
	return distance_loss

def angularscan(field2d,center,thetalim,ylo):
	'''
	given an (x,y) center coordinate of a 2d field (assumed to be gradient)
	and given a range of theta in degrees
	perform an angular scan to return points [(x,y),(x,y)...] of an arc
	ylo scalar lower limit to not use data from flat part of profile
	'''
	xarc, yarc = [], []
	for t in thetalim:
		x = center[0] + np.arange(0,int(0.4*field2d.shape[0]),1)*np.cos(t * (np.pi/180))
		y = center[1] + np.arange(0,int(0.4*field2d.shape[0]),1)*np.sin(t * (np.pi/180)) # length of line segment is a bit hacky
		xline = np.array([int(a) for a in np.floor(x) if not np.isnan(a)])
		yline = np.array([int(a) for a in np.floor(y) if not np.isnan(a)])
		if (len(xline) > 0) and (len(yline) > 0):
			line = field2d[xline,yline] # gradient values along this line segment
			edge_idx = np.argmax(line)
			xarc.append(center[0] + edge_idx * np.cos(t * (np.pi/180)))
			yarc.append(center[1] + edge_idx * np.sin(t * (np.pi/180)))
	xarc, yarc = np.array(xarc), np.array(yarc)
	xarc, yarc = xarc[yarc > ylo], yarc[yarc > ylo]
	return xarc, yarc


# ===================================================

if doCollect:
	print(f'MinZ: {minz}')
	print(f'MaxZ: {maxz}')
	print(f'MinX: {minx}')
	print(f'MaxX: {maxx}')
	nbinsz = int((maxz-minz)/dz) + 1
	nbinsx = int((maxx-minx)/dx) + 1
	print(f"nbinsz: {nbinsz}")
	print(f"nbinsx: {nbinsx}")
	densities = np.zeros((snapshots,nbinsx,nbinsz)) # directly fill in this 3-tensor with correct indices along time axis
	print("Parsing and analyzing dump directly to obtain scalar fields on a grid")
	timestart = time.time()
	tidx = None # time index for numpy array above
	nCollected = 0
	with open(args.input,'r') as f:
		currentTime = 0 # simulation timestep
		previousLine = ''
		for line in f:
			if type(tidx) is int:
				if tidx < densities.shape[0]:
					tokens = purge(line.strip('\n').split(' '))
					if line.startswith("ITEM: ATOMS"):
						typeIdx = tokens.index('type') - 2
						xIdx = tokens.index('x') - 2
						yIdx = tokens.index('y') - 2 # note different naming convention for histogram indices below
						zIdx = tokens.index('z') - 2
						#qIdx = tokens.index('q') - 2 # ... 
					if (len(tokens) > 2):
						checksum = np.sum([not isfloat(t) for t in tokens]) 
						if checksum == 0: # first check to make sure box bound lines aren't being used
							aType = int(tokens[typeIdx])
							x = float(tokens[xIdx]) 
							z = float(tokens[zIdx])  
							if (aType in atom_types) and (z >= minz) and (z <= maxz) and (x >= minx) and (x <= maxx):			
								zidx = int((z-minz)/dz)
								xidx = int((x-minx)/dx)
								densities[tidx,xidx,zidx] += 1 # number density 
			if previousLine.startswith('ITEM: TIMESTEP'):
				currentTime = int(line.strip('\n'))
				if currentTime < start: 
					print(f"Skipping data collection for timestep {currentTime}")
					tidx = None 
				elif currentTime >= end:
					break
				else:
					tidx = int(np.floor((currentTime-start)/window))
					nCollected += 1	
					# reset normalization nCollected if next time window is hit
					if (currentTime-start) % window == 0:
						print(f"Averaging particle positions for snapshot index {tidx}...")	 
						if tidx < densities.shape[0]: # off by one error given certain end flag + window flag combinations 
							densities[tidx,:,:] /= nCollected
							nCollected = 0
			previousLine = line
		print(f"{round((time.time()-timestart)/60,4)} minutes to obtain densities")
	# TODO there should also be metadata about averaging window
	with open(args.output,'wb') as f: np.save(f,densities)

# ==========================================================

if doAnalyze:
	if doCollect == False: 
		with open(args.output,'rb') as f: densities = np.load(f)
	print(f"{densities.shape[0]} gradient snapshots to fit circles to")
	# iterate over snapshots, measure forward, backward angles for each and save to output later
	# contact angle hysteresis for dynamic case calculated as per method in DOI: 10.1021/acs.langmuir.9b00551 


	# only last half for initial shape
	#nUse = int(0.5*densities.shape[0])
	#field = np.mean(densities[nUse:,:,:],axis=0)
	#(X,Y) = densities.shape 
	#densities = np.reshape(densities,(1,X,Y))

	forward = np.zeros(densities.shape[0])
	backward = np.zeros(densities.shape[0])
	times = np.zeros(densities.shape[0]) # sim time ns
	img_array = [] # for video
	timestart = time.time()

	for i in range(densities.shape[0]):
		plt.rcParams['figure.figsize'] = (15,5)
		plt.rcParams['font.size'] = 20
		time_ns = dt*(start+((i+1)*window)) # nanoseconds
		times[i] = time_ns
		field = densities[i]
		x0, y0 = COM(field)
		dqdx = dQdX(field)
		dqdy = dQdY(field)
		gradient = (dqdx**2 + dqdy**2) ** 0.5	
		if args.mode == "static":
			fig, axes = plt.subplots(1,1)
			# use entire arc of ground truth points to fit one circle with one curvature R
			thetascan = np.arange(-90,271,0.05)
			xarc, yarc = angularscan(gradient,(x0,y0),thetascan,2)
			xsplit = x0 # actually redundant but using xsplit variable name in arcLoss function
			arc = np.array((xarc,yarc)).T # for loss function
			r = np.add((xarc-x0)**2,(yarc-y0)**2)**0.5 # array of distances of ground truth points from center
			rinit = np.mean(r)
			[rfit, hfit] = basinhopping(arcLoss,[rinit,y0],niter=1000,T=0.7,stepsize=0.1,minimizer_kwargs={"bounds":[(0.25*rinit,4*rinit),(-2*rinit,2*rinit)]}).x
			axes.scatter([xsplit],[hfit],color='r',marker='x')
			if hfit <= rfit: # otherwise skip this snapshot
				contact_angle = 90 + (180/np.pi) * np.arcsin(hfit/rfit)
				forward[i] = contact_angle
				backward[i] = contact_angle		
				thetamodel = np.arange(-90,271,8) # range of model to output
				xfit = x0 + rfit*np.cos(thetamodel*np.pi/180)
				yfit = hfit + rfit*np.sin(thetamodel*np.pi/180)
				xfit = xfit[np.where(yfit > 0)]
				yfit = yfit[np.where(yfit > 0)]
				axes.imshow(gradient.T,origin="lower")
				axes.scatter(xfit,yfit,color='r',marker='x',sizes=[15]*len(xfit))
				axes.axvline(xsplit,color='r',linestyle="dashed")
				axes.set_xlabel("x (nm)")
				axes.set_ylabel("z (nm)")
				axes.set_title(f"t={round(time_ns,3)}ns, θ={round(contact_angle,3)}°")
		elif args.mode == "dynamic":
			print(f"Fitting snapshot {i}")
			fig, axes = plt.subplots(1,2)
			# find x index of highest point
			# do a dense angular scan to find ground truth points, then given [(x,y),(x,y)...], find index of point with maximum y
			# use x corresponding to this point to split data for two circles and measure two contact angles
			thetascan = np.arange(0,181,0.05)
			xarc, yarc = angularscan(gradient,(x0,y0),thetascan,2)
			if (len(yarc) > 0) and (len(xarc) == len(yarc)):
				ymaxidx = np.argmax(yarc)
				xsplit = xarc[ymaxidx]
				# now do a broader scan from (xsplit, y0) on either side to obtain advancing and receding contact angle
				for j in [0,1]:
					thetascan = (-180*j) + np.arange(90,271,0.1)
					xarc, yarc = angularscan(gradient,(xsplit,y0),thetascan,2)
					arc = np.array((xarc,yarc)).T # for loss function
					r = np.add((xarc-xsplit)**2,(yarc-y0)**2)**0.5
					radius_guess = np.mean(r)
					rinit = np.mean(r)
					print(f"Initial guess: [{rinit}, {y0}]")
					[rfit, hfit] = basinhopping(arcLoss,[rinit,y0],niter=2000,T=0.7,stepsize=0.1,minimizer_kwargs={"bounds":[(0.25*rinit,4*rinit),(-2*rinit,2*rinit)]}).x
					print(f"Optimized guess: [{rfit}, {hfit}]")
					axes[j].scatter([xsplit],[hfit],color='r',marker='x')
					if hfit <= rfit: 
						contact_angle = 90 + (180/np.pi) * np.arcsin(hfit/rfit)
						# !! when hift > rfit
						if j == 0: 
							print("Forward")
							forward[i] = contact_angle
							print(f"theta: {contact_angle}")
							thetamodel = np.arange(90,270,8)
						else: 
							print("Backward")
							backward[i] = contact_angle	
							print(f"theta: {contact_angle}")	
							thetamodel = np.arange(-90,90,8)
						xfit = xsplit + rfit*np.cos(thetamodel*np.pi/180)
						yfit = hfit + rfit*np.sin(thetamodel*np.pi/180)
						xfit = xfit[np.where(yfit > 0)]
						yfit = yfit[np.where(yfit > 0)]
						axes[j].imshow(gradient.T,origin="lower")
						axes[j].scatter(xfit,yfit,color='r',marker='x',sizes=[15]*len(xfit))
						axes[j].axvline(xsplit,color='r',linestyle="dashed")
						axes[j].set_xlabel("x (nm)")
						axes[j].set_ylabel("z (nm)")
						axes[j].set_title(f"t={round(time_ns,3)}ns, θ={round(contact_angle,3)}°")
			else:
				pass
		pngfile = args.output.strip(".npy") + ".rho_" + str(i) + ".png"
		plt.savefig(pngfile,bbox_inches="tight")
		img = cv2.imread(pngfile)
		height, width, size = img.shape
		size = (width,height)
		img_array.append(img)
		plt.clf()
	# make video
	print(f"{round((time.time()-timestart)/60,4)} minutes to calculate contact angles")
	mp4file = args.output.strip(".npy") + ".arcs.mp4"
	out = cv2.VideoWriter(mp4file,cv2.VideoWriter_fourcc(*"mp4v"),5,size)
	for img in img_array: out.write(img)
	out.release()	
	# calculate hysteresis and write measurements to csv file
	times, forward, backward = np.array(times), np.array(forward), np.array(backward)
	hyst = forward - backward 
	csvfile = args.output.strip(".npy") + ".hyst.csv"
	with open(csvfile,'w') as outf:
		outf.write("time,forward,backward,hyst\n")
		for i in range(len(hyst)): outf.write(f"{times[i]},{forward[i]},{backward[i]},{hyst[i]}\n")
	# generate a single 3 panel plot with forward, backward, hysteresis as a function of time
	pngfile = args.output.strip(".npy") + ".hyst.png"
	plt.rcParams["figure.figsize"] = (12,6)
	fig = plt.figure()
	subfigs = fig.subfigures(2,1) # need matplotlib version >= 3.5.3 
	axes = subfigs[0].subplots(1,2,sharey=True)
	print(times)
	print(forward)
	print(backward)
	axes[0].scatter(times,forward,label="advancing")
	axes[0].plot(times,forward)
	axes[1].scatter(times,backward,label="receding")
	axes[1].plot(times,backward)
	for i in range(2): 
	    axes[i].grid()
	    axes[i].legend(loc="upper right")
	    axes[i].get_xaxis().set_visible(False)
	axes[0].set_ylabel("Contact angle (°)")
	axes = subfigs[1].subplots(1,1,sharey=True)
	axes.scatter(times,hyst)
	axes.plot(times,hyst)
	axes.grid()
	axes.set_xlabel("Sim time (ns)")
	axes.set_ylabel("Hysteresis (°)")
	axes.axhline(0,linestyle="dashed",color='k')
	plt.savefig(pngfile,bbox_inches="tight")
	#for fname in glob.iglob(args.output.strip(".npy") + ".rho_*.png"): os.system(f"rm {fname}") # delete generated images 
