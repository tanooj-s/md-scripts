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

parser = argparse.ArgumentParser(description="calculate density profiles")
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
	for i in range(1,q.shape[1]-1):
		dq[:,i] = 0.5 * ((q[:,i+1]-q[:,i]) + (q[:,i]-q[:,i-1]))
	dq[:,0] = 0.5 * ((q[:,1]-q[:,0]) + (q[:,0]-q[:,-1]))
	dq[:,-1] = 0.5 * ((q[:,0]-q[:,-1]) + (q[:,-1]-q[:,-2]))
	return dq # note this is in terms of grid spacings

def dQdX(q):
	dq = np.zeros(q.shape)
	for i in range(1,q.shape[0]-1):
		dq[i,:] = 0.5 * ((q[i+1,:]-q[i,:]) + (q[i,:]-q[i-1,:]))
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
	with open(args.input,'r') as f:
		nCollected = 0 # dump timesteps averaged over for this output snapshot
		currentTime = 0 # simulation timestep
		previousLine = ''
		for line in f:
			tokens = purge(line.strip('\n').split(' '))
			checksum = np.sum([not isfloat(t) for t in tokens]) 
			if line.startswith("ITEM: ATOMS"):
				typeIdx = tokens.index('type') - 2
				xIdx = tokens.index('x') - 2
				yIdx = tokens.index('y') - 2 # note different naming convention for histogram indices below
				zIdx = tokens.index('z') - 2
				#qIdx = tokens.index('q') - 2 # ... 
			if (len(tokens) == 5) and (checksum == 0): # first check to make sure box bound lines aren't being used
				aType = int(tokens[typeIdx])
				x = float(tokens[xIdx]) 
				z = float(tokens[zIdx])  
				if (aType in atom_types) and (z >= minz) and (z <= maxz) and (x >= minx) and (x <= maxx):			
					zidx = int((z-minz)/dz)
					xidx = int((x-minx)/dx)
					if type(tidx) is int: 
						try: 
							densities[tidx,xidx,zidx] += 1 # number density 
						except IndexError:
							pass 
			if previousLine.startswith('ITEM: TIMESTEP'):
				currentTime = int(tokens[0])
				if currentTime < start: 
					#print(f"Skipping data collection at t={currentTime}")
					tidx = None
				elif currentTime >= end:
					break
				else:
					tidx = int(np.floor((currentTime-start) / window))
					nCollected += 1	
					# reset normalization nCollected if next time window is hit
					if (currentTime-start) % window == 0:
						print(f"Averaging particle positions for snapshot index {tidx}...")	
						try: 
							densities[tidx,:,:] /= nCollected
						except IndexError:
							pass 
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
	print(f"{2*densities.shape[0]} contact angles to measure")
	# iterate over snapshots, measure forward, backward angles for each and save to output later
	# contact angle hysteresis calculated as per method in DOI: 10.1021/acs.langmuir.9b00551 
	forward = []
	backward = []
	times = [] # sim time ns
	img_array = [] # for video
	timestart = time.time()
	for i in range(densities.shape[0]):
		plt.rcParams['figure.figsize'] = (15,5)
		plt.rcParams['font.size'] = 20
		fig, axes = plt.subplots(1,2)
		time_ns = dt*(start+((i+1)*window)) # nanoseconds
		times.append(time_ns)
		field = densities[i]
		x0, y0 = COM(field)
		dqdx = dQdX(field)
		dqdy = dQdY(field)
		gradient = (dqdx**2 + dqdy**2) ** 0.5
		# find x index of highest point
		# do a dense angular scan to find ground truth points, then given [(x,y),(x,y)...], find index of point with maximum y
		# use x corresponding to this point to split data
		xarc, yarc = [], []
		thetascan = np.arange(0,181,0.05) 
		for t in thetascan:
			x = x0 + np.arange(0,int(0.4*gradient.shape[0]),1)*np.cos(t * (np.pi/180))
			y = y0 + np.arange(0,int(0.4*gradient.shape[0]),1)*np.sin(t * (np.pi/180)) # length of line segment is a bit hacky
			xline = np.array([int(a) for a in np.floor(x)])
			yline = np.array([int(a) for a in np.floor(y)])
			line = gradient[xline,yline] # gradient values along this line segment
			edge_idx = np.argmax(line)
			xarc.append(x0 + edge_idx * np.cos(t * (np.pi/180)))
			yarc.append(y0 + edge_idx * np.sin(t * (np.pi/180)))
		xarc, yarc = np.array(xarc), np.array(yarc)  
		ymaxidx = np.argmax(yarc)
		xsplit = xarc[ymaxidx]
		ground_truth = np.vstack((xarc,yarc))
		# now do a broader scan from (xsplit, y0) on either side to obtain advancing and receding contact angle
		for j in [0,1]:
			arc = []
			radii = []
			thetascan = (-180*j) + np.arange(91,271,0.1)
			for t in thetascan:
				x = xsplit + np.arange(0,int(0.4*gradient.shape[0]),1)*np.cos(t * (np.pi/180)) # !!
				y = y0 + np.arange(0,int(0.4*gradient.shape[0]),1)*np.sin(t * (np.pi/180))
				xline = np.array([int(a) for a in np.floor(x)])
				yline = np.array([int(a) for a in np.floor(y)])
				line = gradient[xline,yline]
				edge_idx = np.argmax(line)
				radii.append(edge_idx)
				xpoint = xsplit + edge_idx * np.cos(t * (np.pi/180)) # !!
				ypoint = y0 + edge_idx * np.sin(t * (np.pi/180))
				if ypoint > 2: # so that flat parts of profile near substrate aren't used for fits
					arc.append(np.array([xpoint,ypoint]))
			radius_guess = np.mean(radii)
			arc = np.array(arc)
			rinit, hinit = radius_guess, y0
			bounds = [(0.25*rinit,4*rinit),(0,2*rinit)]
			opt_result = basinhopping(arcLoss,[rinit,hinit],niter=500,T=0.7,stepsize=0.1,minimizer_kwargs={"bounds":bounds})
			[rfit,hfit] = opt_result.x
			axes[j].scatter([xsplit],[hfit],color='r',marker='x')
			contact_angle = 90 + (180/np.pi) * np.arcsin(hfit/rfit)
			if np.isnan(contact_angle): contact_angle = 0 # error handling 
			if j == 0: 
				forward.append(contact_angle)
			else: 
				backward.append(contact_angle)			
			thetamodel = (-90*j) + np.arange(60,211,8) # range of model to output
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
	forward, backward = np.array(forward), np.array(backward)
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
	for fname in glob.iglob(args.output.strip(".npy") + ".rho_*.png"): os.system(f"rm {fname}") # delete generated images 
