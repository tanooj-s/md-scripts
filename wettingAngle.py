# this is the correct way to calculate wetting angles in a 2d geometry
# calculate densities, then fit a circle to arc data from edges of density
# equation of fit circle x**2 + (z-h)**2 = r**2 (centered on z axis)
# (fit parameters here are r and h)
# wetting angle is 90 + arccos(h/r)

import argparse
import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="calculate density profiles")
parser.add_argument('-i', action="store", dest="input")
parser.add_argument('-o', action="store", dest="output")
parser.add_argument('-dz', action="store", dest="dz") 
parser.add_argument('-dx', action="store", dest="dx") # density bin widths
parser.add_argument('-minz', action="store", dest="minz")
parser.add_argument('-maxz', action="store", dest="maxz")
parser.add_argument('-minx', action="store", dest="minx")
parser.add_argument('-maxx', action="store", dest="maxx")
parser.add_argument('-start', action="store", dest="start")
parser.add_argument('-t', action="store", dest="atom_types") # atom types to calculate density profiles for, string like '1 3 4'
args = parser.parse_args()
dz = float(args.dz)
dx = float(args.dx)
minz = float(args.minz)
maxz = float(args.maxz)
minx = float(args.minx)
maxx = float(args.maxx)
start = int(args.start)
atom_types = [int(t) for t in args.atom_types.split(' ')]
print(f'Atom types to bin: {atom_types}')

def purge(tokens): return [t for t in tokens if len(t) >= 1] # purge empty strings from token lists

def isfloat(s): # hack function to confirm tokens are numeric as floats
	try:
		float(s)
		return True
	except ValueError:
		return False

# ---- single pass over file ----
print(f'MinZ: {minz}')
print(f'MaxZ: {maxz}')
print(f'MinX: {minx}')
print(f'MaxX: {maxx}')
nbinsz = int((maxz-minz)/dz) + 1
nbinsx = int((maxx-minx)/dz) + 1

# infer dump format from first 10 lines
idIdx, typeIdx, xIdx, yIdx, zIdx = 0, 0, 0, 0, 0
with open(args.input,'r') as f:
	for _ in range(15):
		line = f.readline()
		line = line.strip('\n')
		if line.startswith("ITEM: ATOMS"):
			tokens = purge(line.split(' '))
			print(tokens)
			idIdx = tokens.index('id') - 2
			typeIdx = tokens.index('type') - 2
			xIdx = tokens.index('x') - 2
			yIdx = tokens.index('y') - 2
			zIdx = tokens.index('z') - 2
print(f'{idIdx} {typeIdx} {xIdx} {yIdx} {zIdx}')

# ---- parse data file for densities ----

densities = [] # trajectory of densities
# this will have shape (timesteps,nbinsx,nbinsz)
print("Parsing and analyzing dump directly to obtain 2D densities...")
timestart = time.time()
with open(args.input,'r') as f:
	doCollect = False
	nCollected = 0 # timesteps summed over
	currentTime = 0
	previousLine = '' # keep saving previous line 
	density_t = np.zeros((nbinsx,nbinsz)) # density at this timestep, in case you want a trajectory
	for line in tqdm(f):
		tokens = purge(line.strip('\n').split(' '))
		if doCollect == True: 
			# you only want to collect data when all tokens are numeric and are atomic data
			checksum = np.sum([not isfloat(t) for t in tokens])
			if (len(tokens) > 4) and (checksum == 0):
				aType = int(tokens[1])
				x = float(tokens[2])
				z = float(tokens[4])  # TODO: figure out what's going with token indexing, this shouldn't be hardcoded
				if (aType in atom_types) and (z >= minz) and (z <= maxz) and (x >= minx) and (x <= maxx):
					zIdx = int((z - minz)/dz)
					xIdx = int((x - minx)/dx)
					density_t[xIdx,zIdx] += 1 # density at this timestep
		if (currentTime >= start) and (previousLine.startswith('ITEM: ATOMS')):
			doCollect = True
		if previousLine.startswith('ITEM: TIMESTEP'):
			# normalize density at this timestep, append to trajectory, and reset density to zero for next timestep
			density_t /= np.sum(density_t)
			# discard this frame if there are any NaNs 
			if np.sum(np.isnan(density_t)) < 0.5:
				densities.append(density_t)
				nCollected += 1
			density_t = np.zeros((nbinsx,nbinsz))
			currentTime = int(tokens[0])
		previousLine = line
print(f'{round((time.time()-timestart)/60,4)} minutes')

densities = np.array(densities)
print(densities.shape)
# output time averaged density
rho = np.mean(densities,axis=0)
print(f'{nCollected} timesteps collected for time-averaged 2d density')
with open(args.output+'_rho.npy','wb') as f: np.save(f, rho)


# --------------------------------------------------------------------

# now calculate the time averaged arcs [z,xleft,xright] from densities
# iterate over trajectory of densities, find first x indices where rho is nonzero for each bin in z
# use these to obtain the wetting angle at each side
# this time-averaging should only happen over sufficiently equilibrated configurations 

(frames, nx, nz) = densities.shape
frames_collected = 0 
xleft = np.zeros(nz-1,)
xright = np.zeros(nz-1,)
print("Obtaining arcs at each relevant frame along trajectory...")
for t in tqdm(np.arange(0,frames,1)):
	field = densities[t] 
	z, xl, xr = [], [], []
	for zi in np.arange(0,nz,1): # iterate over row (z) indices
		row = field[:,zi]
		# find index of first nonzero density element on both sides
		nonzero = np.where(row != 0)[0] # need to grab first index of returned tuple for the actual data
		if len(nonzero) > 0:
			xlo = nonzero[0]
			xhi = nonzero[-1]
			z.append(zi * dz)
			xl.append(xlo * dx)
			xr.append(xhi * dx)
	z, xl, xr = np.array(z), np.array(xl), np.array(xr)
	xl -= xl[0]
	xr -= xr[0] # recenter
	xl *= -1 # flip left side of droplet arc for correct mean
	# pad these three with zeros so no issues with time averaging
	npad = nz-1-len(z)
	if npad <= 0:
		pass # discard this frame
	else:
		z = np.pad(z,(0,npad))
		xl = np.pad(xl,(0,npad))
		xr = np.pad(xr,(0,npad))
		xleft += xl
		xright += xr
		frames_collected += 1
print(f"Arcs collected from {frames_collected} out of {frames} frames")

# time average and average both sides
z = dz * np.arange(0,xleft.shape[0],1)
xleft /= frames_collected
xright /= frames_collected
x = 0.5 * (xleft+xright)

# now truncate at whichever index the arc comes back to x=0 
idx = np.where(x < 0)[0][0]
z = z[:idx]
x = x[:idx]
# output arc data in case you want more postprocessing in a notebook
angle_data = np.vstack((z,x))
with open(args.output+'_arc.npy','wb') as f: np.save(f, angle_data)


# --------------------------------------------------------------------

# now fit a circle to the arc data using least squares and calculate wetting angle 
xtrain, ztrain = x, z # rename bc least squares fit is MACHINE LEARNING

def circleResiduals(params,z,x):
    '''
    x(z) with two fit parameters r and h (radius of the circle and z coordinate of center)
    use squared values for residuals to not run into optimization issues
    '''
    r, h = params
    x2 = r**2 - (z-h)**2
    residual = x**2 - x2
    return residual

initial_guess = [np.mean(ztrain),np.mean(ztrain)] # this should be pretty flexible
res_lsq = least_squares(circleResiduals,initial_guess,loss='soft_l1',args=(ztrain,xtrain))
rfit, hfit = res_lsq.x
rfit, hfit = np.abs(rfit), np.abs(hfit) # make sure returned params are positive

if hfit <= rfit:
	# the actual number we care about
	theta = 90 + (180/np.pi) * np.arccos(hfit/rfit)
else:
	theta = 0 # default in case the circle isn't fit well
# note: for proper statistics and error bars this angle does need to be computed at every frame and not just from the time averaged values
# TODO: edit this script later to do so when production sims are being run

print(f"Fitted circle parameters: r={rfit}    |    h={hfit}")
print(f"Calculated wetting angle: {theta}")

# generate an image of the fitted circle 
xfit2 = rfit**2 - (ztrain-hfit)**2
xfit2 = xfit2[np.where(xfit2 >= 0)[0]]
ztrain = ztrain[np.where(xfit2 >= 0)[0]] # only use nonegative values for plotting 
xfit = np.sqrt(xfit2)

plt.plot(x,z,label='simulation data')
plt.scatter(x=xfit,y=ztrain,label=f'fitted circle with r={round(rfit,3)}, h={round(hfit,3)}',color='r',sizes=[5]*len(ztrain))
plt.title(f'Calculated wetting angle θ={round(theta,3)}')
plotbound = 20+np.max(np.hstack((ztrain,xfit)))
plt.xlim(-20,plotbound)
plt.ylim(-20,plotbound)
plt.xlabel('x (A)')
plt.ylabel('z (A)')
plt.axhline(0,color='k')
plt.axvline(0,color='k')
plt.grid()
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(args.output+'_circle.png')








