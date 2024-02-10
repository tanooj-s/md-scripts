# this is the correct(?) way to calculate wetting angles in a 2d geometry
# calculate densities, then obtain contours without any edge detection kernel shenanigans

import argparse
import numpy as np
from tqdm import tqdm
import time

parser = argparse.ArgumentParser(description="calculate density profiles")
parser.add_argument('-i', action="store", dest="input")
parser.add_argument('-o', action="store", dest="output")
parser.add_argument('-dz', action="store", dest="dz") 
parser.add_argument('-dx', action="store", dest="dx") # bin widths
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

# ---- parse data file ----

densities = []
# this will have shape (timesteps,nbinsX,nbinsZ)
print("Parsing and analyzing dump directly to obtain 2D densities...")
timestart = time.time()
with open(args.input,'r') as f:
	doCollect = False
	nCollected = 0 # timesteps summed over
	currentTime = 0
	previousLine = '' # keep saving previous line 
	density_t = np.zeros((nbinsx,nbinsz)) # density at this timestep
	for line in f:
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
					density_t[xIdx,zIdx] += 1
		if (currentTime >= start) and (previousLine.startswith('ITEM: ATOMS')):
			doCollect = True
			nCollected += 1
		if previousLine.startswith('ITEM: TIMESTEP'):
			# normalize density at this timestep, append to trajectory, and reset density to zero for next timestep
			density_t /= np.sum(density_t)
			densities.append(density_t)
			density_t = np.zeros((nbinsx,nbinsz))
			currentTime = int(tokens[0])
			print(f"TIMESTEP: {currentTime}")
		previousLine = line

print(f'{round((time.time()-timestart)/60,4)} minutes')
densities = np.array(densities)
print(densities.shape)

print(f'{nCollected} timesteps collected for 2d densities')
with open(args.output+'rho.npy','wb') as f: np.save(f, densities)

# --------------------------------------------------------------------

# now iterate over trajectory of densities, find first x indices where rho is nonzero for each bin in z
# use these to obtain the wetting angle at each side
# this time-averaging should only happen over sufficiently equilibrated configurations 

(frames, nx, nz) = densities.shape
# time-averaged angle as a function of z on each side
theta_l = np.zeros(nz-1,)
theta_r = np.zeros(nz-1,) # minus 1 to not run into issues for later arctan at z=0
frames_collected = 0 # only compute angle data for non-problematic frames (discard frames that have padding issues, arctan issues etc)

print("Computing tangents and corresponding angles for density at each requested frame along trajectory...")
for t in np.arange(0,frames,1):
	print(f"frame {t}")
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
	xl *= -1 # flip left side of droplet arc for correct arctan values
	# pad these three with zeros so no issues with time averaging
	npad = nz-1-len(z)
	if npad <= 0:
		pass # don't even bother with this frame
	else:
		z = np.pad(z,(0,npad))
		xl = np.pad(xl,(0,npad))
		xr = np.pad(xr,(0,npad))
		# do this so we don't run into annoying issues when arctan(0/0) instead of messing around with indexing
		tangent_l = np.divide(z,xl,out=np.zeros_like(z), where=xl!=0)
		tangent_r = np.divide(z,xr,out=np.zeros_like(z), where=xr!=0)
		# compute angle at this timestep as a function of z for both sides
		theta_l += 180 - (180/np.pi) * np.arctan(tangent_l)
		theta_r += 180 - (180/np.pi) * np.arctan(tangent_r) # wetting angle is the other side of tangent for this triangle
		frames_collected += 1

print(f"Angle data computed from {frames_collected} out of {frames} frames")

# time average
theta_l /= frames_collected
theta_r /= frames_collected
z = dz * np.arange(0,theta_l.shape[0],1)
# really should output the trajectory of computed angles and do time averaging later

# output angle data as shape (nz,3) with [zi,theta_l,theta_r] for each value of z
angle_data = np.vstack((z,theta_l,theta_r))
with open(args.output+'theta.npy','wb') as f: np.save(f, angle_data)







