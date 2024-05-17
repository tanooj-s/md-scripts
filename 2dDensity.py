
# output a 2d density profile from a LAMMPS dump file (e.g n(x,z))
# can modify this to q(x,z), bin other atomic quantities 

import argparse
import numpy as np
from tqdm import tqdm
import time

parser = argparse.ArgumentParser(description="calculate density profiles")
parser.add_argument('-i', action="store", dest="input")
parser.add_argument('-o', action="store", dest="output") # numpy file!!!!!!!!!
parser.add_argument('-dz', action="store", dest="dz") 
parser.add_argument('-dx', action="store", dest="dx") # bin widths
parser.add_argument('-minz', action="store", dest="minz")
parser.add_argument('-maxz', action="store", dest="maxz")
parser.add_argument('-minx', action="store", dest="minx")
parser.add_argument('-maxx', action="store", dest="maxx")
parser.add_argument('-start', action="store", dest="start")
parser.add_argument('-t', action="store", dest="atom_types") # atom types to use for density profiles, string like '1 3 4'
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
nbinsx = int((maxx-minx)/dx) + 1
#assert nbinsz == nbinsx
print(f"nbinsz: {nbinsz}")
print(f"nbinsx: {nbinsx}")

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

densities = [] # this will have shape (timesteps,nbinsx,nbinsz)
print("Parsing and analyzing dump directly to obtain scalar field (number density) on a grid...")
timestart = time.time()
with open(args.input,'r') as f:
	doCollect = False
	nCollected = 0 # timesteps summed over
	currentTime = 0
	previousLine = '' # keep saving previous line 
	density_t = np.zeros((nbinsx,nbinsz)) # density at this timestep
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
					density_t[xIdx,zIdx] += 1
		if (currentTime >= start) and (previousLine.startswith('ITEM: ATOMS')):
			doCollect = True
		if previousLine.startswith('ITEM: TIMESTEP'):
			if np.sum(density_t) > 0:
				density_t /= np.sum(density_t) 
				densities.append(density_t)
				nCollected += 1
			currentTime = int(tokens[0])		
			density_t = np.zeros((nbinsx,nbinsz))
		previousLine = line

print(f"{round((time.time()-timestart)/60,4)} minutes to obtain density from {nCollected} timesteps")
densities = np.array(densities)
print("Trajectory of densities:")
print(densities.shape)
assert len(densities.shape) == 3
#assert densities.shape[1] == densities.shape[2]

density = np.mean(densities[1:,:,:],axis=0) # don't use the first in case NaNs muck everything up (hack)
print("Time averaged")
print(density.shape)
print(f'{nCollected} timesteps averaged over for 2d density')
with open(args.output,'wb') as f: np.save(f, density)

