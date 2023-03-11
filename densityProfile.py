# read in a dump file, calculate z-resolved density profiles of specified atom types
# (useful for interfaces, responses to external potentials etc)
# output a numpy array of shape (1+nTypes, nbins)

import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="calculate density profiles")
parser.add_argument('-i', action="store", dest="input")
parser.add_argument('-o', action="store", dest="output")
parser.add_argument('-dz', action="store", dest="dz") # bin width 
parser.add_argument('-t', action="store", dest="atom_types") # atom types to calculate density profiles for, string like '1 3 4'
parser.add_argument('-f', action="store", dest="frac") # fraction of simulation to use (the system should be equilibrated along z for this entire duration)
args = parser.parse_args()

dz = float(args.dz)
frac = float(args.frac)
atom_types = [int(t) for t in args.atom_types.split(' ')]
print(f'Atom types to bin: {atom_types}')

def purge(tokens): return [t for t in tokens if len(t) >= 1] # purge empty strings from token lists


# ---- parse LAMMPS dump file ----
print("Parsing dump file...")
tsHeadIdxs = []
nHeadIdxs = []
boxHeadIdxs = []
atomHeadIdxs = []
with open(args.input,'r') as f: lines = f.readlines()
lines = [l.strip('\n') for l in lines]
linecounter = 0
for line in lines:
	if line.startswith("ITEM: TIMESTEP"): tsHeadIdxs.append(linecounter)
	if line.startswith("ITEM: NUMBER"): nHeadIdxs.append(linecounter)
	if line.startswith("ITEM: BOX BOUNDS"): boxHeadIdxs.append(linecounter)
	if line.startswith("ITEM: ATOMS"): atomHeadIdxs.append(linecounter)
	linecounter += 1

boxBoundLines = []
atomLines = []
nAtoms = int(lines[nHeadIdxs[0]+1]) # assume constant N
nUse = int(frac*len(tsHeadIdxs)) # choose length of trajectory to analyze, might want to make this a flag
tsHeadIdxs = tsHeadIdxs[-nUse:]
nHeadIdxs = nHeadIdxs[-nUse:]
boxHeadIdxs = boxHeadIdxs[-nUse:]
atomHeadIdxs = atomHeadIdxs[-nUse:]
print(f"Timesteps to average over: {len(tsHeadIdxs)}")
for idx in boxHeadIdxs:
	boxBoundLines.append(lines[idx+1:idx+4])
for idx in atomHeadIdxs:
	atomLines.append(lines[idx+1:idx+nAtoms+1])



#  ---- infer dump format from first atom header ----
atomHeader = lines[atomHeadIdxs[0]].split(' ')
idIdx = atomHeader.index('id') - 2
typeIdx = atomHeader.index('type') - 2
xIdx = atomHeader.index('x') - 2
yIdx = atomHeader.index('y') - 2
zIdx = atomHeader.index('z') - 2
print(f'Dump format: {atomHeader}')


# ------ initial pass to get nBins -----
print('Initial pass to determine number of bins...')
minZ, maxZ = 1000, -1000
for idx in tqdm(tsHeadIdxs):
	timestep = int(lines[idx+1])
	nAtoms = int(lines[idx+3])
	atomlines = lines[idx+9:idx+9+nAtoms]
	for line in atomlines:
		tokens = purge(line.split(' '))
		aType, z = int(tokens[typeIdx]), float(tokens[zIdx])
		if aType in atom_types:
			if (z < minZ): minZ = z
			elif (z > maxZ): maxZ = z

print(f'MinZ: {minZ}')
print(f'MaxZ: {maxZ}')
nBins = int((maxZ-minZ)/dz) + 1
densities = np.zeros((len(atom_types), nBins)) # output array
# print out which atom type is which row in output array
for i1, i2 in enumerate(atom_types):
	print(f'Row {i1+1} | Atom type {i2}') # 0 will be zs


# ----- second pass for actual binning -----
print('Second pass to calculate density profiles...')
for idx in tqdm(tsHeadIdxs):
	timestep = int(lines[idx+1])
	nAtoms = int(lines[idx+3])
	atomlines = lines[idx+9:idx+9+nAtoms]
	for line in atomlines:
		tokens = purge(line.split(' '))
		aType, z = int(tokens[typeIdx]), float(tokens[zIdx])
		if aType in atom_types:
			idx1 = atom_types.index(aType) # index along first axis of output array (i.e. atom type)
			binIdx = int((z - minZ)/dz)
			densities[idx1, binIdx] += 1
densities /= len(tsHeadIdxs) # normalize

# append z values (i.e. x axis of histograms) as first row
zs = np.arange(minZ,maxZ,dz)
assert len(zs) == densities.shape[1]
zs = np.reshape(zs, (1, densities.shape[1]))
densities = np.concatenate((zs,densities))
with open(args.output,'wb') as f: np.save(f, densities)
