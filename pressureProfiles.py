# read in a LAMMPS interface dump file, calculate z-resolved pressures profiles
# (could be used for surface tension calculations)
# output a numpy array of shape (4, nbins), where rows are (z,pxx,pyy,pzz)
# needs a dump file with compute stress/atom per atom values

# (from LAMMPS documentation below)
# Note that as defined in the formula, per-atom stress is the negative of the per-atom pressure tensor. 
# It is also really a stress*volume formulation, meaning the computed quantity is in units of pressure*volume
# lammps stress tensor compute format xx yy zz xy xz yz

import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="calculate density profiles")
parser.add_argument('-i', action="store", dest="input")
parser.add_argument('-o', action="store", dest="output")
parser.add_argument('-dz', action="store", dest="dz") # bin width 
parser.add_argument('-minz', action="store", dest="minz")
parser.add_argument('-maxz', action="store", dest="maxz")
#parser.add_argument('-t', action="store", dest="atom_types") # atom types to calculate density profiles for, string like '1 3 4'
args = parser.parse_args()

dz = float(args.dz)
minz = float(args.minz)
maxz = float(args.maxz)
#atom_types = [int(t) for t in args.atom_types.split(' ')]
#print(f'Atom types to bin: {atom_types}')

def purge(tokens): return [t for t in tokens if len(t) >= 1] # purge empty strings from token lists


# ---- parse LAMMPS dump file ----
print("Parsing dump file...")
tsHeadIdxs = []
nHeadIdxs = []
boxHeadIdxs = []
atomHeadIdxs = []
with open(args.input,'r') as f: lines = f.readlines()
lines = [l.strip('\n') for l in tqdm(lines)]
linecounter = 0
for line in tqdm(lines):
	if line.startswith("ITEM: TIMESTEP"): tsHeadIdxs.append(linecounter)
	if line.startswith("ITEM: NUMBER"): nHeadIdxs.append(linecounter)
	if line.startswith("ITEM: BOX BOUNDS"): boxHeadIdxs.append(linecounter)
	if line.startswith("ITEM: ATOMS"): atomHeadIdxs.append(linecounter)
	linecounter += 1

boxBoundLines = []
atomLines = []
nAtoms = int(lines[nHeadIdxs[0]+1]) # assume constant N
nUse = int(0.5*len(tsHeadIdxs)) # choose length of trajectory to analyze, might want to make this a flag
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
# components of stress tensor
xxIdx = atomHeader.index('c_1[1]') - 2
yyIdx = atomHeader.index('c_1[2]') - 2
zzIdx = atomHeader.index('c_1[3]') - 2 # edit as appropriate for off diagonal components
print(f'Dump format: {atomHeader}')

# get box dimensions for volume normalization
xlo, xhi = float(lines[boxHeadIdxs[0]+1].split(' ')[0]), float(lines[boxHeadIdxs[0]+1].split(' ')[1])
ylo, yhi = float(lines[boxHeadIdxs[0]+2].split(' ')[0]), float(lines[boxHeadIdxs[0]+2].split(' ')[1])
zlo, zhi = float(lines[boxHeadIdxs[0]+3].split(' ')[0]), float(lines[boxHeadIdxs[0]+3].split(' ')[1])

print(f'xlo: {xlo} A| xhi: {xhi} A')
print(f'ylo: {ylo} A| yhi: {yhi} A')
print(f'zlo: {zlo} A| zhi: {zhi} A')

VA3 = (xhi-xlo)*(yhi-ylo)*(zhi-zlo) # volume in cubic angstroms
Vm3 = VA3 * 1e-30 # volume in m3
print(f'Box volume: {VA3} A^3')

# ------ get nbins -----
print(f'MinZ: {minz}')
print(f'MaxZ: {maxz}')
nbins = int((maxz-minz)/dz) + 1
pxx_z = np.zeros(nbins)
pyy_z = np.zeros(nbins)
pzz_z = np.zeros(nbins) # compute for entire system

# append z values (i.e. x axis of histograms) as first row
zs = dz*np.arange(0,pxx_z.shape[0],1)
assert zs.shape == pxx_z.shape
assert zs.shape == pyy_z.shape
assert zs.shape == pzz_z.shape

# ----- bin stress tensor computes -----
# need to compute (pxx(z),pyy(z),pzz(z))

print('Passing over trajectory to compute pressure profiles...')
for idx in tqdm(tsHeadIdxs):
	timestep = int(lines[idx+1])
	nAtoms = int(lines[idx+3])
	atomlines = lines[idx+9:idx+9+nAtoms]
	for line in atomlines:
		tokens = purge(line.split(' '))
		aType, z, pxx, pyy, pzz = int(tokens[typeIdx]), float(tokens[zIdx]), -float(tokens[xxIdx]), -float(tokens[yyIdx]), -float(tokens[zzIdx])
		# these are in units of atm*A3 (assuming real units)
		# first do per atom volume normalization to get units of atm
		for p_i in [pxx, pyy, pzz]: p_i /= (VA3*nbins) # confirm factor of 3 not needed for individual components
		for p_i in [pxx, pyy, pzz]: p_i *= 101325 # convert atm to Pa (edit as necessary)
		
		if ((z < maxz) and (z > minz)): # explicit check in case of weirdness
			binIdx = int((z - minz)/dz)
			pxx_z[binIdx] += pxx
			pyy_z[binIdx] += pyy
			pzz_z[binIdx] += pzz

pxx_z /= len(tsHeadIdxs) # normalize
pyy_z /= len(tsHeadIdxs)
pzz_z /= len(tsHeadIdxs)

pressures = np.vstack((zs,pxx_z,pyy_z,pzz_z)) 
with open(args.output,'wb') as f: np.save(f, pressures)

