# calculate z-resolved velocity profiles for flow sims 
# (useful for nanofluidics, slit-pore type sims, viscosity calculations etc)
# outputs 2 numpy arrays of shape (1+nTypes, nbins)
# one is density profile of each atom type
# the other is the normalized velocity profile of each atom type

# needs lammps sim output using unwrapped coordinates
# needs a user specified time interval for velocity calculations 
# calculate velocity for each atom as displacement / dt
# (this removes noise due to thermostatting)

# ==== logistics ====
# need unwrapped atomic coordinates at current and previous timesteps
# create dictionaries indexed by atom IDs to make this easy
# need to also compute number densities to normalize velocity profiles
# in initial pass infer nbins and dump interval in terms of timesteps
# use that and user specified dt for which windows to actually use 


import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='grab velocity data from lammps dumps and output to numpy')
parser.add_argument('-i', action="store", dest="input") # dump file with unwrapped coordinates
parser.add_argument('-o', action="store", dest="output") # output numpy filename format
parser.add_argument('-dz', action="store", dest="dz") # histogram resolution
parser.add_argument('-dt', action="store", dest="dt") # dt to use for velocity computations in femtoseconds 
parser.add_argument('-ts', action="store", dest="ts") # LAMMPS timestep in femtoseconds - this can't be inferred, must be specified by user (should be in LAMMPS input file)
parser.add_argument('-t', action="store", dest="atom_types") # string of atom types to collect data for
args = parser.parse_args()

dz = float(args.dz)
dt = float(args.dt)
ts = float(args.ts)
atom_types = [int(t) for t in args.atom_types.split(' ')]

def purge(tokens): return [t for t in tokens if len(t) >= 1] # purge empty strings from token lists

# ---- parse LAMMPS dump file ----
print("Parsing dump file...")
tsHeadIdxs = []
nHeadIdxs = []
boxHeadIdxs = []
atomHeadIdxs = []
with open(args.input,'r') as f: lines = f.readlines()
lines = [l.strip('\n') for l in lines]
print("Initial pass to get relevant line indices")
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
tsHeadIdxs = tsHeadIdxs[nUse:]
nHeadIdxs = nHeadIdxs[nUse:]
boxHeadIdxs = boxHeadIdxs[nUse:]
atomHeadIdxs = atomHeadIdxs[nUse:]
print(f"Timesteps to average over: {len(tsHeadIdxs)}")
for idx in boxHeadIdxs:
	boxBoundLines.append(lines[idx+1:idx+4])
for idx in atomHeadIdxs:
	atomLines.append(lines[idx+1:idx+nAtoms+1])


# ---- infer dump format and collection interval ----
atomHeader = lines[atomHeadIdxs[0]].split(' ')
idIdx = atomHeader.index('id') - 2
typeIdx = atomHeader.index('type') - 2
xuIdx = atomHeader.index('xu') - 2
yuIdx = atomHeader.index('yu') - 2
zuIdx = atomHeader.index('zu') - 2
print(f'Dump format: {atomHeader}')

timestep0, timestep1 = int(lines[tsHeadIdxs[0]+1]), int(lines[tsHeadIdxs[1]+1])
dump_interval = timestep1 - timestep0
dt_dump = dump_interval * ts # in femtoseconds 

print(f'LAMMPS dump interval: {dump_interval} timesteps')
print(f'LAMMPS dump interval (sim time): {dt_dump} fs')

collect_window = int(dt / ts) # window to compute displacements over
print(f'Collection interval for velocity computations: {collect_window} timesteps')
assert collect_window >= dump_interval # otherwise this won't work

# ---- initial pass to get nBins ----
print('Initial pass to determine number of bins...')
minZ, maxZ = 1000, -1000
for idx in tqdm(tsHeadIdxs):
	timestep = int(lines[idx+1])
	nAtoms = int(lines[idx+3])
	atomlines = lines[idx+9:idx+9+nAtoms]
	for line in atomlines:
		tokens = purge(line.split(' '))
		aType, z = int(tokens[typeIdx]), float(tokens[zuIdx])
		if aType in atom_types:
			if (z < minZ): minZ = z
			elif (z > maxZ): maxZ = z
print(f'MinZ: {minZ}')
print(f'MaxZ: {maxZ}')
nBins = int((maxZ-minZ)/dz) + 1

vProfiles = np.zeros((1+len(atom_types), nBins)) # output array
densities = np.zeros((1+len(atom_types), nBins)) # array of densities, needed for normalization
# (unnormalized velocity profiles should mostly mirror density if uniform flow across z)
# print out which atom type is which row in output array
for i1, i2 in enumerate(atom_types):
	print(f'Row {i1+1} | Atom type {i2}') # 0 will be zs
# last row is net velocity profile
print(f'Row {1+len(atom_types)} | Atom type NET')



# ---- compute velocity profiles ----
# create two line maps with current and previous atomic data at each timestep
# index dictionaries by atom IDs
# compute velocity for each atom as delY / delT

delT = dt * 1e-15 # convert to seconds 
nCollected = 0 # for normalization
print("Computing velocities from unwrapped displacements...")
for idx in tqdm(tsHeadIdxs): # line indices being iterated over
	timestep = int(lines[idx+1])

	if timestep % collect_window == 0:
		nAtoms = int(lines[idx+3])
		atomlines = lines[idx+9:idx+9+nAtoms] 

		if timestep == timestep0:
			lineMapCurr = {}
			for line in atomlines: # atomlines is just for this timestep
				tokens = purge(line.split(' '))
				atomID = int(tokens[idIdx])
				lineMapCurr[atomID] = line

		elif timestep > timestep0:
			nCollected += 1
			lineMapPrev = lineMapCurr
			lineMapCurr = {} # reset

			for line in atomlines: 
				tokens = purge(line.strip('\n').split(' '))
				atomID = int(tokens[idIdx])
				lineMapCurr[atomID] = line

			# after linemap has been built out, iterate through keys of dict to collect relevant data for each atom
			# i.e. iterate over atom IDs in current and previous timesteps
			for atomID in lineMapCurr.keys():
				tokensCurr = purge(lineMapCurr[atomID].split(' '))
				tokensPrev = purge(lineMapPrev[atomID].split(' '))
				aType = int(tokensCurr[typeIdx])
				if aType in atom_types:
					# confirm same atom ID across timesteps
					aIDCurr, aIDPrev = int(tokensCurr[idIdx]), int(tokensPrev[idIdx])
					assert aIDCurr == aIDPrev

					currZ, prevZ = float(tokensCurr[zuIdx]), float(tokensPrev[zuIdx])
					currYu, prevYu = float(tokensCurr[yuIdx]), float(tokensPrev[yuIdx])
					# assuming flow is along y

					# assign histogram bin based on mean of z displacement across timesteps
					z = 0.5*(currZ+prevZ)
					binIdx = int((z-minZ)/dz)

					delY = currYu - prevYu # in angstroms
					delY *= 1e-10 # angstroms -> meters 
					v = delY / delT # should be in m/s now

					# assign to profile of specific atom type
					idx1 = atom_types.index(aType) # index along first axis of output array (i.e. atom type)
					binIdx = int((z - minZ)/dz)
					vProfiles[idx1,binIdx] += v
					densities[idx1,binIdx] += 1

					# assign to net profile regardless of atom type
					vProfiles[-1,binIdx] += v
					densities[-1,binIdx] += 1

print(f"Number of delta displacement collections: {nCollected}")

# ---------------------------------------------------------------------


# normalize
densities /= nCollected
vProfiles /= nCollected
assert densities.shape == vProfiles.shape
vProfiles = np.divide(vProfiles, densities, out=np.zeros_like(densities), where=densities!=0)

# append z values (i.e. x axis of histograms) as first row
zs = np.arange(minZ,maxZ,dz)
assert len(zs) == densities.shape[1]
assert len(zs) == vProfiles.shape[1]
zs = np.reshape(zs, (1, densities.shape[1]))
densities = np.concatenate((zs,densities))
vProfiles = np.concatenate((zs,vProfiles))

with open(f'rho-{args.output}','wb') as f: np.save(f,densities)
with open(f'v-{args.output}','wb') as f: np.save(f,vProfiles)
