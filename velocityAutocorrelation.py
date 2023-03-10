# calculate velocity autocorrelation functions from md simulation output

# ==== logistics ====
# need an outer loop for reference timesteps (because we would want to average this function over multiple windows)
# need an inner loop over each timestep within each window

# TODO: add support for simultaneous computation of multiple atom types later

import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='compute autocorrelations from a lammps dump file and output to numpy')
parser.add_argument('-i', action="store", dest="input") # dump file with granular output (should dump at least every 10 timesteps for smooth curves)
parser.add_argument('-o', action="store", dest="output") # output numpy file
parser.add_argument('-w', action="store", dest="window") # window over which to calculate autocorrelations, in picoseconds 
parser.add_argument('-M', action="store", dest="M") # how many windows to compute
parser.add_argument('-ts', action="store", dest="ts") # LAMMPS timestep in femtoseconds - this can't be inferred, must be specified by user (should be in LAMMPS input file)
parser.add_argument('-t', action="store", dest="atom_types") # string of atom types to collect data for
args = parser.parse_args()

window = int(args.window)
M = int(args.M)
ts = float(args.ts)
atom_types = [int(t) for t in args.atom_types.split(' ')]

def purge(tokens): return [t for t in tokens if len(t) >= 1] # purge empty strings from token lists

def correlate(t0_vels, t1_vels):
	'''
	assume inputs are (N, 3) shaped tuples of velocities of every atom of a given type 
	need to ensure that atoms are ordered in the same way here before passing to this function
	'''
	assert t0_vels.shape == t1_vels.shape
	t0_vels = np.reshape(t0_vels,(t0_vels.shape[0]*t0_vels.shape[1]))
	t1_vels = np.reshape(t1_vels,(t1_vels.shape[0]*t1_vels.shape[1])) # reshape to 1D arrays to simplify dot product
	corr = np.dot(t0_vels,t1_vels) / np.dot(t0_vels, t0_vels)
	return corr


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
print(f"Timesteps to average over: {len(tsHeadIdxs)}")
timestepIdxs = [i+1 for i in tsHeadIdxs]

for idx in boxHeadIdxs: # don't need box data here
	boxBoundLines.append(lines[idx+1:idx+4])
for idx in atomHeadIdxs:
	atomLines.append(lines[idx+1:idx+nAtoms+1])


# ---- infer dump format and collection interval ----
atomHeader = lines[atomHeadIdxs[0]].split(' ')
idIdx = atomHeader.index('id') - 2
typeIdx = atomHeader.index('type') - 2
vxIdx = atomHeader.index('vx') - 2
vyIdx = atomHeader.index('vy') - 2
vzIdx = atomHeader.index('vz') - 2
print(f'Dump format: {atomHeader}')


# ---- relevant time computations ----
timestep0, timestep1 = int(lines[timestepIdxs[0]]), int(lines[timestepIdxs[1]])
last_step = int(lines[timestepIdxs[-1]])
dump_interval = timestep1 - timestep0
dt_dump = dump_interval * ts # in femtoseconds 
collect_window = int(window * 1000 / dt_dump) # window to compute correlations over

print(f'LAMMPS dump interval: {dump_interval} timesteps')
print(f'LAMMPS dump interval (sim time): {dt_dump} fs')
print(f'Frames to correlate for a single window: {collect_window} ') # given window in picoseconds
print(f'Last timestep: {last_step}')


# ---- map each timestep to its line index ----
tsIdxMap = {}
for i in timestepIdxs: tsIdxMap[int(lines[i])] = i

vacf = [] # list of autocorrelation functions calculated for each window, what's going to be output
print(f'{M} windows with {collect_window} timesteps to correlate over')


# ---- loop over windows ----
for i in range(M):	
	t0 = last_step - (M-i)*collect_window
	ti = last_step - (M-i-1)*collect_window
	print(f'Window {i}: timestep {t0} to timestep {ti}')

	cvv = [] # autocorrelation function for this specific interval/window

	for tj in t0+np.arange(0,ti-t0+1,dump_interval): # compute correlations with t0 at every timestep within this interval
		print(f'Correlating {t0} and {tj}')

		# grab line indices of atom data corresponding to each relevant timetep
		atomlines_t0 =  lines[tsIdxMap[t0]+9:tsIdxMap[t0]+9+nAtoms-1]
		atomlines_tj =  lines[tsIdxMap[tj]+9:tsIdxMap[tj]+9+nAtoms-1] 

		# make dictionaries of atom data at each timestep with {aID: [vx,vy,vz]}
		# this makes it easier to later make sure data at each timestep is ordered the same way when calculating correlations 
		# TODO: add aType to dict values

		t0_dict = {}
		for line in atomlines_t0:
			tokens = purge(line.split(' '))
			t0_dict[int(tokens[idIdx])] = np.array([float(tokens[vxIdx]), float(tokens[vyIdx]), float(tokens[vzIdx])])

		tj_dict = {}
		for line in atomlines_tj:
			tokens = purge(line.split(' '))
			tj_dict[int(tokens[idIdx])] = np.array([float(tokens[vxIdx]), float(tokens[vyIdx]), float(tokens[vzIdx])])

		# (LAMMPS does make sure atom data is already ordered correctly at every timestep here, but could be pedantic and add an assertion later)

		t0_vels = np.array(list(t0_dict.values()))
		tj_vels = np.array(list(tj_dict.values()))
		cvv.append(correlate(t0_vels,tj_vels))

	cvv = np.array(cvv)
	vacf.append(cvv)
	#print(cvv)
	#print(cvv.shape)

vacf = np.array(vacf) 

# append a time axis at the front
times = dt_dump * np.arange(0,vacf.shape[1],1) # femtoseconds 
assert len(times) == vacf.shape[1]
times = np.reshape(times, (1, vacf.shape[1]))
vacf = np.concatenate((times,vacf))
with open(f'{args.output}','wb') as f: np.save(f,vacf) 

# can average over the M separate windows in a notebook later
# with open('vacf.npy','rb') as f: vacf = np.load(f)
# plt.plot(vacf[0],np.mean(vacf[1:],axis=0)) note x axis will be femtoseconds














































	#print()
	#for idx in headIdxs:
	#	timestep = int(lines[idx+1])
	#	print(timestep)


exit()


nUse = int(0.5*len(tsHeadIdxs)) # choose length of trajectory to analyze, might want to make this a flag
tsHeadIdxs = tsHeadIdxs[startCollect:endCollect]
nHeadIdxs = nHeadIdxs[startCollect:endCollect]
boxHeadIdxs = boxHeadIdxs[startCollect:endCollect]
atomHeadIdxs = atomHeadIdxs[startCollect:endCollect]





vProfiles = np.zeros((1+len(atom_types), nBins)) # output array
autocorrelations = np.zeros((1+len(atom_types), nBins)) # array of densities, needed for normalization
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
