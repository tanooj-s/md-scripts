# calculate velocity autocorrelation functions from md simulation output

# ==== logistics ====
# need an outer loop for reference timesteps (because we would want to average this function over multiple windows)
# need an inner loop over each timestep within each window

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

window = float(args.window)
M = int(args.M)
ts = float(args.ts)
atom_types = [int(t) for t in args.atom_types.split(' ')]
#atom_type = int(args.atom_types) # for now just assume a single atom type requested

print(f'Atom types to calculate time correlations for: {atom_types}')
print(f'Windows to average over: {M}')
print(f'Length of a single window: {window} picoseconds')

def purge(tokens): return [t for t in tokens if len(t) >= 1] # purge empty strings from token lists

def correlate(t0_vels, t1_vels):
	'''
	assume inputs are (N, 4) shaped tuples of velocities of every atom of a given type 
	(type, vx, vy, vz)
	need to ensure that atoms are ordered in the same way here before passing to this function
	'''
	t0_vels, t1_vels = t0_vels[:,1:], t1_vels[:,1:] # remove atom types
	assert t0_vels.shape[1] == t1_vels.shape[1] == 3
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
lines = [l.strip('\n') for l in tqdm(lines)]
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

# ---- loop over intervals ----
for i in range(M):	
	t0 = last_step - (M-i)*collect_window*dump_interval
	ti = last_step - (M-i-1)*collect_window*dump_interval
	print(f'Window {i+1}: timestep {t0} to timestep {ti}')

	cvv = [] # autocorrelation functions for each atom type, for this specific interval
	for aType in atom_types: cvv.append([])
	for tj in t0+np.arange(0,ti-t0+1,dump_interval): # compute correlations with t0 at every timestep within this interval
		#print(f'Correlating {t0} and {tj}')
		# grab line indices of atom data corresponding to each relevant timetep
		atomlines_t0 =  lines[tsIdxMap[t0]+8:tsIdxMap[t0]+8+nAtoms]
		atomlines_tj =  lines[tsIdxMap[tj]+8:tsIdxMap[tj]+8+nAtoms] 
		t0_dict, tj_dict = {}, {}
		for line in atomlines_t0:
			tokens = purge(line.split(' '))
			aType = int(tokens[typeIdx])
			if aType in atom_types: t0_dict[int(tokens[idIdx])] = np.array([aType, float(tokens[vxIdx]), float(tokens[vyIdx]), float(tokens[vzIdx])])
		for line in atomlines_tj:
			tokens = purge(line.split(' '))
			aType = int(tokens[typeIdx])
			if aType in atom_types: tj_dict[int(tokens[idIdx])] = np.array([aType, float(tokens[vxIdx]), float(tokens[vyIdx]), float(tokens[vzIdx])])
		# (LAMMPS does make sure atom data is already ordered correctly at every timestep here, but could be pedantic and add an assertion later)
		t0_vels = np.array(list(t0_dict.values()))
		tj_vels = np.array(list(tj_dict.values()))   
		# now calculate correlations by type 
		for i in range(len(atom_types)):
			aType = atom_types[i]
			cvv[i].append(correlate(t0_vels[np.where(t0_vels[:,0] == aType)],tj_vels[np.where(tj_vels[:,0] == aType)]))
	cvv = np.array(cvv)
	vacf.append(cvv)
vacf = np.array(vacf) 

# average over M collections
print("Averaging over windows...")
vacf = np.sum(vacf,axis=0) / M

# append net VACF averaged over all types at end
print("Averaging over atom types for net VACF...")
net = np.sum(vacf,axis=0) / vacf.shape[0]
net = np.reshape(net, (1, vacf.shape[1]))
vacf = np.concatenate((vacf,net))

# append a time axis at the front
times = np.arange(0,vacf.shape[1],1) * (dt_dump/1000) # picoseconds
assert len(times) == vacf.shape[1]
times = np.reshape(times, (1, vacf.shape[1]))
vacf = np.concatenate((times,vacf))
with open(f'{args.output}','wb') as f: np.save(f,vacf) 






