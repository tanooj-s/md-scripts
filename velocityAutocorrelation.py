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
parser.add_argument('-M', action="store", dest="M") # how many windows to average over
parser.add_argument('-ts', action="store", dest="ts") # LAMMPS timestep in femtoseconds - this can't be inferred, must be specified by user (should be in LAMMPS input file)
parser.add_argument('-t', action="store", dest="atom_types") # string of atom types to collect data for
parser.add_argument('-norm', action="store", dest="norm") # calcualate normalized or unnormalized vacf, either 'y' or 'n'
args = parser.parse_args()

window = float(args.window)
M = int(args.M)
ts = float(args.ts)
atom_types = [int(t) for t in args.atom_types.split(' ')]
norm = args.norm

assert norm in ['y','n']

def purge(tokens): return [t for t in tokens if len(t) >= 1] # purge empty strings from token lists

def correlate_normalized(t0_vels, t1_vels):
	'''
	cvv = v(0).v(t) / v(0).v(0)
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

def correlate_unnormalized(t0_vels, t1_vels):
	'''
	cvv = v(0).v(t) 
	(use this version for calculting diffusion coefficients)
	assume inputs are (N, 4) shaped tuples of velocities of every atom of a given type 
	(type, vx, vy, vz)
	need to ensure that atoms are ordered in the same way here before passing to this function

	'''
	t0_vels, t1_vels = t0_vels[:,1:], t1_vels[:,1:] # remove atom types
	assert t0_vels.shape[1] == t1_vels.shape[1] == 3
	N = t0_vels.shape[0] # need to normalize by number of particles in this case
	t0_vels = np.reshape(t0_vels,(t0_vels.shape[0]*t0_vels.shape[1]))
	t1_vels = np.reshape(t1_vels,(t1_vels.shape[0]*t1_vels.shape[1])) # reshape to 1D arrays to simplify dot product
	corr = np.dot(t0_vels,t1_vels) / N
	return corr


# ---- parse LAMMPS dump file ----
tsHeadIdxs = []
nHeadIdxs = []
boxHeadIdxs = []
atomHeadIdxs = []
print("Reading dump file...")
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
nAtoms = int(lines[nHeadIdxs[0]+1]) # assume constant N, need this for unnormalized VACF computation
lineIdxs = [i+1 for i in tsHeadIdxs] # array with line index corresponding to each timestep

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
print(f'Atom types to calculate time correlations for: {atom_types}')
print(f'Windows to average over: {M}')
print(f'Length of a given window: {window} picoseconds')
print(f'Length of a given window in simulation timesteps: {window * 1000 / ts}')

timestep0, timestep1 = int(lines[lineIdxs[0]]), int(lines[lineIdxs[1]])
last_step = int(lines[lineIdxs[-1]])
dump_interval = timestep1 - timestep0
dt_dump = dump_interval * ts # in femtoseconds 
collect_window = int(window * 1000 / dt_dump) # window to compute correlations over

print(f'Length of a single window in dump frames: {window * 1000 / (ts * dump_interval)}')
print(f'Last timestep: {last_step}')
print(f'LAMMPS dump interval: {dump_interval} timesteps')
print(f'LAMMPS dump interval (sim time): {dt_dump} fs')
print(f'Frames to correlate for a single window: {collect_window} ') # number of frames to compute correlations over 
print(f'Total number of frames available to access data from: {len(lineIdxs)}')
print(f'{M} windows with {collect_window} frames to correlate over')

vacf = [] # output data, cvv(t) for each requested atom type

# ---- collect data for each window to average over ----
for i in range(M):	
	idx0 = len(lineIdxs) - (M-i)*collect_window
	idxi = len(lineIdxs) - (M-i-1)*collect_window # this has to be in units of frames, not physical time
	print(f'Window {i+1}: frame number {idx0} to frame number {idxi}')

	cvv = [] # autocorrelation functions for each atom type, for this specific window
	for aType in atom_types: cvv.append([])
	
	# ---- iterate over pairs of timesteps within each window ----
	for idxj in tqdm(np.arange(idx0,idxi,1)):
		#print(f'Correlating frame {idx0} and {idxj}')

		atomlines_t0 = lines[lineIdxs[idx0]+8 : lineIdxs[idx0]+8+nAtoms]
		atomlines_tj = lines[lineIdxs[idxj]+8 : lineIdxs[idxj]+8+nAtoms] 
		
		t0_dict, tj_dict = {}, {}
		# data for atoms to "remember" for each pair of timesteps

		# TODO: add z for striated data collection along interface normal
		for line in atomlines_t0:
			tokens = purge(line.split(' '))
			aType = int(tokens[typeIdx])
			if aType in atom_types: t0_dict[int(tokens[idIdx])] = np.array([aType, 
																			float(tokens[vxIdx]), 
																			float(tokens[vyIdx]), 
																			float(tokens[vzIdx])])
		for line in atomlines_tj:
			tokens = purge(line.split(' '))
			aType = int(tokens[typeIdx])
			if aType in atom_types: tj_dict[int(tokens[idIdx])] = np.array([aType, 
																			float(tokens[vxIdx]), 
																			float(tokens[vyIdx]), 
																			float(tokens[vzIdx])])
		# TODO: binning by z
		t0_vels = np.array(list(t0_dict.values()))
		tj_vels = np.array(list(tj_dict.values()))   
		# now compute correlations by type 
		for i in range(len(atom_types)):
			aType = atom_types[i]
			if norm == 'y':
				cvv[i].append(correlate_normalized(t0_vels[np.where(t0_vels[:,0] == aType)],tj_vels[np.where(tj_vels[:,0] == aType)]))
			elif norm == 'n':
				cvv[i].append(correlate_unnormalized(t0_vels[np.where(t0_vels[:,0] == aType)],tj_vels[np.where(tj_vels[:,0] == aType)]))
	cvv = np.array(cvv)
	vacf.append(cvv)
vacf = np.array(vacf) 

# average over M windows
print("Averaging over windows...")
vacf = np.sum(vacf,axis=0) / M

# append net VACF averaged over all types at end
# TODO: make this logic more sophisticated (only average over certain types for net e.g. when using coreshell models)
# ALSO: this needs to be properly weighted by the number of particles of each type for net
print("Averaging over atom types for net VACF...")
net = np.sum(vacf,axis=0) / vacf.shape[0]
net = np.reshape(net, (1, vacf.shape[1]))
vacf = np.concatenate((vacf,net))

# append a time axis at the front
times = np.arange(0,vacf.shape[1],1) * (dt_dump/1000) # picoseconds
assert len(times) == vacf.shape[1]
times = np.reshape(times, (1, vacf.shape[1]))
vacf = np.concatenate((times,vacf))

print(f"Output matrix shape: {vacf.shape}")
print(f"Columns in output data")
print(f"Time | VACF_1 | VACF_2 | ... ... | VACF_NET")

with open(f'{args.output}','wb') as f: np.save(f,vacf) 
