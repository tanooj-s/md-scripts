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
args = parser.parse_args()

window = float(args.window)
M = int(args.M)
ts = float(args.ts)
atom_types = [int(t) for t in args.atom_types.split(' ')]
#atom_type = int(args.atom_types) # for now just assume a single atom type requested

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
nAtoms = int(lines[nHeadIdxs[0]+1]) # assume constant N
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
print(f'Length of a given window in femtoseconds: {window * 1000} femtoseconds')
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

# ---- map each timestep to its line index ----

#print(lineIdxs)
#for i in lineIdxs:
#	print(f" | line index : {i} | timestep : {lines[i]}")


vacf = [] # list of autocorrelation functions calculated for each window, what's going to be output

#print('LineMap to timesteps')

#print(lineIdxs)


print(f'{M} windows with {collect_window} frames to correlate over')

# ---- loop over intervals ----
for i in range(M):	
	idx0 = len(lineIdxs) - (M-i)*collect_window
	idxi = len(lineIdxs) - (M-i-1)*collect_window # this has to be in units of frames, not physical time

	# mapping from t0 --> idx0 as the frame 
	#idx0 = int((t0 - timestep0) / dump_interval)
	#idxi = int((ti - timestep0) / dump_interval)
	print(f'Window {i+1}: frame number {idx0} to frame number {idxi}')

	#print(f'Window {i+1}: frame {idx0} to frame {idxi}')

	cvv = [] # autocorrelation functions for each atom type, for this specific interval
	for aType in atom_types: cvv.append([])
	#for idxj in idx0+np.arange(0,idxi-idx0+1,1): # compute correlations with t0 at every timestep within this interval
	for idxj in np.arange(idx0,idxi,1):
	#for idxj in range(idxi-idx0):
		print(f'Correlating frame {idx0} and {idxj}')
		# grab line indices of atom data corresponding to each relevant timetep
		#print(f"---> idxj={idxj}")
		#print(f"---> lineIdxs[idxj]={lineIdxs[idxj]}")


		atomlines_t0 = lines[lineIdxs[idx0]+8 : lineIdxs[idx0]+8+nAtoms]
		atomlines_tj = lines[lineIdxs[idxj]+8 : lineIdxs[idxj]+8+nAtoms] 
		# NEED A MAP GOING FROM FRAME i = 0, 1, 2, etc to line number in dump file
		# FRAME i = 0, 1, 2 etc is idx0 and idxi and idxj


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

print(f"Output matrix shape: {vacf.shape}")
print(f"Columns in output data")
print(f"Time | VACF_1 | VACF_2 | ... ... | VACF_NET")

with open(f'{args.output}','wb') as f: np.save(f,vacf) 