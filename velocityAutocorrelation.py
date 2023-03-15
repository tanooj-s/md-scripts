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
#atom_types = [int(t) for t in args.atom_types.split(' ')]
atom_type = int(args.atom_types) # for now just assume a single atom type requested

def purge(tokens): return [t for t in tokens if len(t) >= 1] # purge empty strings from token lists

def correlate(t0_vels, t1_vels):
	'''
	assume inputs are (N, 3) shaped tuples of velocities of every atom of a given type 
	need to ensure that atoms are ordered in the same way here before passing to this function
	'''
	
	print(t0_vels.shape)
	print(t1_vels.shape)
	#assert t0_vels.shape == t1_vels.shape

	# HACK - if one is shorter than the other for some reason then truncate the longer one to the same length
	if t0_vels.shape[0] > t1_vels.shape[0]: t0_vels = t0_vels[:t1_vels.shape[0]]
	elif t1_vels.shape[0] > t0_vels.shape[0]: t1_vels = t1_vels[:t0_vels.shape[0]]


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
		atomlines_t0 =  lines[tsIdxMap[t0]+8:tsIdxMap[t0]+8+nAtoms]
		atomlines_tj =  lines[tsIdxMap[tj]+8:tsIdxMap[tj]+8+nAtoms] 

		# make dictionaries of atom data at each timestep with {aID: [vx,vy,vz]}
		# this makes it easier to later make sure data at each timestep is ordered the same way when calculating correlations 
		# TODO: add aType to dict values

		t0_dict = {}
		for line in atomlines_t0:
			tokens = purge(line.split(' '))
			aType = int(tokens[typeIdx])
			if aType == atom_type: t0_dict[int(tokens[idIdx])] = np.array([float(tokens[vxIdx]), float(tokens[vyIdx]), float(tokens[vzIdx])])

		tj_dict = {}
		for line in atomlines_tj:
			tokens = purge(line.split(' '))
			aType = int(tokens[typeIdx])
			if aType == atom_type: tj_dict[int(tokens[idIdx])] = np.array([float(tokens[vxIdx]), float(tokens[vyIdx]), float(tokens[vzIdx])])

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






