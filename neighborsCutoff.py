# basic order parameter that calculates nearest neighbors of each atom within some cutoff
# output distibution of this parameter and mean value as a function of time
# can add more complex order parameters later

# (yields 4.28 for liquid and 5.39 for rocksalt NaCl at T=1100K with a cutoff of 3.5A with fairly distinct distributions)

import argparse
import numpy as np
import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description="parse lammps dump file")
parser.add_argument("-i", action="store", dest="input")
parser.add_argument("-cut", action="store", dest="cut") # cutoff to calculate neighbors within
parser.add_argument("-o", action="store", dest="output")
args = parser.parse_args()

cut = float(args.cut)
print(f'Cutoff to identify neighbors: {cut}')

def purge(tokens): return [t for t in tokens if len(t) >= 1]

class Atom:
	def __init__(self,atomID,atomType,x,y,z):
		self.id = atomID
		self.type = atomType
		self.r = np.array([x,y,z])

def minimumImage(r,L):
	r -= L*np.array([round(i) for i in r/L])
	return r

def findNeighbors(atom,atoms,L,cutoff):
	'''
	find n_neighbors of reference atom within cutoff
	loop over all other atoms
	'''
	n_neighbors = 0
	for other in atoms:
		if other.id != atom.id:
			rij = minimumImage(other.r-atom.r,L)
			rij_sc = np.sqrt(np.dot(rij,rij))
			if rij_sc <= cutoff: n_neighbors += 1
	return n_neighbors

def binNeighbors(n_neighbors):
	hist = np.zeros(12,)
	for n in n_neighbors:
		idx = int(n) # implicitly assuming a bin width of 1 since this is a distribution of ints
		hist[idx] += 1
	return hist

# ---- parse LAMMPS dump file -----
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
nAtoms = int(lines[nHeadIdxs[0]+1]) # no grand canonical shenanigans
timestepIdxs = [i+1 for i in tsHeadIdxs]
timestep0, timestep1 = int(lines[timestepIdxs[0]]), int(lines[timestepIdxs[1]])
dump_interval = timestep1 - timestep0

# which frames to calculate at, make this a flag later
# you want to skip over some of these (nEvery)
frames = list(np.arange(50,501,1))
frames = [dump_interval*round(f) for f in frames]


#nUse = int(0.8*len(tsHeadIdxs)) # choose length of trajectory to analyze, might want to make this a flag
#tsHeadIdxs = tsHeadIdxs[-nframes:]
#nHeadIdxs = nHeadIdxs[-nframes:]
#boxHeadIdxs = boxHeadIdxs[-nframes:]
#atomHeadIdxs = atomHeadIdxs[-nframes:]
print(f"Timesteps to calculate neighbor distributions for: {len(frames)}")

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


distributions = [] # time series of distribution of order parameter over system
mean_nbs = [] # time series of mean order parameter over system

# ---- loop through frames, calculate neighbors at each timestep ---
print("Reading ensemble trajectory to calculate neighbor distributions...")
for idx in tqdm(tsHeadIdxs):
	timestep = int(lines[idx+1])
	print("TIMESTEP: " + str(timestep))
	if timestep in frames:
		nAtoms = int(lines[idx+3])
		atomlines = lines[idx+9:idx+9+nAtoms]	
		# redundant
		xDimLine = lines[idx+5].strip('\n')
		yDimLine = lines[idx+6].strip('\n')
		zDimLine = lines[idx+7].strip('\n')
		xLo, xHi = float(xDimLine.split(' ')[0]), float(xDimLine.split(' ')[1])
		yLo, yHi = float(yDimLine.split(' ')[0]), float(yDimLine.split(' ')[1])
		zLo, zHi = float(zDimLine.split(' ')[0]), float(zDimLine.split(' ')[1])
		L = np.array([xHi-xLo,yHi-yLo,zHi-zLo]) # box dims at this timestep
		
		atoms = []
		n_neighbors = [] # n_neighbors for each atom at this timestep (2D array)

		for line in atomlines:
			tokens = purge(line.split(' ')) 
			atoms.append(Atom(atomID=int(tokens[idIdx]),atomType=int(tokens[typeIdx]),x=float(tokens[xIdx]),y=float(tokens[yIdx]),z=float(tokens[zIdx])))

		# find n_neighbors for each atom
		start = time.time()
		for atom in atoms:
			n = findNeighbors(atom,atoms,L,cutoff=cut)
			n_neighbors.append(n)

		assert len(n_neighbors) == len(atoms)
		n_neighbors = np.array(n_neighbors)
		mean_nbs.append(np.mean(n_neighbors))
		
		# calculate histogram
		hist = binNeighbors(n_neighbors)
		distributions.append(hist)
		print(f'{time.time() - start} seconds')

distributions = np.array(distributions)
mean_nbs = np.array(mean_nbs)
with open(args.output+'_hists.npy','wb') as f: np.save(f, distributions)
with open(args.output+'_mean.npy','wb') as f: np.save(f, mean_nbs)
