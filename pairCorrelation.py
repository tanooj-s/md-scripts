
import argparse
import numpy as np
import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description="parse lammps dump file")
parser.add_argument("-i", action="store", dest="input")
parser.add_argument("-p", action="store", dest="pairstring") # string of atom type pairs to calculate RDFs for e.g. '1 1 2 2 1 3 2 3'
parser.add_argument("-dr", action="store", dest="dr")
parser.add_argument("-o", action="store", dest="output")
args = parser.parse_args()

dr = float(args.dr)
pairstring = args.pairstring

pairtokens = pairstring.split(' ')
assert len(pairtokens) % 2 == 0
# split pairtypes so you have an array of pairs like ['1 1','2 2','1 3','2 3']
pairs = []
while len(pairtokens) > 0:
	pairs.append(pairtokens[:2])
	pairtokens = pairtokens[2:]
# output array will be of shape (1+nPairs, nbins)
# iterate over pairs 
# for each pair, calculate pairCorrelation(pair0, pair1, L, r)

def purge(tokens): return [t for t in tokens if len(t) >= 1]

class Atom:
	def __init__(self,atomID,atomType,x,y,z):
		self.id = atomID
		self.type = atomType
		self.r = np.array([x,y,z])

def minimumImage(r,L):
	r -= L*np.array([round(i) for i in r/L])
	return r

def pairCorrelation(atomsA,atomsB,L,r):
	'''
	take in two lists of atom objects 
	return g(r)
	this is assumed to be at a single timestep
	need box dims at each timestep for ideal gas normalization 
	also take in r which is the x axis of the histogram, should be common for all gij's
	'''
	gr = np.zeros(nbins,) # nbins should be common for all timesteps so don't pass in as an arg to this function
	# get number of particles within shells of thickness dr i.e. dn(r)
	for atomi in atomsA:
		for atomj in atomsB:
			if atomi.id != atomj.id:
				rij = minimumImage(atomj.r - atomi.r,L)
				rij_sc = np.sqrt(np.dot(rij,rij))
				if rij_sc < rLim:
					binIdx = int(rij_sc/dr) # if dr=1 and rij=0.8 then we want the bin indexed at 0 to be filled
					gr[binIdx] += 1 

	# normalization
	shellV = 4*np.pi*np.power(r,2)*dr # 4πr2dr
	V = L[0]*L[1]*L[2]
	igRho = (len(atomsA)+len(atomsB))/V
	rhoA = len(atomsA)/V # number density of A
	rhoB = len(atomsB)/V # number density of B
	gr /= shellV # normalize by volume of spherical shell at each bin 
	gr /= rhoB # normalize by number density of "other" atoms
	gr /= len(atomsA) # normalize by number of "self" atoms 
	return gr


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
# choose length of trajectory to analyze based on timesteps 
nUse = int(0.8*len(tsHeadIdxs))
# choose fraction based on number of timesteps 
tsHeadIdxs = tsHeadIdxs[nUse:]
nHeadIdxs = nHeadIdxs[nUse:]
boxHeadIdxs = boxHeadIdxs[nUse:]
atomHeadIdxs = atomHeadIdxs[nUse:]

print(f"Timesteps to average g(r) over: {len(tsHeadIdxs)}")

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



# ---- initial pass over box dims to obtain nbins ----
rLim = 1000
print("Reading box dimensions to determine number of histogram bins...")
for idx in tsHeadIdxs: # line indices being iterated over
	timestep = int(lines[idx+1])
	xDimLine = lines[idx+5].strip('\n')
	yDimLine = lines[idx+6].strip('\n')
	zDimLine = lines[idx+7].strip('\n')
	xLo, xHi = float(xDimLine.split(' ')[0]), float(xDimLine.split(' ')[1])
	yLo, yHi = float(yDimLine.split(' ')[0]), float(yDimLine.split(' ')[1])
	zLo, zHi = float(zDimLine.split(' ')[0]), float(zDimLine.split(' ')[1])
	L = np.array([xHi-xLo,yHi-yLo,zHi-zLo]) # box dimensions for this timestep
	minDim = np.min(L)
	if minDim < rLim: rLim = minDim

rLim /= 2 # half minimum box dim
print(f"Generating g(r) up to r={rLim}")
nbins = int(rLim/dr) + 1
print(f"Number of bins: {nbins}")



r = dr*np.arange(0.001,nbins,1) # x axis of histograms
rdfs = np.zeros((len(pairs), nbins)) # output array
# print out which pairs are which row in output array
for idx, pair in enumerate(pairs):
	print(f'Row {idx+1} | Types {pair[0]} {pair[1]}') # 0 will be zs

# need a map to grab only which types and positions are needed to calculate at each timestep
relevantTypes = list(set(pairstring.split(' ')))
relevantTypes = [int(t) for t in relevantTypes]
print(relevantTypes)
typeMap = {}
for idx, t in enumerate(relevantTypes):
	typeMap[t] = idx

# ------------------------------------
# loop through frames of interest and get RDFs at each frame

print("Reading ensemble trajectory to calculate pair correlations...")

for idx in tqdm(tsHeadIdxs):


	timestep = int(lines[idx+1])
	nAtoms = int(lines[idx+3])
	atomlines = lines[idx+9:idx+9+nAtoms]	
	print("TIMESTEP: " + str(timestep))
	# redundant
	xDimLine = lines[idx+5].strip('\n')
	yDimLine = lines[idx+6].strip('\n')
	zDimLine = lines[idx+7].strip('\n')
	xLo, xHi = float(xDimLine.split(' ')[0]), float(xDimLine.split(' ')[1])
	yLo, yHi = float(yDimLine.split(' ')[0]), float(yDimLine.split(' ')[1])
	zLo, zHi = float(zDimLine.split(' ')[0]), float(zDimLine.split(' ')[1])
	

	# at each timestep, create an array of atom objects for each relevant type 
	# then iterate over pairs, grab pair0 and pair1 from the array of positions
	# need to map indices of the atoms array to what type it is 
	atoms = []
	for t in relevantTypes:
		atoms.append([])

	for line in atomlines:
		tokens = purge(line.split(' ')) 
		aID, aType, x, y, z = int(tokens[idIdx]), int(tokens[typeIdx]), float(tokens[xIdx]), float(tokens[yIdx]), float(tokens[zIdx])
		if aType in relevantTypes:
			atoms[typeMap[aType]].append(Atom(atomID=aID,atomType=aType,x=x,y=y,z=z))


	#print(atoms)
	#print(typeMap)
	#print(atoms[typeMap])

	for pair in pairs:
		type1_atoms = atoms[typeMap[int(pair[0])]]
		type2_atoms = atoms[typeMap[int(pair[1])]]
		idx = pairs.index(pair)
		start = time.time()
		rdfs[idx] = pairCorrelation(type1_atoms, type2_atoms, L, r)
		print(f'{round(time.time()-start)}s for g{pair[0]}{pair[1]}(r)')


# append rs
assert len(r) == rdfs.shape[1]
r = np.reshape(r, (1, rdfs.shape[1]))
rdfs = np.concatenate((r,rdfs))
with open(args.output,'wb') as f: np.save(f, rdfs)
