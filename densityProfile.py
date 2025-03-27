# read in a dump file, calculate z-resolved density profiles of specified atom types
# this version for large dump files - do analysis while file is being read in
# user needs to specify minz, maxz and start timestep for analysis here

# (useful for interfaces, responses to external potentials etc)
# output a numpy array of shape (1+nTypes, nbins)

import argparse
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="calculate density profiles")
parser.add_argument('-i', action="store", dest="input")
parser.add_argument('-dz', action="store", dest="dz") # bin width 
parser.add_argument('-minz', action="store", dest="minz")
parser.add_argument('-maxz', action="store", dest="maxz")
parser.add_argument('-start', action="store", dest="start")
parser.add_argument('-t', action="store", dest="atom_types") # atom types to calculate profiles for, string like '1 3 4'
args = parser.parse_args()

dz = float(args.dz)
minz = float(args.minz)
maxz = float(args.maxz)
start = int(args.start)
atom_types = [int(t) for t in args.atom_types.split(' ')]
print(f'Atom types to bin: {atom_types}')

def purge(tokens): return [t for t in tokens if len(t) >= 1] # purge empty strings from token lists

def isfloat(s): # hack function to confirm tokens are numeric as floats
	try:
		float(s)
		return True
	except ValueError:
		return False

# ---- single pass over file ----
print(f'MinZ: {minz}')
print(f'MaxZ: {maxz}')
nBins = int((maxz-minz)/dz) + 1
rho = np.zeros((1+len(atom_types), nBins)) # output array
# print out which atom type is which row in output array
for i1, i2 in enumerate(atom_types):
	print(f'Row {i1+1} | Atom type {i2}') # 0 will be zs

# last is species blind

# infer dump format from first 10 lines
idIdx, typeIdx, xIdx, yIdx, zIdx = 0, 0, 0, 0, 0
with open(args.input,'r') as f:
	for _ in range(15):
		line = f.readline()
		line = line.strip('\n')
		if line.startswith("ITEM: ATOMS"):
			tokens = purge(line.split(' '))
			print(tokens)
			idIdx = tokens.index('id') - 2
			typeIdx = tokens.index('type') - 2
			xIdx = tokens.index('x') - 2
			yIdx = tokens.index('y') - 2
			zIdx = tokens.index('z') - 2
print(f'{idIdx} {typeIdx} {xIdx} {yIdx} {zIdx}')


# add mean field normalization
# which is just the average number of particles within a dz bin (i.e. an x-y thin slab)


print("Parsing and analyzing dump directly to calculate rho...")
timestart = time.time()
with open(args.input,'r') as f:
	nCollected = 0 # timesteps summed over
	doCollect = False
	currentTime = 0
	previousLine = '' # keep saving previous line 
	for line in tqdm(f):
		lines = line.strip('\n')
		tokens = purge(line.split(' '))
		if doCollect == True:
			# you only want to collect data when all tokens are numeric 
			checksum = np.sum([not isfloat(t) for t in tokens])
			if (len(tokens) > 4) and (checksum == 0):
				# bin data here 
				aType, z = int(tokens[typeIdx]), float(tokens[zIdx])
				if (aType in atom_types) and (z >= minz) and (z <= maxz):
					idx1 = atom_types.index(aType) # index atom type along first axis of output array
					binIdx = int((z - minz)/dz)
					rho[idx1, binIdx] += 1
					rho[-1, binIdx] += 1 # species blind
		if (currentTime > start) and (previousLine.startswith('ITEM: ATOMS')):
			doCollect = True
			nCollected += 1
		if previousLine.startswith('ITEM: TIMESTEP'):
			currentTime = int(tokens[0])
		previousLine = line
print(f'{round((time.time()-timestart)/60,4)} minutes')
rho /= nCollected
print(rho.shape)
print(f'{nCollected} timesteps collected')

# append z values (i.e. x axis of histograms) as first row
zs = np.arange(minz,maxz+0.1*dz,dz) # hack
print(zs.shape)
assert len(zs) == rho.shape[1]
zs = np.reshape(zs, (1, rho.shape[1]))
rho = np.concatenate((zs,rho))

# actually normalize so that each computed distribution adds up to 1
N = np.trapz(x=rho[0],y=rho[1:]) # note this is broadcasting every integrand to each separate profile
N = np.reshape(N,(N.shape[0],1))
N = np.tile(N,rho.shape[1])
rho[1:] /= N

with open(args.input[:-5]+".rho_z.npy",'wb') as f: np.save(f, rho)
