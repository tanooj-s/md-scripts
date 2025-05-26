

import argparse
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="calculate 1d density profiles")
parser.add_argument('-i', action="store", dest="input")
parser.add_argument('-dz', action="store", dest="dz") # bin width 
parser.add_argument('-start', action="store", dest="start")
args = parser.parse_args()

atom_types = [1,2,3] # edit this as appropriate 

dz = float(args.dz)
start = int(args.start)
print(f'Atom types to bin: {atom_types}')

def purge(tokens): return [t for t in tokens if len(t) >= 1] # purge empty strings from token lists

def isfloat(s): # hack function to confirm tokens are numeric as floats
	try:
		float(s)
		return True
	except ValueError:
		return False

# ---- single pass over file ----
# edit dimensions as appropriate for other directions 
minz, maxz = 0, 0
with open(args.input,'r') as f:
	nline = 0
	for line in f:
		tokens = purge(line.strip('\n').split(' '))
		print(tokens)
		if nline == 7: 
			minz, maxz = float(tokens[0]), float(tokens[1])
			break
		nline += 1

print(f"minz: {minz}")
print(f"maxz: {maxz}")

nbins = int((maxz-minz)/dz) + 1
rho = np.zeros((1+len(atom_types), nbins)) # output array
# print out which atom type is which row in output array

# last is species blind

# infer dump format from first 10 lines
idIdx, typeIdx, xIdx, yIdx, zIdx = 0, 0, 0, 0, 0
with open(args.input,'r') as f:
	for _ in range(15):
		line = f.readline()
		line = line.strip('\n')
		if line.startswith("ITEM: ATOMS"):
			tokens = purge(line.split(' '))
			idIdx = tokens.index('id') - 2
			typeIdx = tokens.index('type') - 2
			xIdx = tokens.index('x') - 2
			yIdx = tokens.index('y') - 2
			zIdx = tokens.index('z') - 2

print("Parsing and analyzing dump to calculate rho...")
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
			checksum = np.sum([not isfloat(t) for t in tokens])
			if (len(tokens) > 2) and (checksum == 0):
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
print(f'{nCollected} timesteps collected')

rho /= nCollected
print(rho.shape)

# append z values (i.e. x axis of histograms) as first row
zs = minz + dz*np.arange(0,rho.shape[1],1)
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
# assuming input file is named .dump and not .dump.0



