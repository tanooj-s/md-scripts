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
parser.add_argument('-o', action="store", dest="output")
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
densities = np.zeros((1+len(atom_types), nBins)) # output array
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

print("Parsing and analyzing dump directly to calculate densities...")
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
					densities[idx1, binIdx] += 1
					densities[-1, binIdx] += 1 # species blind
		if (currentTime > start) and (previousLine.startswith('ITEM: ATOMS')):
			doCollect = True
			nCollected += 1
		if previousLine.startswith('ITEM: TIMESTEP'):
			currentTime = int(tokens[0])
		previousLine = line
print(f'{round((time.time()-timestart)/60,4)} minutes')
densities /= nCollected
print(densities.shape)
print(f'{nCollected} timesteps collected for densities')
# append z values (i.e. x axis of histograms) as first row
zs = np.arange(minz,maxz+0.1*dz,dz) # hack
print(zs.shape)
assert len(zs) == densities.shape[1]
zs = np.reshape(zs, (1, densities.shape[1]))
densities = np.concatenate((zs,densities))
with open(args.output+".rho_z.npy",'wb') as f: np.save(f, densities)


# normalization, plot profile for each species and net
# only consider profiles out to 4 nm
# everything below is for a specific system and not general but make it general later
rho = densities
idx = int(40/dz) + 1
z = dz * np.arange(0,len(rho[0][:idx]),1)

pltwidth = 12
pltheight = int(np.ceil((rho.shape[0]-1) * 3.))
fig, axes = plt.subplots(rho.shape[0]-1,1,sharex=True)
plt.rcParams['figure.figsize'] = (pltwidth,pltheight)
ionMap = {1: 'F', 2: 'Li', 3: 'Na', 4: 'K', 5: 'Net'}
for i in range(1+len(atom_types)):
    rho_i = rho[i+1][:idx]
    N_i = np.trapz(x=z,y=rho_i) # normalization factor up to this z (TODO move normalization up)
    rho_i /= N_i
    axes[i].plot(z,rho_i,label=f"{ionMap[i+1]}",lw=2)
    # plot long range value 
    limit_val = np.mean(rho_i[-30:])
    axes[i].axhline(limit_val,color='k',linestyle='dashed')
    axes[i].legend(loc="upper right")
    axes[i].grid()
    axes[i].set_ylim(0,2*limit_val)
axes[-1].set_xlabel("z (Å)")
axes[0].set_title("Ion density profiles ρ(z)")
plt.tight_layout()
plt.savefig(args.output+".rho_z.png")