# output a 2d composition profile (e.g X(x,z)) for each atom species

import argparse
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="calculate composition profiles")
parser.add_argument('-i', action="store", dest="input")
parser.add_argument('-o', action="store", dest="output") # should be .npy
parser.add_argument('-dz', action="store", dest="dz") 
parser.add_argument('-dx', action="store", dest="dx") # bin widths
parser.add_argument('-minz', action="store", dest="minz")
parser.add_argument('-maxz', action="store", dest="maxz")
parser.add_argument('-minx', action="store", dest="minx")
parser.add_argument('-maxx', action="store", dest="maxx")
parser.add_argument('-start', action="store", dest="start") 
parser.add_argument('-t', action="store", dest="atom_types") # atom types to use for composition profiles, string like '1 2 3 4'
args = parser.parse_args()

dz = float(args.dz)
dx = float(args.dx)
minz = float(args.minz)
maxz = float(args.maxz)
minx = float(args.minx)
maxx = float(args.maxx)
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
print(f'MinX: {minx}')
print(f'MaxX: {maxx}')
nbinsz = int((maxz-minz)/dz) + 1
nbinsx = int((maxx-minx)/dx) + 1
print(f"nbinsz: {nbinsz}")
print(f"nbinsx: {nbinsx}")

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

# ---- parse data file ----
Xs = [] # this will have shape (timesteps,ntypes,nbinsx,nbinsz)
print("Parsing and analyzing dump directly to obtain composition profiles on a grid...")
timestart = time.time()
with open(args.input,'r') as f:
	doCollect = False
	nCollected = 0 # timesteps summed over
	currentTime = 0
	previousLine = '' # keep saving previous line 
	N_t = np.zeros((5,nbinsx,nbinsz)) # counts at this timestep
	for line in tqdm(f):
		tokens = purge(line.strip('\n').split(' '))
		if doCollect == True: 
			# you only want to collect data when all tokens are numeric and are atomic data
			checksum = np.sum([not isfloat(t) for t in tokens])
			if (len(tokens) > 4) and (checksum == 0):
				aType = int(tokens[1])
				x = float(tokens[2])
				z = float(tokens[4])  # TODO: figure out what's going with token indexing, this shouldn't be hardcoded
				if (aType in atom_types) and (z >= minz) and (z <= maxz) and (x >= minx) and (x <= maxx):
					typeIdx = aType-1
					zIdx = int((z - minz)/dz)
					xIdx = int((x - minx)/dx)
					N_t[typeIdx,xIdx,zIdx] += 1
					N_t[-1,xIdx,zIdx] += 1 # net for normalization
		if (currentTime >= start) and (previousLine.startswith('ITEM: ATOMS')):
			doCollect = True
		if previousLine.startswith('ITEM: TIMESTEP'):
			if np.sum(N_t) > 0:
				X_t = N_t[1:-1,:,:] # only need cation counts for composition calculations 
				for i in range(1,N_t.shape[0]-1): 
					for j in range(N_t.shape[1]):  
						for k in range(N_t.shape[2]):  
							if N_t[-1,j,k] != 0:
								X_t[i-1,j,k] *= (200 / N_t[-1,j,k])
							else:
								X_t[i-1,j,k] = 0
				Xs.append(X_t)
				nCollected += 1
			currentTime = int(tokens[0])		
			N_t = np.zeros((5,nbinsx,nbinsz))
		previousLine = line

print(f"{round((time.time()-timestart)/60,4)} minutes to obtain density from {nCollected} timesteps")
Xs = np.array(Xs)
print("Trajectory of densities:")
print(Xs.shape)
assert len(Xs.shape) == 4
X = np.mean(Xs,axis=0) 
print("Time averaged")
print(X.shape)
with open(args.output,'wb') as f: np.save(f, X)

# plot out 
x = dx*np.arange(0,X.shape[1],1)
y = dx*np.arange(0,X.shape[2],1)
xmin, xmax = x[0], x[-1]
ymin, ymax = y[0], y[-1]
# colormap params
scaledn, scaleup = (2/3), (4/3)
vmins = [scaledn*46.5,scaledn*11.5,scaledn*42]
vmaxs = [scaleup*46.5,scaleup*11.5,scaleup*42]
components = ['LiF','NaF','KF']
plt.rcParams['figure.figsize'] = (20,5)
plt.rcParams['font.size'] = 18
fig, axes = plt.subplots(1,3,sharey=True)
for i in range(3):
	xtent = dx*np.arange(0,X[i].shape[0],1)
	ytent = dz*np.arange(0,X[i].shape[1],1)
	pos = axes[i].imshow(X[i].T,origin='lower',vmin=vmins[i],vmax=vmaxs[i],extent=[xmin,xmax,ymin,ymax])
	fig.colorbar(pos,ax=axes[i])
	axes[i].set_title(f'{components[i]} X(x,z)')
	axes[i].set_xlabel('x (A)')
axes[0].set_ylabel('z (A)')
fname = args.output[:-3] + ".png"
plt.savefig(fname)
plt.clf()


















