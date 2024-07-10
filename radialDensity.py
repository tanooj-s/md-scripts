# 1d radial density profiles n_i(r) for a spherical geometry

import argparse
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="calculate density profiles")
parser.add_argument('-i', action="store", dest="input")
parser.add_argument('-o', action="store", dest="output") # numpy file
parser.add_argument('-dr', action="store", dest="dr") 
parser.add_argument('-rlim', action="store", dest="rlim")
parser.add_argument('-start', action="store", dest="start")
parser.add_argument('-nevery', action="store", dest="nevery")
parser.add_argument('-t', action="store", dest="atom_types") # atom types, string like '1 3 4'
args = parser.parse_args()

dr = float(args.dr)
rlim = float(args.rlim)
start = int(args.start)
nevery = int(args.nevery)
atom_types = [int(t) for t in args.atom_types.split(' ')]
print(f'Atom types to bin: {atom_types}')

def purge(tokens): return [t for t in tokens if len(t) >= 1] # purge empty strings from token lists

def isfloat(s): # hack function to confirm tokens are numeric as floats
	try:
		float(s)
		return True
	except ValueError:
		return False

# ---- get indices etc ----
nbins = int((rlim)/dr) + 1
print(f"nbins: {nbins}")
idIdx, typeIdx, xIdx, yIdx, zIdx = 0, 0, 0, 0, 0
lines = []
with open(args.input,'r') as f:
	for _ in range(15):
		line = f.readline()
		line = line.strip('\n')
		lines.append(line)
for line in lines:		
	if line.startswith("ITEM: ATOMS"):
		tokens = purge(line.split(' '))
		print(tokens)
		idIdx = tokens.index('id') - 2
		typeIdx = tokens.index('type') - 2
		xIdx = tokens.index('x') - 2
		yIdx = tokens.index('y') - 2
		zIdx = tokens.index('z') - 2
print(f'{idIdx} {typeIdx} {xIdx} {yIdx} {zIdx}')
# assuming N is constant
N = int(lines[3])
print(f"{N} particles in system")

# ---- parse file ----
# charge map system specific
qMap = {1: -1., 2: 1., 3: 1., 4: 1.}
counts = []
densities = [] # these will have shape (timesteps,nbins,2+ntypes)
print("Parsing and analyzing dump directly for density profiles")
timestart = time.time()
with open(args.input,'r') as f:
	doCollect = False
	nCollected = 0 # timesteps summed over
	currentTime = 0
	previousLine = '' # keep saving previous line 
	n_t = np.zeros((2+len(atom_types),nbins)) # (r,n_1(r),n_2(r)....,n_net(r))
	N_t = np.zeros((2+len(atom_types),nbins)) # (r,N_1(r),N_2(r)...) explicit number in each shell for later composition calcualtions
	atom_data = np.zeros((N,4)) # (type,x,y,z) for each atom at this timestep, postprocess once all atoms are read in
	for line in tqdm(f):
		tokens = purge(line.strip('\n').split(' '))
		if doCollect == True: 
			# you only want to collect data when all tokens are numeric and are atomic data
			checksum = np.sum([not isfloat(t) for t in tokens])
			if (len(tokens) > 4) and (checksum == 0):
				aID = int(tokens[idIdx])
				aType = int(tokens[typeIdx])
				x = float(tokens[xIdx])
				y = float(tokens[yIdx])
				z = float(tokens[zIdx])  
				atom_data[aID-1] = (aType,x,y,z) # LAMMPS indexes from 1		 	
		if (currentTime >= start) and ((currentTime % nevery) == 0) and (previousLine.startswith('ITEM: ATOMS')):
			doCollect = True
		if previousLine.startswith('ITEM: TIMESTEP'):
			if np.sum(atom_data) != 0.: # make sure array isn't empty
				# there's probably a scipy function to do this in two lines but just loop over particles
				for i in range(N):
					idx1 = int(atom_data[i][0])
					r = atom_data[i][1:] # positions 
					# assuming particles are centered at 0, otherwise find origin
					rsc = (r[0]**2 + r[1]**2 + r[2]**2) ** 0.5 # r[0] - origin[0] etc
					if rsc < rlim:
						idx2 = int(rsc/dr) 
						N_t[idx1,idx2] += 1 # note that since lammps indexes from 1 don't need to add +1 to first index
						# n_t[-1,idx2] += chargeMap[idx1] # charge density 
						N_t[-1,idx2] += 1 # net 
					N_t[0] = dr*np.arange(0,nbins,1) # r, distance of spherical shell
				# need to normalize by volume of spherical shell (4pir**2)dr at each r i.e. n(r) = N(r)/V(r) for all profiles
				V_r = np.zeros(nbins,)
				for i in range(1,1+nbins):
					# compute normalization factor for density
					r_i = dr * i
					V_r[i-1] = 4 * np.pi * (r_i**2) * dr # V_r probably just needs to be computed once
					# compute N, X_i within that bin 
				n_t[1:] = N_t[1:] / V_r 
				n_t[0] = N_t[0] # r, distance of spherical shell
				densities.append(n_t)
				counts.append(N_t)
				# reset arrays for next timestep
				n_t = np.zeros((2+len(atom_types),nbins)) # (r,n_1(r),n_2(r)....,n_net(r))
				N_t = np.zeros((2+len(atom_types),nbins))
				atom_data = np.zeros((N,4)) # (type,x,y,z) for each atom at this timestep
				nCollected += 1
			currentTime = int(tokens[0])		
		previousLine = line

print(f"{round((time.time()-timestart)/60,4)} minutes to obtain density from {nCollected} timesteps")
densities = np.array(densities)
counts = np.array(counts)
# (timesteps,types,nbins)

# output compositions a a numpy array for notebook analysis
with open(f"{args.output}.spherical_rho.npy","wb") as f: np.save(f,densities)
with open(f"{args.output}.spherical_counts.npy","wb") as f: np.save(f,counts)

# time average and plot densities on separate panels
rho = np.mean(densities,axis=0)
plt.rcParams['figure.figsize'] = (20,3)
plt.rcParams['font.size'] = 16
fig, axes = plt.subplots(1,4,sharey=True)
labelMap = ['F','Li','Na','K'] # system specific
for i in range(4):
    axes[i].plot(0.1*rho[0],rho[1+i],label=f'{labelMap[i]}')
    axes[i].grid()
    axes[i].legend(loc='upper right')
    axes[i].set_xlabel('r (nm)')
axes[0].set_ylabel('n(r)')
plt.tight_layout()
plt.savefig(f"{args.output}.spherical_rho.png")
plt.clf()


# plot composition on a single panel for all ions
# this is system specific
plt.rcParams['figure.figsize'] = (8,5)
plt.rcParams['font.size'] = 16
counts = np.mean(counts,axis=0)
r = 0.1*counts[0][1:]
N = counts[-1][1:]
for i in range(4):
    Xi = counts[i+1][1:] / N
    plt.plot(r,Xi,label=f'{labelMap[i]}')
plt.grid()
plt.legend(loc='upper right')
plt.xlabel('r (nm)')
plt.ylabel('Ion wt %')
plt.tight_layout()
plt.savefig(f"{args.output}.spherical_counts.png")


