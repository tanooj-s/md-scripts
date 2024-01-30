# generate contour plots for thin slab simulations 
# at each timestep
# bin along z
# for each of these bins
# output a 3-tuple (binz,min_x,max_x) along # binz should be the z-coordinate at the center of that bin


import argparse
import numpy as np
from tqdm import tqdm
import time

parser = argparse.ArgumentParser(description="calculate density profiles")
parser.add_argument('-i', action="store", dest="input")
parser.add_argument('-o', action="store", dest="output")
parser.add_argument('-dz', action="store", dest="dz") # bin width 
parser.add_argument('-minz', action="store", dest="minz")
parser.add_argument('-maxz', action="store", dest="maxz")
parser.add_argument('-start', action="store", dest="start")
parser.add_argument('-t', action="store", dest="atom_types") # atom types to calculate density profiles for, string like '1 3 4'
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
nBinsZ = int((maxz-minz)/dz) + 1
print(nBinsZ)

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

contours = []
# this will have shape (timesteps,nbinsZ,3)
# the data at each timestep is essentially a list of [[z0, minx0, maxx0], [z1, minx1, maxx1] .... [zN, minxN, maxxN]]
# where the three data points are z coordinate, maxx and minx of particles within each bin of z
print("Parsing and analyzing dump directly to obtain contour map...")
timestart = time.time()
with open(args.input,'r') as f:
	doCollect = False
	nCollected = 0 # timesteps summed over
	currentTime = 0
	previousLine = '' # keep saving previous line 
	coords = [] # list of z and x coordinates of particles 
	contour_t = np.zeros((nBinsZ,3)) # contour at this timestep as list of (z, xmin, xmax) over all the bins
	for line in f:
		tokens = purge(line.strip('\n').split(' '))
		if currentTime >= start:
			# you only want to collect data when all tokens are numeric and are atomic data
			checksum = np.sum([not isfloat(t) for t in tokens])
			if (len(tokens) > 4) and (checksum == 0):
				# here just collect all of the relevant data
				# then parse and analyze properly at a new timestep once you hit the line with timestep
				aType, z, x = int(tokens[typeIdx]), float(tokens[zIdx]), float(tokens[xIdx])
				if (aType in atom_types) and (z >= minz) and (z <= maxz):
					binzIdx = np.floor((z - minz)/dz)
					coords.append([binzIdx,x]) 
					#print((aType, z, x))
			if (currentTime > start) and (previousLine.startswith('ITEM: ATOMS')):
				doCollect = True
				nCollected += 1
				coords = np.array(coords)
				contour_t = np.zeros((nBinsZ,3)) # contour at this timestep
				for i in range(nBinsZ):
					# use np.where to find all particles in that z bin, then pull out all the x coords ([:,1]) and find max/min
					z = dz*(i+0.5) # z coordinate at center of bin
					#xmin = np.min(coords[np.where(coords[:,0] == i)][:,1])
					#xmax = np.max(coords[np.where(coords[:,0] == i)][:,1]) # this should be a one-liner
					these_coords = coords[np.where(coords[:,0] == float(i))]
					if len(these_coords) > 0:
						xmin = np.min(these_coords[:,1])
						xmax = np.max(these_coords[:,1])
						contour_t[i] = np.array([z,xmin,xmax])
						#print(f"binIdx: {i}  |  z: {z}  |  xmin: {xmin}  |  xmax: {xmax}")
					else:
						contour_t[i] = np.array([z,0.,0.]) # edge case if no particles present in that bin
				contour_t = np.array(contour_t)
				#print(f"shape of contour_t at this timestep: {contour_t.shape}")
				contours.append(contour_t)
				coords = [] # reset coords for next timestep
		if previousLine.startswith('ITEM: TIMESTEP'):
			currentTime = int(tokens[0])
			print(f"TIMESTEP: {currentTime} | collecting: {doCollect}")
		previousLine = line

print(f'{round((time.time()-timestart)/60,4)} minutes')
contours = np.array(contours)
print(contours.shape)

print(f'{nCollected} timesteps collected for contour maps')
with open(args.output,'wb') as f: np.save(f, contours)

