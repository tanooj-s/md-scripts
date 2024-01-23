# prototype algorithm to calculate wetting angle by tracing out circles at different slices of z
# you only need the first two dz slices to calculate an angle 
# assumption here is that drops are "circularly" symmetric

# slice closest to surface has a radius r0
# the next slice has a radius r1 
# define dr = r0 - r1
# calculate the wetting angle as arccos(dr/r0)

# need (x, y) coordinates of all atoms that are in each slice
# center = mean((x,y))
# radius of each slice = mean((xi, yi) - center)

# average this over equilibrated configurations


# === logistics ====
# at each timestep, create slice0_coords, slice1_coords which are lists of the (x, y) coords of atoms in each dz
# pass those into a wettingAngle function that returns the calculated theta at that timestep
# in wettingAngle, first obtain the center of each circle by using the mean of slice0_coords, slice1_coords
# calculate radius as mean((xi, yi) - center) for both
# calculate dr = r0 - r1 
# return arccos(dr/r0)
# debugging: print out r0, r1 at each timestep, plot out timeseries of calculated radii and angles
# also need to make sure each slice is not sparsely populated so dz should be > 2 angstroms, possibly more
# need z position of the substrate so that atoms that evaporate from the drop aren't used 


import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser(description="parse lammps dump file for wetting sims")
parser.add_argument("-i", action="store", dest="input")
parser.add_argument("-o", action="store", dest="output")
parser.add_argument("-t", action="store", dest="droptypes") # atom types in droplet, pass in a string like '1 2 3'
parser.add_argument("-dz", action="store", dest="dz")
args = parser.parse_args()
dz = float(args.dz)
dropstring = args.droptypes

def purge(tokens): return [t for t in tokens if len(t) >= 1]


# ---- relevant functions go here ----
# !! logic below is bad and I am feeling bad
# this gets the mean radius and not the actual radius
# need to do some kind of smarter fitting
# scan across angles and find coordinates of furthest atom from the center
# then only use these "furthest" atoms to find the mean radius in each slice
# (really would be some kind of scan over a 2d solid angle)
def wettingAngle(slice0_coords, slice1_coords):
	# assume each is a list of 2-tuples
	# cast to numpy array of shape (nAtoms, 2)
	slice0_coords = np.array(slice0_coords)
	slice1_coords = np.array(slice1_coords)
	# recenter coordinates
	coords0 = slice0_coords - np.mean(slice0_coords, axis=0)
	coords1 = slice1_coords - np.mean(slice1_coords, axis=0)
	# radius of each circle 
	r0 = np.mean((coords0[:,0]**2 + coords0[:,1]**2) ** 0.5)
	r1 = np.mean((coords1[:,0]**2 + coords1[:,1]**2) ** 0.5)
	print(f'r0: {round(r0,2)} | r1: {round(r1,2)}')
	dr = r0 - r1
	theta = np.arccos(dr/r0) * (180/np.pi)
	print (f'θ: {round(theta,2)}')
	return r0, r1, theta


# ---- parse LAMMPS dump file -----
print("Parsing dump file...")
tsHeadIdxs = []
nHeadIdxs = []
boxHeadIdxs = []
atomHeadIdxs = []

with open(args.input,'r') as f: lines = f.readlines()
lines = [l.strip('\n') for l in tqdm(lines)]

linecounter = 0
for line in tqdm(lines):
	if line.startswith("ITEM: TIMESTEP"): tsHeadIdxs.append(linecounter)
	if line.startswith("ITEM: NUMBER"): nHeadIdxs.append(linecounter)
	if line.startswith("ITEM: BOX BOUNDS"): boxHeadIdxs.append(linecounter)
	if line.startswith("ITEM: ATOMS"): atomHeadIdxs.append(linecounter)
	linecounter += 1

boxBoundLines = []
atomLines = []

nAtoms = int(lines[nHeadIdxs[0]+1]) 
nUse = int(0.5*len(tsHeadIdxs)) # choose length of trajectory to analyze 
tsHeadIdxs = tsHeadIdxs[nUse:]
nHeadIdxs = nHeadIdxs[nUse:]
boxHeadIdxs = boxHeadIdxs[nUse:]
atomHeadIdxs = atomHeadIdxs[nUse:]
print(f"Timesteps to average over: {len(tsHeadIdxs)}")
for idx in boxHeadIdxs:
	boxBoundLines.append(lines[idx+1:idx+4])
for idx in atomHeadIdxs:
	atomLines.append(lines[idx+1:idx+nAtoms+1])


# ------- calculate -----------
# define liquid atom types here, should be a flag (see sim output)
dropTypes = [int(t) for t in purge(dropstring.split(' '))]
# timeseries of relevant quantites
r0s = []
r1s = []
thetas = [] 
dropradii = [] # nominal drop radius

print("Reading ensemble trajectory to estimate wetting angle...")
for idx in tqdm(tsHeadIdxs):
	atoms = [] # list of atom objects at this timestep
	timestep = int(lines[idx+1])
	nAtoms = int(lines[idx+3])
	atomlines = lines[idx+9:idx+9+nAtoms]	
	print("TIMESTEP: " + str(timestep))
	start = time.time()
	xDimLine = lines[idx+5].strip('\n')
	yDimLine = lines[idx+6].strip('\n')
	zDimLine = lines[idx+7].strip('\n')
	xLo, xHi = float(xDimLine.split(' ')[0]), float(xDimLine.split(' ')[1])
	yLo, yHi = float(yDimLine.split(' ')[0]), float(yDimLine.split(' ')[1])
	zLo, zHi = float(zDimLine.split(' ')[0]), float(zDimLine.split(' ')[1])

	# ==== read atom information, populate slice lists ====
	# you need the minimum z position of droplet atoms to assign bins

	subZ = 0 # note you need an edge case for atoms that might dissociate from the drop and fly away so need position of the substrate too 
	minZ = 1000 # minimum position of droplet atoms, used to populate slice lists
	
	for line in atomlines:
		tokens = purge(line.split(' ')) # id type x y z 
		aType, x, y, z = int(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])	
		if aType in dropTypes:
			if (z < minZ) and (z > subZ): minZ = z
	print(f'minZ: {minZ}')


	# angle calculation
	slice0_coords = []
	slice1_coords = []

	# nominal radius
	x_coords = []
	y_coords = []

	for line in atomlines:
		tokens = purge(line.split(' ')) # id type x y z 
		aType, x, y, z = int(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])
		if aType in dropTypes:
			if (z >= minZ) and (z <= minZ + dz): 
				slice0_coords.append((x,y))
			elif (z >= minZ + dz) and (z <= minZ + 2*dz):
				slice1_coords.append((x,y))
			x_coords.append(x)
			y_coords.append(y)

	# radius calculation
	x_coords, y_coords = np.array(x_coords), np.array(y_coords)
	xmin, xmax = np.min(x_coords), np.max(x_coords)
	ymin, ymax = np.min(y_coords), np.max(y_coords)
	R = np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2) # really diameter
	R /= 2
	dropradii.append(R)
	print(f'Drop radius: {R}')

	print(f'Atoms in first slice: {len(slice0_coords)}')
	print(f'Atoms in second slice: {len(slice1_coords)}')
	r0, r1, theta = wettingAngle(slice0_coords, slice1_coords)
	r0s.append(r0)
	r1s.append(r1) # for debugging
	thetas.append(theta)

timesteps = np.arange(0,len(thetas),1) # thermo_interval
theta_bar = np.mean(thetas)

print(f'Time-averaged wetting angle: {theta_bar}')
out = np.vstack((timesteps,dropradii,thetas)) # r0s as nominal radii of each droplet
with open(args.output,'wb') as f: np.save(f,out)


	



