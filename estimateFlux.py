# prototype algorithm to estimate charge flux as a vector field
# since these are fixed nodes we only estimate flux from nearest neighbors 
# only need to build the neighbor list once 
# should be applicable to more than MD

import argparse
import numpy as np
import time
from tqdm import tqdm

parser = argparse.ArgumentParser(description="convert MD data into a grid")
parser.add_argument('-i', action="store", dest="input")
parser.add_argument('-o', action="store", dest="output")
parser.add_argument('-lat',action="store", dest="lattice") # 2D lattice type, either hc for honeycomb or sq for square
parser.add_argument('-lc',action="store", dest="lc") # lattice constant / distance to neighbors
parser.add_argument('-start', action="store", dest="start") # collection start timestep
parser.add_argument('-end', action="store", dest="end") # collection start timestep
parser.add_argument('-nevery',action="store", dest="nevery") # interval in timesteps to calculate at
parser.add_argument('-thresh',action="store", dest="thresh") # distance threshold for nearest neighbors, fixed
parser.add_argument('-t', action="store", dest="atom_types") # atom types to assess flux over
args = parser.parse_args()

lat = args.lattice
lc = float(args.lc) # 1.42A for honeycomb graphene, 1A for test square lattice
nevery = int(args.nevery)
start = int(args.start)
end = int(args.end)
thresh = float(args.thresh)
atom_types = [int(t) for t in args.atom_types.split(' ')]


def minimumImage(r,L):
	r -= L*np.array([round(i) for i in r/L])
	return r # minimum image convention for PBCs

def purge(tokens): 
	return [t for t in tokens if len(t) >= 1] # purge empty strings from token lists

def isfloat(s): # hack function to confirm tokens are numeric as floats
	try:
		float(s)
		return True
	except ValueError:
		return False


# interface nodes
class Node:
	def __init__(self,atom_id,atom_type,q,x,y,z,L):
		self.id = atom_id
		self.type = atom_type
		self.q0 = q
		self.q1 = 0 # charge at two timesteps for current as a scalar 
		self.curr = 0 # the scalar quantity that flows in time
		self.r = np.array([x,y,z])
		self.neighbors = [] # 3 for honeycomb, 4 for square
		self.area = 1 # make this an arg later, area per interface particle for square lattice test 
		self.L = L # box bounds for correct PBC distances

	def updateCharge(self,q): 
		# assume this is called every timestep except t=0
		# q is charge at next timestep read in from dump file 
		self.q0 = self.q1
		self.q1 = q
		self.curr = self.q1 - self.q0

	def getFlux(self):
		# flux as a local vector 
		# we only want the (x,y) coords here
		assert len(self.neighbors) > 0
		flux = np.zeros(3,)
		# loop over neighbors, 
		# compute scalar current for each 
		# then weight each displacement vector by the difference in scalar current
		for nbr in self.neighbors:
			# !! make sure this current is only getting updated once for every node
			rij = minimumImage(nbr.r - self.r,self.L) # vector displacement
			d_curr = self.curr - nbr.curr
			flux += rij * d_curr # normalize by d_curr/some charge for a velocity vector field
		flux /= self.area
		return flux


# ---- initial pass over file for idxs ----
# infer dump format from first 10 lines
idIdx, typeIdx, xIdx, yIdx, zIdx, qIdx = 0, 0, 0, 0, 0, 0
xlo, xhi, ylo, yhi, zlo, zhi = 1000,-1000,1000,-1000, 1000,-1000
nToks = 0 # number of tokens in each atomic data line, needed for later pass through
with open(args.input,'r') as f:
	counter = 0
	for counter in range(15):
		line = f.readline()
		line = line.strip('\n')
		if line.startswith("ITEM: ATOMS"):
			tokens = purge(line.split(' '))
			print(tokens)
			nToks = len(tokens) - 2
			idIdx = tokens.index('id') - 2
			typeIdx = tokens.index('type') - 2
			xIdx = tokens.index('x') - 2
			yIdx = tokens.index('y') - 2
			zIdx = tokens.index('z') - 2
			qIdx = tokens.index('q') - 2
		elif counter == 5:
			tokens = purge(line.split(' '))
			print(tokens)
			xlo, xhi = float(tokens[0]), float(tokens[1].strip('\n'))
		elif counter == 6:
			tokens = purge(line.split(' '))
			print(tokens)
			ylo, yhi = float(tokens[0]), float(tokens[1].strip('\n'))
		elif counter == 7:
			tokens = purge(line.split(' '))
			print(tokens)
			zlo, zhi = float(tokens[0]), float(tokens[1].strip('\n'))
		counter += 1
print(f'{idIdx} {typeIdx} {xIdx} {yIdx} {zIdx} {qIdx}')
print(f'xlo xhi {xlo} {xhi}')
print(f'ylo yhi {ylo} {yhi}')
print(f'zlo zhi {zlo} {zhi}')


# ---- second pass for fixed box bounds -----
# read in box bounds
L = np.array([xhi-xlo,yhi-ylo,zhi-zlo])
print(L)

# ---- third pass to construct fixed neighbor lists ----
# if honeycomb, look for three nearest neighbors
# if square, look for 4 nearest neighbors 
ctr = 0
t1 = False
nodes = [] # initial list of fixed nodes to find neighbors from
with open(args.input,'r') as f:
	while t1 == False:
		line = f.readline()
		if (ctr > 0) and line.startswith("ITEM: TIMESTEP"):
			t1 = True
		tokens = purge(line.strip('\n').split(' '))
		if len(tokens) == nToks:
			print(tokens)
			checksum = np.sum([not isfloat(t) for t in tokens]) # make sure only using lines where everything is numeric
			if checksum == 0:
				aID, aType, x, y, z, q = int(tokens[idIdx]) , int(tokens[typeIdx]), float(tokens[xIdx]), float(tokens[yIdx]), float(tokens[zIdx]), float(tokens[qIdx])
				nodes.append(Node(atom_id=aID,atom_type=aType,q=q,x=x,y=y,z=z,L=L))
		ctr += 1
print(f'{ctr} lines read before hitting next timestep')

# construct nodeMap, 
# later may want make a NodeList class to use __getitem__ to access nodes more Pythonically
nodeMap = {}
for n in nodes:
	nodeMap[n.id] = n

# ---- build neighbor lists ----
print(f"Building fixed neighbor lists...")
for n_i in nodes:
	for n_j in nodes:
		if n_i.id != n_j.id:
			rij = minimumImage(n_i.r - n_j.r,L)
			dist = np.sqrt(np.dot(rij,rij))
			if dist <= thresh:
				n_i.neighbors.append(n_j)

# confirm each neighbor list is the correct length based on 2D lattice
lim = 3 if lat == 'hc' else 4 # number of nearest neighbors for lattice type
for i in nodes:
	print(f"Node: {i.id}")
	assert len(i.neighbors) == lim
	for j in i.neighbors: # ids of neighbors
		rij = minimumImage(i.r - j.r,L) # divide by lc to check geometry, but otherwise retain the actual displacement
		print(f" -- Nbr: {j.id} | r: {rij} ")


# ---- fourth pass, parse trajectory ----
# generate vector field of flux at every timestep
fields = [] # time-series of flux vector fields
currentTime = 0
nCollected = 0
starttime = time.time()
with open(args.input,'r') as f:
	for line in tqdm(f):
		tokens = purge(line.strip('\n').split(' '))
		# first check timestep
		if len(tokens) == 1 and previousLine.startswith("ITEM: TIMESTEP"):
			currentTime = int(tokens[0])	
		if currentTime > end: 
			break
		if (currentTime >= start) and (currentTime % nevery == 0):
			if len(tokens) == 1 and previousLine.startswith("ITEM: NUMBER OF ATOMS"):
				nodes = [] # create new list of all nodes at this timestep
				flux = np.zeros(4,) # (x,y,z,flux_x,flux_y), later make this a 5-tuple to split by interface
				nCollected += 1 # number of timesteps collected
			if len(tokens) == nToks:
				checksum = np.sum([not isfloat(t) for t in tokens]) # make sure only using lines where everything is numeric
				if checksum == 0:
					# parse atomic data here
					# update charge of individual nodes, then append to nodelist for this timestep
					aType, aID, x, y, z, q = int(tokens[typeIdx]), int(tokens[idIdx]), float(tokens[xIdx]), float(tokens[yIdx]), float(tokens[zIdx]), float(tokens[qIdx])
					node = nodeMap[aID]
					node.updateCharge(q=q)
					if z < 2: # hack, only look at one interface for now
						nodes.append(node)
			# once all nodes have been updated, compute flux at this timestep
			if line.startswith("ITEM: TIMESTEP") and currentTime != 0: 
				field_t = [] # vector field of flux at this timestep
				for node in nodes:
					flux = node.getFlux() 
					local_field = np.array([node.r[0],node.r[1],flux[0],flux[1]]) # flux vector at this r
					field_t.append(local_field)
				field_t = np.array(field_t)
				fields.append(field_t) # append to timeseries of vector fields 
		previousLine = line

fields = np.array(fields)
print(f"Shape of time-series of vector fields: {fields.shape}")

print(f"{time.time()-starttime} seconds")
print(f"{nCollected} timesteps collected for analysis")
with open(f'{args.output}','wb') as f: np.save(f,fields) 





