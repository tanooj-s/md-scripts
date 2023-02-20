# add bonded interactions to a carbon lattice
# initial lattice is assumed to be generated via atomsk, a LAMMPS input script or any other method that generates a periodic lattice

# add angles and bonds for water as well here

# we know the topology we want for this system

# 2 bond types - O-H and C-C
# 2 angle types - H-O-H and C-C-C
# 1 dihedral type - C-C-C-C


import numpy as np
import os
import argparse
import time

parser = argparse.ArgumentParser(description="add bond, angle and dihedral connectivity to a molecular mechanics system")
parser.add_argument("-i", action="store", dest="infile")
parser.add_argument("-o", action="store", dest="outfile")
parser.add_argument("-b", action="store", dest="bonds", const='y', nargs='?') # string of bond types
parser.add_argument("-a", action="store", dest="angles", const='y', nargs='?') # pass in an empty string '' if no angles
parser.add_argument("-d", action="store", dest="dihedrals", const='y', nargs='?') # pass in an empty string '' if no dihedrals
parser.add_argument("-t", action="store", dest="thresholds", const='y', nargs='?') # list of min dist thresholds for each 2-bond, throw an error if not consistent with bond type info 
args = parser.parse_args()

# add args for atom types to generate bonds for

# purge all empty strings from token lists
def purge(tokens): return [t for t in tokens if len(t) >= 1]


def minimumImage(r):
	'''
	apply minimum image convention for displacements
	assume we have a 3-tuple passed in
	which is x1-x0, y1-y0, z1-z0 for two atoms
	'''
	r -= L*np.array([round(i) for i in r/L])
	return r


# class definitions for each entity 
class Atom:
	def __init__(self,atom_id,mol_id,atom_type,charge,x,y,z):
		self.atom_id = atom_id
		self.mol_id = mol_id
		self.atom_type = atom_type
		self.charge = charge
		self.position = np.array([x,y,z])

class Bond:
	def __init__(self,bond_id,bond_type,atom1,atom2):
		self.bond_id = bond_id
		self.bond_type = bond_type
		self.atom1 = atom1
		self.atom2 = atom2

class Angle:
	def __init__(self,angle_id,angle_type,bond1,bond2,leftAtom,sharedAtom,rightAtom):
		self.angle_id = angle_id
		self.angle_type = angle_type
		self.bond1 = bond1
		self.bond2 = bond2
		self.leftAtom = leftAtom
		self.sharedAtom = sharedAtom
		self.rightAtom = rightAtom

class Dihedral:
	def __init__(self,dihedral_id,dihedral_type,angle1,angle2,startAtom,midAtom1,midAtom2,endAtom):
		self.dihedral_id = dihedral_id
		self.dihedral_type = dihedral_type
		self.angle1 = angle1
		self.angle2 = angle2
		self.startAtom = startAtom
		self.midAtom1 = midAtom1
		self.midAtom2 = midAtom2
		self.endAtom = endAtom


# ---- parse args for user-specified bond definitions ----
infile = args.infile

bondString = args.bonds
bondTokens = [int(t) for t in bondString.split(' ')]
# populate dicts from user-specified bond definitions
bondDefs = {} # e.g. {1: [1,2], 2: [6,7]} bond type 1 between atom types 1 and 2, bond type 2 between atom types 6 and 7
for i in range(len(bondTokens)-2):
	if i % 3 == 0: bondDefs[bondTokens[i]] = [bondTokens[i+1], bondTokens[i+2]]

thresholds = [float(t) for t in args.thresholds.split(' ')]
assert len(thresholds) == len(bondDefs.keys())


# ---- lists of relevant entities ----

masses = []
atoms = []
bonds = [] # all 2-bond objects in system
angles = [] # all 3-bond objects
dihedrals = [] # all 4-bond objects
bounds = [[0,0],[0,0],[0,0]] # box dimensions, xlo xhi ylo yhi zlo zhi
n_types = 0



# ---- read in non-bonded datafile ----

with open(infile,'r') as f:
	lines = f.readlines()
lines = [l.strip('\n') for l in lines]
atomstart = lines.index('Atoms # full')

# read in number of atom types 
for l in lines[:20]:
	tokens = purge(l.split(' '))
	if (len(tokens) > 0) and (tokens[-1] == 'types'): n_types = int(tokens[0])

# read in box bounds
for line in lines[:atomstart]:
	tokens = purge(line.split(' '))
	if len(tokens) > 0:
		if tokens[-1] == 'xhi':	bounds[0][0], bounds[0][1] = float(tokens[0]), float(tokens[1])
		elif tokens[-1] == 'yhi': bounds[1][0], bounds[1][1] = float(tokens[0]), float(tokens[1])	 			
		elif tokens[-1] == 'zhi': bounds[2][0], bounds[2][1] = float(tokens[0]), float(tokens[1])
L = np.array([bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0], bounds[2][1] - bounds[2][0]])

# read in masses and atomic coordinates
with open(infile,'r') as f:
	lines = f.readlines()
lines = [l.strip('\n') for l in lines]
mass_start = lines.index('Masses')
atomstart = lines.index('Atoms # full')
atomend = lines.index('Bonds') if 'Bonds' in lines else len(lines)
#atomend = lines.index('Velocities') if 'Velocities' in lines else len(lines)

for line in lines[mass_start:atomstart]:
		tokens = purge(line.split(' '))
		if len(tokens) > 1:
			masses.append([int(tokens[0]), float(tokens[1])])


for line in lines[atomstart:atomend]:
	tokens = purge(line.split(' '))
	if len(tokens) > 3:
		atoms.append(Atom(atom_id=int(tokens[0]),
						  mol_id=int(tokens[1]), 
						  atom_type=int(tokens[2]),
						  charge=float(tokens[3]),
						  x=float(tokens[4]),
						  y=float(tokens[5]),
						  z=float(tokens[6]))) 


# build out a map to directly access atoms by IDs in case needed 
atomMap = {}
for a in atoms:
	atomMap[a.atom_id] = a

# build out a map that stores lists of each atom by type, useful for generating 2-bonds efficiently
atomsByType = {}
for aType in range(1,1+n_types):
	atomsByType[aType] = [a for a in atoms if a.atom_type == aType]


# ----- generate 2-bonds -----

print('Bond definitions')
print(bondDefs)
print('Iterating over pairs of atoms...')
# iterate over bond definitions and create bonds 
tCounter = 0 # counter to access threshold values for each bond type
bondCounter = 1 # bondID to pass into lammps
start = time.time()
for bType, [aType1, aType2] in bondDefs.items():
	type1_atoms = atomsByType[aType1]
	type2_atoms = atomsByType[aType2]
	threshold = thresholds[tCounter]
	for atom_i in type1_atoms:
		for atom_j in type2_atoms:
			if atom_i.atom_id < atom_j.atom_id:
				rij = minimumImage(atom_i.position - atom_j.position)
				dist = np.sqrt(np.dot(rij,rij))
				if dist <= threshold:
					bonds.append(Bond(bond_id=bondCounter,
									  bond_type=bType,
									  atom1=atom_i,
									  atom2=atom_j))
					bondCounter += 1
	tCounter += 1
end = time.time()
print(f"Took {end-start} to find and create 2-bonds")
print(f'{len(bonds)} bonds created')





# ------ generate 3-bonds (angles) -----

# an angle is defined by two bonds that share an atom

# iterate over pairs of bonds
# find bond pairs where the total number of distinct atoms is 3 
# use the common atom as the center (shared) atom of the angle 

# create a list of atom triplets that satisfy the above criteria
# find atom triplets that satisfy the criterion used to define each angle type

angleString = args.angles
if len(angleString) > 0:
	angleTokens = [int(t) for t in angleString.split(' ')]
	angleDefs = {}
	for i in range(len(angleTokens)-3):
		if i % 4 == 0: angleDefs[angleTokens[i]] = [angleTokens[i+1], angleTokens[i+2], angleTokens[i+3]]


	print('Angle definitions')
	print(angleDefs)
	print('Iterating over pairs of 2-bonds...')

	start = time.time()
	if args.angles != 'n': # figure out how to handle these edge cases appropriately
		angleCounter = 1
		for bond_i in bonds:
			for bond_j in bonds:
				sharedAtom = 0 # atom shared by the two bonds
				leftAtom = 0
				rightAtom = 0 # left/right are arbitrary
				atomTriplets = [] # add triplets in order here - left, shared, right

				if bond_i.bond_id < bond_j.bond_id:
					distinctAtoms = list(set([bond_i.atom1.atom_id,bond_i.atom2.atom_id,bond_j.atom1.atom_id,bond_j.atom2.atom_id]))
					if len(distinctAtoms) == 3:
						angleAtomTypes = [str(atomMap[a].atom_type) for a in distinctAtoms]					
						if (bond_i.atom1.atom_id == bond_j.atom1.atom_id):
							sharedAtom = bond_i.atom1
							leftAtom = bond_i.atom2
							rightAtom = bond_j.atom2						
						elif (bond_i.atom2.atom_id == bond_j.atom1.atom_id):
							sharedAtom = bond_i.atom2
							leftAtom = bond_i.atom1
							rightAtom = bond_j.atom2						
						elif (bond_i.atom1.atom_id == bond_j.atom2.atom_id):
							sharedAtom = bond_i.atom1
							leftAtom = bond_i.atom2
							rightAtom = bond_j.atom1						
						elif (bond_i.atom2.atom_id == bond_j.atom2.atom_id):
							sharedAtom = bond_i.atom2
							leftAtom = bond_i.atom1
							rightAtom = bond_j.atom1						
						atomTriplets.append([leftAtom, sharedAtom, rightAtom])

						# need a snippet to make sure there aren't duplicates?

						for angType, [aType1, aType2, aType3] in angleDefs.items():
							for [leftAtom, sharedAtom, rightAtom] in atomTriplets: # ideally you only want to iterate over this list of triplets once 
								if (leftAtom.atom_type == aType1) and (sharedAtom.atom_type == aType2) and (rightAtom.atom_type == aType3):
									angles.append(Angle(angle_id=angleCounter,
														angle_type=angType,
														bond1=bond_i,
														bond2=bond_j,
														leftAtom=leftAtom,
														sharedAtom=sharedAtom,
														rightAtom=rightAtom))
									angleCounter += 1

	end = time.time()
	print(f"Took {end-start} to find and create 3-bonds")
	print(f'{len(angles)} 3-bonds (angles) created')





# ------ generate 4-bonds (dihedrals) -----

# a dihedral is defined by two angles sharing a bond
# constraints:
# the 4 atoms must be distinct
# the middle atom of the two angles must be different

#  |
# / \
# this is not a dihedral

# \
#  |
# /
# this is a dihedral


# iterate over pairs of angles
# find pairs where the number of distinct atoms is 4 
# if the middle (shared) atom of both angles is different then create a dihedral
# create an atom quadruplets list for this, check types based on definition (same as for angles)


dihedralString = args.dihedrals
if len(dihedralString) > 0:
	dihedralTokens = [int(t) for t in dihedralString.split(' ')]
	dihedralDefs = {}
	for i in range(len(dihedralTokens)-4):
		if i % 5 == 0: dihedralDefs[dihedralTokens[i]] = [dihedralTokens[i+1], dihedralTokens[i+2], dihedralTokens[i+3], dihedralTokens[i+4]]

	print('Dihedral definitions')
	print(dihedralDefs)
	print('Iterating over pairs of 3-bonds...')

	start = time.time()
	if args.dihedrals != 'n':
		dihedralCounter = 1
		for angle_i in angles:
			for angle_j in angles:
				startAtom = 0
				midAtom1 = 0
				midAtom2 = 0
				endAtom = 0
				atomQuadruplets = []

				if angle_i.angle_id < angle_j.angle_id:
					angle_i_atoms = [angle_i.leftAtom, angle_i.sharedAtom, angle_i.rightAtom]
					angle_j_atoms = [angle_j.leftAtom, angle_j.sharedAtom, angle_j.rightAtom]
					distinctAtoms = list(set([angle_i.leftAtom.atom_id,
											 angle_i.sharedAtom.atom_id,
											 angle_i.rightAtom.atom_id,
											 angle_j.leftAtom.atom_id,
											 angle_j.sharedAtom.atom_id,
											 angle_j.rightAtom.atom_id]))

					if (len(distinctAtoms) == 4) and (angle_i.sharedAtom != angle_j.sharedAtom): # 4 distinct atoms and different shared atom
						if angle_i.leftAtom not in angle_j_atoms:
						# then we know exactly which atom is which
							startAtom = angle_i.leftAtom # note this means angle_i.rightAtom === angle_j.sharedAtom here
							midAtom1 = angle_i.sharedAtom
							midAtom2 = angle_i.rightAtom
							endAtom = list(set(angle_j_atoms) - set(angle_i_atoms))[0] 
						elif angle_i.rightAtom not in angle_j_atoms: 
							startAtom = angle_i.rightAtom # angle_i.leftAtom === angle_j.sharedAtom here
							midAtom1 = angle_i.sharedAtom
							midAtom2 = angle_i.leftAtom
							endAtom = list(set(angle_j_atoms) - set(angle_i_atoms))[0]
						atomQuadruplets.append([startAtom, midAtom1, midAtom2, endAtom])

						for dhType, [aType1, aType2, aType3, aType4] in dihedralDefs.items():
							for [startAtom, midAtom1, midAtom2, endAtom] in atomQuadruplets:
								if (startAtom.atom_type == aType1) and (midAtom1.atom_type == aType2) and (midAtom2.atom_type == aType3) and (endAtom.atom_type == aType4):
									dihedrals.append(Dihedral(dihedral_id=dihedralCounter,
															  dihedral_type=dhType,
															  angle1=angle_i,
															  angle2=angle_j,
															  startAtom=startAtom,
															  midAtom1=midAtom1,
															  midAtom2=midAtom2,
															  endAtom=endAtom))
									dihedralCounter += 1

	end = time.time()
	print(f"Took {end-start} to find and create 4-bonds")
	print(f'{len(dihedrals)} 4-bonds (dihedrals) created')
	



# ------- write data file ------

print(f"Writing to {args.outfile}")
with open(args.outfile,'w') as f:
	f.write('LAMMPS data file with molecular mechanics topology built out\n')

	f.write(f'{len(atoms)} atoms\n')
	f.write(f'{len(bonds)} bonds\n')
	if len(angleString) > 0: f.write(f'{len(angles)} angles\n')
	if len(dihedralString) > 0: f.write(f'{len(dihedrals)} dihedrals\n')
	f.write(f'{n_types} atom types\n')

	f.write(f'{len(list(bondDefs.keys()))} bond types\n')
	if len(angleString) > 0: f.write(f'{len(list(angleDefs.keys()))} angle types\n') 
	if len(dihedralString) > 0: f.write(f'{len(list(dihedralDefs.keys()))} dihedral types\n') 

	f.write(f'{bounds[0][0]} {bounds[0][1]} xlo xhi\n')
	f.write(f'{bounds[1][0]} {bounds[1][1]} ylo yhi\n')
	f.write(f'{bounds[2][0]} {bounds[2][1]} zlo zhi\n\n')

	f.write('Masses\n\n') # read in from file
	for m in masses:
		f.write(f'{m[0]} {m[1]}\n')
	f.write('\n')

	f.write('Atoms # full\n\n')
	for a in atoms:
		f.write(f'{a.atom_id} {a.mol_id} {a.atom_type} {a.charge} {a.position[0]} {a.position[1]} {a.position[2]} \n')
	f.write('\n')

	f.write('Bonds\n\n')
	for b in bonds:
		f.write(f'{b.bond_id} {b.bond_type} {b.atom1.atom_id} {b.atom2.atom_id} \n')
	f.write('\n')

	if len(angleString) > 0:
		f.write('Angles\n\n')
		for a in angles:
			f.write(f'{a.angle_id} {a.angle_type} {a.leftAtom.atom_id} {a.sharedAtom.atom_id} {a.rightAtom.atom_id} \n')
		f.write('\n')

	if len(dihedralString) > 0:
		f.write('Dihedrals\n\n')
		for d in dihedrals: 
			f.write(f'{d.dihedral_id} {d.dihedral_type} {d.startAtom.atom_id} {d.midAtom1.atom_id} {d.midAtom2.atom_id} {d.endAtom.atom_id} \n')





