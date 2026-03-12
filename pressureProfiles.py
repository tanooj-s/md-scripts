# bin and plot 1d pressure as a function of z 
# arguments are inputfile, dz, zlo, zhi


import numpy as np
import os
import sys
import matplotlib.pyplot as plt

inputfile = sys.argv[1]
dz = float(sys.argv[2])
zlo = float(sys.argv[3])
zhi = float(sys.argv[4]) # these you need to rationally set from the dump file in ovito 

# unit conversion factors
eV_per_cubic_angstrom_to_gigapascal = 160.2176634 

# ---------------

def purge(tokens): 
	return [t for t in tokens if len(t) >= 1]

def moving_avg(x,n): 
	# moving average of multiple numpy arrays
	if len(x.shape) > 1:
		avg_arr = []
		for i in range(x.shape[1]): 
			avg_arr.append(np.convolve(x[:,i],np.ones(n),mode='valid')/n)
		avg_arr = np.array(avg_arr).T
		return avg_arr
	else:
		return np.convolve(x,np.ones(n),mode='valid') / n

def isfloat(s): 
	# hack function to confirm tokens are numeric as floats
	try:
		float(s)
		return True
	except ValueError:
		return False

# ----------------

# two passes 
# compute trajectory of box bounds, atom_data in first pass
# then bin data afterwards 
# you can just keep appending and reshape afterwards
# wtf
# all my old code was so shit 


timesteps = []
natoms = [] 
bounds = []
# timeseries of profiles
atom_data = [] # atom_data reshaping only works if N is constant 

# timeseries of (3,2) shaped objects 
with open(inputfile,'r') as f:
	previousline = ""
	for line in f:
		tokens = purge(line.strip('\n').split(' '))
		if len(tokens) == 2:
			checksum = np.sum([not isfloat(t) for t in tokens])
			if checksum == 0:
				bounds.append(np.array([float(tokens[0]),float(tokens[1])]))
		if len(tokens) == 1 and previousline.startswith("ITEM: TIMESTEP"):
			timesteps.append(int(tokens[0]))
		if len(tokens) == 1 and previousline.startswith("ITEM: NUMBER"):
			natoms.append(int(tokens[0]))
		elif len(tokens) > 6 and not line.startswith("ITEM: ATOMS"):
			# we only want x, y, z, c_1[1], c_1[2], c_1[3]
			x, y, z = float(tokens[2]), float(tokens[3]), float(tokens[4])
			sxx, syy, szz = float(tokens[5]), float(tokens[6]), float(tokens[7])
			atom_data.append(np.array([x,y,z,sxx,syy,szz]))
		previousline = line
timesteps = np.array(timesteps)
bounds = np.array(bounds)
atom_data = np.array(atom_data)
natoms = np.array(natoms) 

# bin separately 
N = natoms[0] # this isn't changing in regular sims
bounds = bounds.reshape((timesteps.shape[0],int(bounds.shape[0]/timesteps.shape[0]),2))
atom_data = atom_data.reshape((timesteps.shape[0],int(atom_data.shape[0]/timesteps.shape[0]),atom_data.shape[1]))
print(bounds.shape)
print(atom_data.shape)
assert bounds.shape[0] == atom_data.shape[0]



# now actually compute profiles 
nbins = int((zhi-zlo)/dz) + 1
# histogram arrays, implicitly as a function of time averaged later
counts_z = []
Pzz_z = []
Pxx_z = []
Pyy_z = []

# collect data 
for t in range(bounds.shape[0]):
	xlo = bounds[t][0][0]
	xhi = bounds[t][0][1]
	ylo = bounds[t][1][0]
	yhi = bounds[t][1][1]
	Lx = xhi - xlo
	Ly = yhi - ylo
	A = Lx * Ly
	print(A)
	# histograms at this timestep
	counts_zt = np.zeros(nbins,)
	Pzz_zt = np.zeros(nbins,)
	Pxx_zt = np.zeros(nbins,)
	Pyy_zt = np.zeros(nbins,)

	for i in range(atom_data[t].shape[0]):
		zi = atom_data[t][i][2]
		idx = int((zi-zlo)/dz)
		pxx_it = -atom_data[t][i][3]
		pyy_it = -atom_data[t][i][4]
		pzz_it = -atom_data[t][i][5]
		Pzz_zt[idx] += pzz_it
		Pxx_zt[idx] += pxx_it
		Pyy_zt[idx] += pyy_it
		counts_zt += 1

	# number normalization
	Pzz_zt = np.divide(Pzz_zt,counts_zt,out=np.zeros_like(Pzz_zt,dtype=float),where=counts_zt!=0)
	Pxx_zt = np.divide(Pxx_zt,counts_zt,out=np.zeros_like(Pxx_zt,dtype=float),where=counts_zt!=0)
	Pyy_zt = np.divide(Pyy_zt,counts_zt,out=np.zeros_like(Pyy_zt,dtype=float),where=counts_zt!=0)
	# volume normalization
	Pzz_zt /= dz*A
	Pxx_zt /= dz*A
	Pyy_zt /= dz*A

	Pzz_z.append(Pzz_zt)
	Pxx_z.append(Pxx_zt)
	Pyy_z.append(Pyy_zt)


# cast and convert cubic angstrom to gigapascal
Pzz_z = np.array(Pzz_z) * eV_per_cubic_angstrom_to_gigapascal
Pxx_z = np.array(Pxx_z) * eV_per_cubic_angstrom_to_gigapascal
Pyy_z = np.array(Pyy_z) * eV_per_cubic_angstrom_to_gigapascal

print(Pzz_z.shape)
print(Pxx_z.shape)
print(Pyy_z.shape)

Pzz_z = np.mean(Pzz_z,axis=0)
Pxx_z = np.mean(Pxx_z,axis=0)
Pyy_z = np.mean(Pyy_z,axis=0)


# convert to normal and tangential 
Pn_z = Pzz_z
Pt_z = 0.5*(Pxx_z+Pyy_z)


# moving average to plot out 
# denoise
moving_avg_wdw = 20
Pn_z = moving_avg(Pn_z,moving_avg_wdw)
Pt_z = moving_avg(Pt_z,moving_avg_wdw)
z = dz*np.arange(0,Pn_z.shape[0],1)



# plot for debugging

plt.rcParams["figure.figsize"] = (8,9)
fig, axes = plt.subplots(3,1,sharex=True)
axes[0].plot(z,Pn_z,label="Pn")
axes[1].plot(z,Pt_z,label="Pt")
axes[2].plot(z,Pn_z-Pt_z,label="Pn-Pt")
plt.tight_layout()
plt.legend(loc="upper right")
plt.savefig("pressure_profile_test.png")

exit()


# clearly glass interface sims are weird so need to confirm whether this is a valid strategy at all for gamma incorporating entropy
# but should be valid for vacuum sims 
# so I guess what I'm seeing is a nonzero pressure in the glass, but close to zero for CoSi

