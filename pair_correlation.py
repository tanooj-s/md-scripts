# compute g(r) for a LAMMPS data file, plot out, output a numpy file, print out intrinsic cutoff 

import argparse
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description="compute g(r) for a data file")
parser.add_argument("-i", action="store", dest="input")
parser.add_argument("-dr", action="store", dest="dr")
args = parser.parse_args()
dr = float(args.dr)
def purge(tokens): return [t for t in tokens if len(t) >= 1]
def isfloat(s):
	try:
		float(s)
		return True
	except ValueError:
		return False
# ---- parse LAMMPS data file -----
atom_data = []
xlo, xhi, ylo, yhi, zlo, zhi = 0, 0, 0, 0, 0, 0
atom_style = "atomic"
elements = []
with open(args.input,'r') as f:
	lines = f.readlines()
	for l in lines:
		tokens = purge(l.strip('\n').split(' '))
		checksum = np.sum([not isfloat(t) for t in tokens])
		if (len(tokens) == 4) and (tokens[-1] == "xhi"):
			xlo, xhi = float(tokens[0]), float(tokens[1])
		if (len(tokens) == 4) and (tokens[-1] == "yhi"):
			ylo, yhi = float(tokens[0]), float(tokens[1])
		if (len(tokens) == 4) and (tokens[-1] == "zhi"):
			zlo, zhi = float(tokens[0]), float(tokens[1])
		if (len(tokens) == 3) and (tokens[0] == "Atoms"):
			atom_style = tokens[-1]
		if (len(tokens) == 4) and (tokens[-2] == "#"): 
			elements.append(tokens[-1])
		if (checksum == 0) and (len(tokens) == 9):
			atom_data.append(tokens)
		if atom_style == "atomic":
			if (checksum == 0) and (len(tokens) == 5):
				atom_data.append(tokens)
		elif atom_style == "charge":
			if (checksum == 0) and (len(tokens) == 6):
				atom_data.append(tokens)
		elif atom_style == "full":
			if (checksum == 0) and (len(tokens) >= 7):
				atom_data.append(tokens)
bounds = [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
L = np.array([xhi-xlo,yhi-ylo,zhi-zlo])
V = L[0]*L[1]*L[2]
# create map from atomID to [type,position]
atom_map = {}
for datum in atom_data:
	if atom_style == "atomic": atom_map[int(datum[0])] = np.array([float(datum[1]),float(datum[2]),float(datum[3]),float(datum[4])])
	elif atom_style == "charge": atom_map[int(datum[0])] = np.array([float(datum[1]),float(datum[3]),float(datum[4]),float(datum[5])])
	elif atom_style == "full": atom_map[int(datum[0])] = np.array([float(datum[2]),float(datum[4]),float(datum[5]),float(datum[6])])
max_dist = 0.5 * np.max(L)
nbins = int(max_dist/dr) + 1
r = dr * np.arange(0, nbins, 1)
N = len(sorted(list(atom_map.keys())))
n = N / V

# --- generalize to arbitrary species ---
species = sorted(set(int(atom_map[i][0]) for i in atom_map))
pairs = [(a, b) for idx, a in enumerate(species) for b in species[idx:]]  # unique pairs (a,b) a<=b
pair_gr = {p: np.zeros(nbins) for p in pairs}

for i in sorted(list(atom_map.keys())):
	ri = atom_map[i][1:]
	type_i = int(atom_map[i][0])
	for j in range(1, i):
		rj = atom_map[j][1:]
		type_j = int(atom_map[j][0])
		dx = ri[0] - rj[0]
		dy = ri[1] - rj[1]
		dz = ri[2] - rj[2]
		dx = dx - round(dx/L[0])*L[0]
		dy = dy - round(dy/L[1])*L[1]
		dz = dz - round(dz/L[2])*L[2]
		dist = (dx**2 + dy**2 + dz**2) ** 0.5
		if dist <= max_dist:
			idx = int(dist/dr)
			pair_key = (min(type_i, type_j), max(type_i, type_j))
			pair_gr[pair_key][idx] += 1

# normalization
for p in pairs:
	pair_gr[p] *= 2
	pair_gr[p][1:] /= 4 * np.pi * (r[1:]**2) * dr
	pair_gr[p] /= n
	pair_gr[p] /= N

# --- intrinsic neighborhood cutoff: lowest r capturing first peak of ALL pairs ---
def first_peak_cutoff(gr_arr, r_arr, min_idx=2, tol=1e-10):
    """Return r just after the first peak drops back to zero."""
    # find first non-zero bin (start of first peak region)
    for k in range(min_idx, len(gr_arr) - 1):
        if gr_arr[k] > tol:  # entered the peak
            # now find first zero after the peak
            for m in range(k+1, len(gr_arr)):
                if gr_arr[m] <= tol:
                    return r_arr[m]
            return r_arr[k]  # peak never drops to zero, return peak position
    return None

cutoffs = {}
for p in pairs:
    cutoffs[p] = first_peak_cutoff(pair_gr[p], r)
intrinsic_cutoff = max(v for v in cutoffs.values() if v is not None)
print(f"{intrinsic_cutoff:.4f}")

material = args.input.split('_')[0]
plt.rcParams["figure.figsize"] = (8,2*len(pairs))
fig, axes = plt.subplots(len(pairs),1,sharex=True)
for i in range(len(pairs)):
	ai, bi = pairs[i]
	axes[i].plot(r,pair_gr[(ai,bi)],label=f"{elements[ai-1]}-{elements[bi-1]}")
	axes[i].legend(loc="upper right")
	axes[i].set_ylabel("g (r)")
	axes[i].axvline(intrinsic_cutoff,color='k',linestyle="dotted")
	axes[i].grid()
axes[-1].set_xlabel(f"r (Å)")
axes[0].set_title(material)
plt.xlim(0,8)
plt.tight_layout()
plt.savefig(f"{args.input[:-4]}_rdf.png")
plt.clf()

outfile = args.input.split('.')[0] + "_rdf.npy"
rows = [r] + [pair_gr[p] for p in pairs]
with open(outfile, 'wb') as f: np.save(f, np.vstack(rows))