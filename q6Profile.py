# compute z-resolved order parameter profile
# obtain estimates of interface position as a function of time

import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping

parser = argparse.ArgumentParser(description="calculate 1d temperature profiles")
parser.add_argument('-i', action="store", dest="input")
parser.add_argument('-dz', action="store", dest="dz") # bin width 
args = parser.parse_args()

dz = float(args.dz)
dt = 1e-3 # femtoseconds

temp = int(args.input.split('.')[0].split('_')[2].split('-')[-1])


def purge(tokens): return [t for t in tokens if len(t) >= 1] # purge empty strings from token lists

def isfloat(s): # hack function to confirm tokens are numeric as floats
	try:
		float(s)
		return True
	except ValueError:
		return False


# ---- relevant fitting code for position of interface ----
def tanh_model(z, D_cryst, D_liq, z0, w):
    """4-parameter sigmoid model for L1(r) profile."""
    return D_cryst + 0.5 * (D_liq - D_cryst) * (1 + np.tanh((z - z0) / w))

def loss_fn(params, z_data, D_data):
    D_cryst, D_liq, z0, w = params
    model = tanh_model(z_data, D_cryst, D_liq, z0, w)
    return np.sum((model - D_data)**2)

def fit_sigmoid_tanh(data):
    z_data = data[0]
    D_data = data[1]
    # Initial guesses
    D_cryst0 = np.min(D_data) # modify
    D_liq0 = np.max(D_data) # modify
    z0 = z_data[np.argmax(np.gradient(D_data))]  # initial guess for inflection
    w0 = 1.0
    x0 = [D_cryst0, D_liq0, z0, w0]
    # Optional bounds for sanity
    bounds = [(0, 1), (0, 1), (np.min(z_data), np.max(z_data)), (0.1, 10)]
    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "bounds": bounds,
        "args": (z_data, D_data),
    }
    result = basinhopping(loss_fn, x0, minimizer_kwargs=minimizer_kwargs, niter=500, disp=False)
    local_opt_result = result.lowest_optimization_result # local optimizer result, for inverse Hessian matrix to estimate error bars
    if local_opt_result.success:
        D_cryst, D_liq, R, w = result.x
        cov_matrix = local_opt_result.hess_inv.todense() # for error bars 
        param_errors = np.sqrt(np.diag(cov_matrix))
        return {
            'params': result.x,
            'error_bars': param_errors, 
            'fit_func': lambda r: tanh_model(r, *result.x),
            'residual': result.fun
        }
    else:
        raise RuntimeError("Fit did not converge")

# ----------------------------------------------------




# ---- single pass over file ----
# edit dimensions as appropriate for other directions 
# given box oscillations just set these values beforehand
minz, maxz = 0, 0
with open(args.input,'r') as f:
	nline = 0
	for line in f:
		tokens = purge(line.strip('\n').split(' '))
		if nline == 7: 
			minz, maxz = float(tokens[0]), float(tokens[1])
			break
		nline += 1

minz = -5
maxz = 85
print(f"minz: {minz}")
print(f"maxz: {maxz}")

# infer dump format from first 10 lines
idIdx, typeIdx, xIdx, yIdx, zIdx, q6Idx = 0, 0, 0, 0, 0, 0
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
			q6Idx = tokens.index("c_Q[2]") - 2


nbins = int((maxz-minz)/dz) + 1
# need a number density and a temperature array
rho = np.zeros(nbins,) # number density normalization array
q6_z = np.zeros(nbins,) # unnormalized order param array
timestep = 0 
# timeseries of order param profiles
q6_z_T = [] 
time = [] 
print("Parsing dump file for order parameter profiles...")
with open(args.input,'r') as f:
	nCollected = 0 # timesteps summed over
	doCollect = False
	currentTime = 0
	previousLine = '' # keep saving previous line 
	for line in f:
		lines = line.strip('\n')
		tokens = purge(line.split(' '))
		checksum = np.sum([not isfloat(t) for t in tokens])
		if currentTime > 20000:
			break
		if (len(tokens) > 2) and (checksum == 0):
			# bin data here 
			aType, z, q6 = int(tokens[typeIdx]), float(tokens[zIdx]), float(tokens[q6Idx])
			binIdx = int((z-minz)/dz)
			rho[binIdx] += 1
			q6_z[binIdx] += q6
		if (previousLine.startswith('ITEM: TIMESTEP')):
			timestep = int(tokens[0])
			print(f"t={timestep}")
			if (timestep > 0):
				currentTime = timestep
				time.append(dt*timestep)
				q6_z_T.append(np.divide(q6_z,rho,out=np.zeros_like(q6_z,dtype=float),where=rho!=0))
				rho = np.zeros(nbins,)
				q6_z = np.zeros(nbins,)
		previousLine = line

time = np.array(time)
q6_z_T = np.array(q6_z_T)
assert time.shape[0] == q6_z_T.shape[0]
print(q6_z_T.shape)
#for i in range(len(temp_T[0])): print(temp_T[0][i])

# this is mostly just to visually debug profiles
with open(args.input[:-5]+".q6_z.npz",'wb') as f: np.savez(f,array=q6_z_T,minz=minz,maxz=maxz,dz=dz,time=time)

# ----------------------
# fit sigmoids to track position of interface
z = np.arange(minz,maxz+1,dz)
# there's a characteristic value of the solid phase and the liquid phase 
# use the mean of these to specify a dicrete z point of the interface
# solid from z = 10 to 20
# liquid from z = 60 to 70
# these are somewhat arbitrary based on simbox
solid_q6 = np.mean(q6_z_T[1][np.where(z > 10)[0][0]:np.where(z < 20)[0][-1]])
liquid_q6 = np.mean(q6_z_T[1][np.where(z > 60)[0][0]:np.where(z < 70)[0][-1]])
crossover = 0.5*(solid_q6+liquid_q6)

# compute interface_z (t) function
interface_z = []
with open(args.input[:-5]+".interface_z_T.csv",'w') as f:
	f.write(f"t,z\n") # picoseconds, angstroms
	for i in range(1,40): # this should be over the first 20 picoseconds of sim time
		# filter out all indices where q6 = 0
		q6_unfiltered = q6_z_T[i]
		z_unfiltered = np.arange(minz,maxz+dz,dz)
		idx_in = np.where(q6_unfiltered > 0)[0]
		q6 = q6_unfiltered[idx_in]
		z = z_unfiltered[idx_in]
		# only use 5 < z < 75 to fit data
		idx1 = np.where(z > 5)[0][0]
		idx2 = np.where(z < 75)[0][-1]
		zdense = np.linspace(np.min(z[idx1:idx2]),np.max(z[idx1:idx2]),200)
		tanh_fit = fit_sigmoid_tanh(np.vstack((z[idx1:idx2],q6[idx1:idx2])))
		D_cryst, D_liq, z0, width = tanh_fit['params']
		q6_tanh = tanh_model(zdense,D_cryst, D_liq, z0, width)
		interface_idx = np.where(np.abs(q6_tanh-crossover) < 0.01)[0]
		if len(interface_idx) == 0:
			pass
		else:
			int_z = np.mean(zdense[interface_idx])
			interface_z.append(int_z)
			print(f"t={time[i]} => z={int_z}")
			f.write(f"{time[i]},{int_z}\n")

# read csv into a notebook to estimate velocity of interface

