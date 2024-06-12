import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ideal gas term definitions
mF = 18.9984*(1.660539e-27) # AMU -> kg
mLi = 6.941*(1.660539e-27)
mNa = 22.9898*(1.660539e-27)
mK = 39.0983*(1.660539e-27)
kB = 1.380649e-23 # J/K
h = 6.626070e-34 # J.s
N = 10000

def thermalLambda(m,T):
    '''
    return thermal wavelength of a particle
    '''
    return h/np.power(2*np.pi*m*kB*T,1.5)

def idealGasGibbs(n,m,V,T):
    '''
    return analytical Gibbs energy of a mix of ideal gases
    here n and m are a list of numbers and masses of each of the ideal has species
    V is the volume of the enclosing box
    there is a net NkTlnV term for the entire system
    and a n_ikTln(lambda_i) n_ikTln(n_i) term for each species
    '''
    assert len(n) == len(m)
    G = 0
    N = np.sum(n)
    G += -N*kB*T*np.log(V)
    for i in range(len(n)):
        n_i = n[i]
        m_i = m[i]
        lambda_i = thermalLambda(m_i,T)
        G += n_i*kB*T*np.log(n_i)
        G += n_i*kB*T*np.log(lambda_i)
    G *= (6.241509e18/N) # joules -> eV, per particle
    return G

T = 600 # edit as appropriate
steps = ['bmh2lj','lj2ideal']
lambdas = np.arange(0.0,1.01,0.1)
pathway_dG = 0

plt.rcParams['figure.figsize'] = (8,8)
fig, axes = plt.subplots(2,1)

for i in range(len(steps)):
    step = steps[i]
    print(f"step {step}")
    tdfs = []
    for l in lambdas:
        l = round(l,1)
        df = pd.read_csv(f"T-{T}.lam-{l}.{step}.log.csv")
        nUse = int(0.5*df.shape[0])
        N = df['Atoms'][0] # assuming thermo_modify norm no so you need to normalize per atom here
        if step == 'lj2ideal':
            tdf = -1*np.mean(df['PotEng'][-nUse:])/N * 0.0043 # kcal/mol -> eV
        else:
            tdf = np.mean(df['v_tdf'][-nUse:]) * 0.0043 # kcal/mol -> eV
        tdfs.append(tdf)
    tdfs = np.array(tdfs)
    df0 = pd.read_csv(f"T-{T}.lam-0.0.{step}.log.csv")
    df1 = pd.read_csv(f"T-{T}.lam-1.0.{step}.log.csv")
    P1 = np.mean(df1['Press'][-nUse:])
    P0 = np.mean(df0['Press'][-nUse:])
    print(f"P0: {P0} | P1: {P1}")
    N = df1['Atoms'][0]
    V = df1['Volume'][0]
    VdP = V*(P1-P0) * (1e-30) * (1.01325e5) * (6.241509e18) # [atm][A^3] -> eV
    VdP /= N
    print(f"VdP: {VdP} eV")
    dA = np.trapz(x=lambdas,y=tdfs)
    print(f"dA: {dA} eV")
    dG = dA + VdP
    print(f"dG: {dG} eV/atom")
    pathway_dG += dG
    axes[i].scatter(x=lambdas,y=tdfs,label=f"{step}, dA={round(dA,6)} eV")
    axes[i].grid()
    axes[i].legend(loc="lower right")
    axes[i].set_ylabel("dU/dlambda (eV/atom)")
axes[1].set_xlabel("Lambda")

# ideal gas term at FLiNaK composition
nF = 0.5*N
nLi = 0.465*0.5*N
nNa = 0.115*0.5*N
nK = 0.42*0.5*N
Gideal = idealGasGibbs([nF,nLi,nNa,nK],[mF,mLi,mNa,mK],V,T)
print(f"ideal gas term: {Gideal} eV/atom")
Gliquid = Gideal - pathway_dG
print(f"Absolute gibbs free energy / chemical potential of liquid at this state point: {Gliquid} eV")

plt.tight_layout()
plt.savefig("pathway_energies.png")
