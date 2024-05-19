This repo contains analysis and datafile generation scripts for LAMMPS molecular dynamics simulations.

Scripts currently available:
- topologyBuilder.py (builds bond topology for molecular mechanics e.g. CHARMM)
- densityProfile.py (1-d cartesian density profile)
- pressureProfiles.py (z-resolved pressure tensor elements, needs a dump file with compute stress/atom output)
- velocityProfile.py (calculates velocity profiles along some dimension binned along another dimension, needs a dump file with unwrapped coordinates)
- fitCircle.py (used with 2dDensity below, assuming some inhomogeneous structure where you want to fit a circle to some structure and estimate curvature / other properties)
- pairCorrelation.py (RDFs)
- vacf.py (velocity autocorrelation)
- neighborsCutoff.py (calculate n nearest neighbors within a given cutoff as the most basic order parameter)
- estimateFlux.py (estimate local charge flux / current as a vector field over a fixed set of nodes, this *may* generalize for any arbitrary quantity computed as a distribution on a fixed grid, need to come back to this)
 
