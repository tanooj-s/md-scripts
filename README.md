This repo contains analysis and datafile generation scripts for LAMMPS molecular dynamics simulations.

Scripts currently available:
- topologyBuilder.py (builds bond topology for molecular mechanics e.g. CHARMM)
- densityProfile.py (calculates density profiles along a specific dimension from a dump file)
- densityProfileIO.py (same as above, but does analysis as dump file is being read, necessary for large systems)
- pressureProfiles.py (calculate z-resolved pressure tensor elements, needs a dump file with compute stress/atom output)
- velocityProfile.py (calculates velocity profiles along some dimension binned along another dimension, needs a dump file with unwrapped coordinates)
- wettingCircles.py (calculates wetting angle from a dump file)
- pairCorrelation.py (calculates radial pair distribution functions for given pairs of atom types, needs a dump file)
- velocityAutocorrelation.py (calculates velocity autocorrelations for specific atom types, needs a dump file with granular output)
- neighborsCutoff.py (calculates n nearest neighbors within a given cutoff as a basic order parameter)
- estimateFlux.py (estimate local charge flux / current as a vector field over a fixed set of nodes, this should generalize for any arbitrary quantity computed as a distribution on a fixed grid)

Shortly available:
- template scripts for different sorts of alchemical simulations
 
