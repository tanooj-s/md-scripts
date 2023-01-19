This repo contains analysis and datafile generation scripts for LAMMPS molecular dynamics simulations.

Scripts currently available:
- topologyBuilder.py (builds bond topology for use with something like a CHARMM force field)
- densityProfile.py (calculates density profiles along a specific dimension from a dump file)
- velocityProfile.py (calculates velocity profiles along some dimension binned along another dimension, needs a dump file with unwrapped coordinates)
- wettingCircles.py (calculates wetting angle from a dump file)
- pairCorrelation.py (calculates radial pair distribution functions for given pairs of atom types, needs a dump file)

Shortly available:
- velocityAutocorrelation.py (calculates velocity autocorrelations, needs a dump file with granular output)

 
