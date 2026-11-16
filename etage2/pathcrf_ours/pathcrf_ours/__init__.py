"""pathcrf_ours: a clean-room re-implementation of PathCRF (Kim et al., KDD 2026).

The model reads player positions only and infers the ball-possession path: at every time step
it picks one edge on a graph of 22 players plus 4 "outside" nodes, and a masked conditional
random field (CRF) forbids impossible sequences. See README.md for the full spec and results.
"""
