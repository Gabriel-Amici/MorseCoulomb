MorseCoulomb is a repo containing a set of scripts and jupyter notebooks used to explore the dynamics of different one-dimensional potentials.

## About

The scripts and notebooks make use of the ```emerald``` library, containing the implmentation of a few different potentials:
- 1D Coulomb potential (*∝ -1/r*);
- 1D soft-Coulomb (sC) potential: contains a parameter *α* controlling well depth at the origin and is Coulomb-like in the long range;
- 1D Morse-soft-Coulomb (MsC) potential: soft-Coulomb potential for positive *r* smoothly joint to Morse barrier at the origin. The *α* parameter controls well depth and barrier steepnes;
- 1D normalised Morse-soft-Coulomb (nMsC) potential: MsC multiplied by *α*, providing unitary well depth for all *α*.

This repo is the result of 2+ years of research supported by the Brazilian agency São Paulo Research Foundation, FAPESP, grant 23/00690-9. Results concerning the numerical study of chaos in the MsC potential and applications of classical Optimal Control Theory were published by [Physical Review E](https://doi.org/10.1103/hbr7-ctsn), a preliminary version of the paper is found in [ArXiv](https://arxiv.org/abs/2411.06199).