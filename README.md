# Restricted Boltzmann Machine (RBM) for Collaborative Filtering 
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch)](https://pytorch.org/)

PyTorch implementation of a Restricted Boltzmann Machine for movie recommendations using the Movielens dataset.

## Features
- Gibbs sampling for latent feature discovery
- Contrastive Divergence (CD-k) training
- Binary feedback handling (likes/dislikes)
- Sparse user-item interaction support

## Requirements
- Python 3.8+
- PyTorch 2.2.2
- NumPy 1.26.4
- Pandas 2.2.1
- Jupyter Notebook

## Dataset Setup
1. Download datasets:
   - [Movielens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)
   - [Movielens 100K Dataset](https://grouplens.org/datasets/movielens/100k/)

2. Create folders and place files:
```text
data/
├── ml-1m/
│   ├── movies.dat
│   ├── ratings.dat
│   └── users.dat
└── ml-100k/
    ├── u.data
    └── u.item