# Overview

CMPS 351 Project: Discrete Optimization — Vertex Cover (LP Relaxation & Integrality Gap)

This project studies the **Minimum Vertex Cover** problem using optimization and approximation techniques.

Given an undirected graph \( G = (V, E) \), the goal is to select the smallest set of vertices such that every edge has at least one of its endpoints in the set.

Since this problem is NP-hard, we explore efficient approaches based on:

- Integer Programming (exact solution)
- Linear Programming (relaxation)
- Rounding algorithms (approximation)
- Greedy heuristics (baseline)

---

## Graph Families

We evaluate performance across different graph structures:

- Random graphs \( G(n, p) \)
- Grid graphs
- Bipartite graphs
- Cycle graphs (including odd cycles)
- Dense graphs / near-cliques
- Complete graphs

---

## Evaluation Metrics

We assess performance using:

- **Solution size**
- **Approximation ratio**
- **Integrality gap**
- **Runtime**

---

## How to run (Reproducibility)

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd vertex-cover-project
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

On Windows:
```bash
.venv\Scripts\activate
```

On macOS/Linux:
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run experiments

```bash
python src/experiments.py
```

## Authors
- Ali Mohsen: amm117@mail.aub.edu
- Nanar Aintablian: nsa81@mail.aub.edu
- Julia Dirawi: jnd06@mail.aub.edu