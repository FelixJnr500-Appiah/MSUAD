# MSUAD: Multi-Stage Uncertainty-Aware Distillation

**A Framework for High-Dimensional Causal Inference & Model Calibration**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](#)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](#)
[![Research Status: Under Review](https://img.shields.io/badge/Research-Under%20Review-yellow)](#)

## 📌 Overview
**MSUAD** (Multi-Stage Uncertainty-Aware Distillation) is a research engineering framework designed for **Causal Inference in high-dimensional feature spaces ($p > 10,000$)**. 

The framework utilizes a **Teacher-Student Distillation** architecture to improve model calibration and treatment effect estimation. This approach integrates Deep Gaussian Processes (Teachers) with production-efficient Neural Networks (Students) through an uncertainty-aware loss function.

## 🚀 Key Features
* **Uncertainty-Aware Loss:** Custom distillation loss utilizing KL-Divergence and Variance Regularization.
* **GPyTorch Integration:** Leverages GPU-accelerated Variational Inference for scalable Gaussian Processes.
* **HPC Optimized:** Designed for Slurm-based clusters (RMACC Alpine) with parallelized Monte Carlo cross-validation.
* **Robust Estimation:** Built on Double Machine Learning (DML) principles to mitigate high-dimensional confounding bias.

## 🛠 Tech Stack
Core: Python, PyTorch, GPyTorch
Statistical ML: DoubleML, Scikit-Learn
Infrastructure: Slurm, Linux/Bash, Git

## 📑 Confidentiality Note
The full source code for the MSUAD framework is currently under peer review as part of a Doctoral Dissertation at the University of Northern Colorado. Publicly available files demonstrate architectural patterns and API design.

## 📂 Project Structure
*Note: Core mathematical logic and proprietary kernels are omitted pending peer review.*

```text
├── src/
│   ├── models/             # PyTorch Architectures
│   ├── distillation/       # Custom training loops & Loss functions
│   └── estimation/         # DML & Orthogonalization logic
├── tests/                  # Statistical consistency checks
├── configs/                # Hydra/YAML experiment configs
└── README.md
