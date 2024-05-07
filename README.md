# Gen-FVGN-steady
A fully differentiable GNN-based PDE Solver: With Applications to Poisson and Navier-Stokes Equations[Arxiv]
(This repository only displays the code for solving the steady-state PDE part. The code for the unsteady case examples will also be published soon.)

# Highlights
This code could solve multiple unstructured grids, multiple boundary conditions, multiple source terms, multiple PDEs in A single model without pre-computed data!

# Abstract
In this study, we present a novel computational framework that integrates the finite volume method with graph neural networks to address the challenges in Physics-Informed Neural Networks(PINNs). Our approach leverages the flexibility of graph neural networks to adapt to various types of two-dimensional unstructured grids, enhancing the model's applicability across different physical equations and boundary conditions. The core innovation lies in the development of an unsupervised training algorithm that utilizes GPU parallel computing to implement a fully differentiable finite volume method discretization process. This method includes differentiable integral and gradient reconstruction algorithms, enabling the model to directly solve partial-differential equations(PDEs) during training without the need for pre-computed data. Our results demonstrate the model's superior mesh generalization and its capability to handle multiple boundary conditions simultaneously, significantly boosting its generalization capabilities. The proposed method not only shows potential for extensive applications in CFD but also establishes a new paradigm for integrating traditional numerical methods with deep learning technologies, offering a robust platform for solving complex physical problems.

# Usage Method
The specific usage instructions for this repository are currently being prepared. All algorithms related to FVM are contained within the Integrator class in the model.py file.

# License
Feel free to clone this repository and modify it! If it's of good use for you, give it a star and please cite our publications!