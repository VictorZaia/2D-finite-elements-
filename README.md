# 2D finite elements-

This repository contains a Python implementation of a 2D finite element solver using isoparametric plane stress elements. The code is designed to solve linear elasticity problems in which the stress and strain tensors are related by Hooke's law.

The solver uses a quad mesh to discretize the domain, and implements the isoparametric mapping technique to efficiently map the reference element to the physical element. The code uses Gaussian quadrature to perform numerical integration of the element stiffness matrix and load vector.

The solver is capable of handling arbitrary geometries and material properties. The code includes functions for visualizing the displacement within the domain, and for computing other derived quantities of interest, such as strain and stress.

Contributions to the code are welcome, and suggestions for improvements or additional features are encouraged.
