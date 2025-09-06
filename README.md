# psiCI: a configuration-interaction module using Psi4 with DFT/HF orbitals

![Version](https://img.shields.io/badge/version-v00.01.004-blue)
[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![License](https://img.shields.io/badge/license-BSD-lightgray)](https://github.com/fmauger1/psiCI/blob/main/LICENSE)

Use `psiCI` to perform configuration-interaction (CI) calculations with Psi4 using density-functional theory (DFT) and Hartree-Fock (HF) orbitals. The module enable CI calculations using arbitrary configuration space for the CI expansion. It also provides support for building the configuration space associated with common active space (single- and multi-reference CIS(D) and CAS/RAS). `psiCI` is a Python class.

**Dependencies:** `psiCI` requires the `numpy`, `scipy`, `itertools`, and `psi4` packages.


## Content
**Root folder in main:**
* `psiCI.py` is the CI module itself
* `documentation.ipynb` is the documentation for the module, including a description of the methods, input parameters, and output as well as a few examples
* `test_PsiCI.ipynb` can be used to test the CI module (for future updates)

**Examples folder:**
* `Configuration_basis.ipynb` illustrates CI calculations with various sets of molecular orbitals (HF vs DFT)
* `H2O_molecule.ipynb` illustrates CI calculations in the H<sub>2</sub>O molecule (with a 3500 configuration state basis)
