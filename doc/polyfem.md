Polyfempy
=========

[![Anaconda-Server Badge](https://anaconda.org/conda-forge/polyfempy/badges/downloads.svg)](https://anaconda.org/conda-forge/polyfempy)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/polyfempy/badges/installer/conda.svg)](https://conda.anaconda.org/conda-forge)

PolyFEM is a simple C++ and Python finite element library. We provide a wide set of common PDEs including:

 - Laplace
 - Helmholtz
 - Linear Elasticity
 - Saint-Venant Elasticity
 - Neo-Hookean Elasticity
 - Stokes

PolyFEM simplicity lies on the interface: just pick a problem, select some boundary condition, and solve. No need to construct complicated function spaces, or learn a new scripting language: everything is set-up trough a [JSON interface](documentation.md) or trough the [Setting class](polyfempy_doc.md) in python.


For instance, PolyFEM seamlessly integrates quad/hexes and tri/tets of order up to 4, and integrate state-of-the-art techniques such as the adaptive $p$-refinement presented in "Decoupling Simulation Accuracy from Mesh Quality" or the spline and polygonal bases in "Poly-Spline Finite-Element Method".

The library is actively used in our research so expect frequent updates, fixes, and new features!

For more information visit [https://polyfem.github.io/](https://polyfem.github.io/) or the [Python section](https://polyfem.github.io/python/).


It can be easily install trough conda:
```bash
conda install polyfempy
```


[Jupyter Notebook](https://github.com/polyfem/polyfem.github.io/blob/docs/docs/python_examples.ipynb)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/polyfem/polyfem.github.io.git/docs?filepath=docs%2Fpython_examples.ipynb).