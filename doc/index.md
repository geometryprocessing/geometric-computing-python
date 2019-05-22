Geometry Computing with Python
=======

This course is based on efficient C++ libraries binded to python.
The main philosophy is to use `NumPy` arrays as a common interface, making them highly composable with each-other as well as existing scientific computing packages.

All libraries are part of **conda forge**. We advise to add `conda forge` to your conda channels:
```bash
conda config --add channels conda-forge
```

# Introduction
Many disciplines of computer science have access to high level libraries allowing researchers and engineers to quickly produce prototypes. For instance, in machine learning, one can construct complex, state-of-the-art models which run on the GPU in a few lines of Python.

In the field of geometric computing, however such high-level libraries are sparse. As a result, writing prototypes in geometry is time consuming and difficult even for advanced users.

In this course, we present a set of easy-to-use Python packages for applications in geometric computing. We have designed these libraries to have a shallow learning curve, while also enabling programmers to easily accomplish a wide variety of complex tasks. Furthermore, the libraries we present share NumPy arrays as a common interface, making them highly composable with each-other as well as existing scientific computing packages. Finally, our libraries are blazing fast, doing most of the heavy computations in C++ with a minimal constant-overhead interface to Python.

In the course, we will present a set of real-world examples from geometry processing, physical simulation, and geometric deep learning. Each example is prototypical of a common task in research or industry and is implemented in a few lines of code. By the end of the course, attendees will have exposure to a swiss-army-knife of simple, composable, and high-performance tools for geometric computing.

# Libigl

!!! warning
	Windows is currently unsupported, we expect to have it soon

**Libigl** is a simple python and C++ geometry processing library. We have a wide functionality including construction of sparse discrete differential geometry operators and finite-elements matrices such as the cotangent Laplacian and diagonalized mass matrix, and simple facet and edge-based topology data structures.

It can be easily install trough conda:
```bash
conda install igl
```

[Jupiter Notebook](https://github.com/geometryprocessing/libigl-python-bindings/blob/master/tutorial/tutorials.ipynb)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/geometryprocessing/libigl-python-bindings/master?filepath=tutorial%2Ftutorials.ipynb)


# Polyfempy

**Polyfem** is a polyvalent easy to use C++ finite element library.


It can be easily install trough conda:
```bash
conda install polyfempy
```


[Jupiter Notebook](https://github.com/polyfem/polyfem.github.io/blob/docs/docs/python_examples.ipynb)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/polyfem/polyfem.github.io.git/docs?filepath=docs%2Fpython_examples.ipynb).