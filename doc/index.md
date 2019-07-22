Geometric Computing with Python
=======

This course is based on efficient C++ libraries binded to python.
The main philosophy is to use `NumPy` arrays as a common interface, making them highly composable with each-other as well as existing scientific computing packages.

All libraries are part of [**conda forge**](https://conda-forge.org/). We advise to add `conda forge` to your conda channels:
```bash
conda config --add channels conda-forge
```


Then just install them
```bash
conda install mehsplot
conda install igl
conda install wildmeshing
conda install polyfempy
```


Here you can find more details and examples of the libraries:

- [3D Viewer](meshplot)
- [Wildmeshing](wildmeshing)
- [igl](igl.md)
- [polyfem](polyfem.md)
- ABC dataset CAD Processing (coming soon)


And here the final notebooks and slides used in our presentation:

- Introduction
- Visualization, CAD Processing, and Machine Learning
- [Meshing and Simulation](Polyfem-2d)
- All together
- Closing remarks


# Motivation
Many disciplines of computer science have access to high level libraries allowing researchers and engineers to quickly produce prototypes. For instance, in machine learning, one can construct complex, state-of-the-art models which run on the GPU in a few lines of Python.

In the field of geometric computing, however such high-level libraries are sparse. As a result, writing prototypes in geometry is time consuming and difficult even for advanced users.

In this course, we present a set of easy-to-use Python packages for applications in geometric computing. We have designed these libraries to have a shallow learning curve, while also enabling programmers to easily accomplish a wide variety of complex tasks. Furthermore, the libraries we present share NumPy arrays as a common interface, making them highly composable with each-other as well as existing scientific computing packages. Finally, our libraries are blazing fast, doing most of the heavy computations in C++ with a minimal constant-overhead interface to Python.

In the course, we will present a set of real-world examples from geometry processing, physical simulation, and geometric deep learning. Each example is prototypical of a common task in research or industry and is implemented in a few lines of code. By the end of the course, attendees will have exposure to a swiss-army-knife of simple, composable, and high-performance tools for geometric computing.


# Contact
This course is a group endeavor by Sebastian Koch, Teseo Schneider, Francis Williams, ChengChen Li, and Daniele Panozzo. Please contact us if you have questions or comments. For troubleshooting, please post an issue on github. We are grateful to the authors of all open source C++ libraries we are using. In particular, libigl, tetwild, polyfem, pybind11, and Jupyter.