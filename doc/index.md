Geometric Computing with Python
=======

This course is based on efficient C++ libraries binded to python.
The main philosophy is to use `NumPy` arrays as a common interface, making them highly composable with each-other as well as existing scientific computing packages.

## Installation

The easiest way to install the libraries is trough the [conda](https://anaconda.org/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) python package manager.

All libraries are part of the channel [conda forge](https://conda-forge.org/), which we advise to add to your conda channels by:
```bash
conda config --add channels conda-forge
```
This step allows to install any conda forge package simply with `conda install <package>`.

To install all our packages just run:
```bash
conda install meshplot
conda install igl
conda install wildmeshing
conda install polyfempy
```

**Note 1**: that you can install only the ones you need.

**Note 2**: in case of problem we advise to create a new conda environment `conda create -n <name>`.

**Note 3**: if problem persist or your want you feature please post issues on the github bugtracker of each library or [here](https://github.com/geometryprocessing/geometric-computing-python/issues).

### Packages Description

The four packages have specific functionalities and own website.

- [Meshplot](https://skoch9.github.io/meshplot/): fast 2d and 3d mesh viewer based on `pythreejs`.
- [Wildmeshing](https://wildmeshing.github.io/): robust 2d and 3d meshing package ([python documentation](https://wildmeshing.github.io/wildmeshing-notebook/))
- [igl](https://libigl.github.io/): swiss-army-knife of geometric processing functions ([python documentation](https://geometryprocessing.github.io/libigl-python-bindings/))
- [polyfempy](https://polyfem.github.io/): simple but powerful finite element library ([python documentation](https://polyfem.github.io/python/))

### Additional Useful Dataset
- [ABC CAD dataset](https://deep-geometry.github.io/abc-dataset/): 1 million meshed CAD models with feature file


## Course Material

Most of the course material consist of [Jupyter Notebook](https://jupyter.org) which can be easily installed trough conda:
```bash
conda install jupyter
```

For completeness we include the *html rendered* notebook in this website and *interactive and editable* binder version.

The course is divided in five parts:

- [Introduction](Intro.pdf)
- [Geometry Processing and Visualization](viz_basic) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/geometryprocessing/geometric-computing-python/doc?filepath=doc%2Fviz_basic.ipynb)
<!-- - [CAD Processing and Machine Learning](cad_ml) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/geometryprocessing/geometric-computing-python/doc?filepath=doc%2Fcad_ml.ipynb) -->
- [Meshing and Simulation](Polyfem-2d) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/geometryprocessing/geometric-computing-python/doc?filepath=doc%2FPolyfem-2d.ipynb)
- [Ultimate Example](All) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/geometryprocessing/geometric-computing-python/doc?filepath=doc%2FAll.ipynb)
- [Closing Remarks](concluding.pdf)





## Motivation
Many disciplines of computer science have access to high level libraries allowing researchers and engineers to quickly produce prototypes. For instance, in machine learning, one can construct complex, state-of-the-art models which run on the GPU in a few lines of Python.

In the field of geometric computing, however such high-level libraries are sparse. As a result, writing prototypes in geometry is time consuming and difficult even for advanced users.

In this course, we present a set of easy-to-use Python packages for applications in geometric computing. We have designed these libraries to have a shallow learning curve, while also enabling programmers to easily accomplish a wide variety of complex tasks. Furthermore, the libraries we present share NumPy arrays as a common interface, making them highly composable with each-other as well as existing scientific computing packages. Finally, our libraries are blazing fast, doing most of the heavy computations in C++ with a minimal constant-overhead interface to Python.

In the course, we will present a set of real-world examples from geometry processing, physical simulation, and geometric deep learning. Each example is prototypical of a common task in research or industry and is implemented in a few lines of code. By the end of the course, attendees will have exposure to a swiss-army-knife of simple, composable, and high-performance tools for geometric computing.


## Contact
This course is a group endeavor by Sebastian Koch, Teseo Schneider, Francis Williams, Chengchen Li, and Daniele Panozzo. Please contact us if you have questions or comments. For troubleshooting, please post an issue on github. We are grateful to the authors of all open source C++ libraries we are using. In particular, libigl, tetwild, polyfem, pybind11, and Jupyter.
