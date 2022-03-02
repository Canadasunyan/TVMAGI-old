This repository provides the PyTorch implementation of MAGI-X, an extension of the MAnifold-constrained Gaussian process Inference (MAGI) for learning unknown ODEs system, where the associated derivative function does not have known parametric form.

MAGI-X codes and requirements
The codes for MAGI-X are provided under directory magix/. To successfully run the MAGI-X code, we require the installation of the following python packages. We provide the version that we use, but other version of the packages is also allowed as long as it is compatible.

pip3 install numpy==1.19.5 scipy==1.6.1 torch==1.8.0 matplotlib==3.3.4
Please see demo.ipynb for tutorial of running MAGI-X.

References
Our paper is available on arXiv. If you found this repository useful in your research, please consider citing
