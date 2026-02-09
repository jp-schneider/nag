# Installation

[Back to README](../README.md) | [Next (Datasets)](./datasets.md)

This document provides instructions for installing the Neural Atlas Graphs (NAG) repository and its dependencies.

## Prerequisites

Before installing NAG, ensure that you have the following prerequisites:
- Python 3.9 or higher (e.g. via pyenv)
- CUDA-compatible GPU with the appropriate drivers installed
- CUDA toolkit installed (version 12.4 recommended)


Further we highly recommend using poetry for managing the python environment and dependencies, yet you can also use pip, virtualenv, conda or any other tool if preferred.
To ease out the process, we assume you have poetry installed on your system. You can find installation instructions for poetry at https://python-poetry.org/docs/#installation.

The code has been tested with cuda version 12.4, yet newer versions should also work.


## Installation Steps

### Clone the Repository

First, clone the NAG repository from GitHub:

```bash
git clone --recurse-submodules https://github.com/jp-schneider/nag.git 
```
Which includes also our tools submodule.

Navigate to the cloned directory:

```bash
cd nag
```

### Initialize the Python Environment

Create and activate a new poetry environment:

```bash
poetry shell
```

In case this fails given version constraints, you might need to create the environment by specifying a python version:

```bash
poetry env use [path/to/python3.9/or/higher]
```

### Install Python Dependencies


#### TL;DR
To install with all recommended dependencies, including those for depth processing and assuming you are using CUDA 12.4, run:

```bash
poetry install --with dev --extras depth --extras torch --extras post-torch
```

Then you can skip the detailed instructions below and continue to the tinycudann installation section.

#### Detailed Instructions
All our runtime and development dependencies are specified within the `pyproject.toml` file. To install them, run:

```bash
poetry install
```

This command will install the required packages, except pytorch, tinycudann and packages depending on these.
If you are using a GPU with CUDA support **and** the **recommended CUDA version 12.4**, you can directly install pytorch using:

```bash
poetry install --extras torch
```

> If you **use a different CUDA version**, you need to update the source entry in the `pyproject.toml` file, by updating the index url provided for the torch, torchvision and torchaudio packages to match your cuda version. You can find the appropriate index urls at https://pytorch.org/get-started/locally/. After updating the `pyproject.toml` file, run the above command to install pytorch and its related packages.


Further install the torch-dependent packages using:

```bash
poetry install --extras post-torch
```

### Build and Install Tinycudann

For our implementation, we rely on [tinycudann](https://github.com/NVlabs/tiny-cuda-nn), a small and efficient library for neural networks on the GPU. Yet, in there default implementation, the default network dtype is float16, which caused us some numerical issues during training. Therefore, we recommend using our modified version of tinycudann, which uses float32 as default dtype.

To build and install tinycudann, first clone our modified repository, ideally outside of the nag repository:

```bash
cd ..
git clone --recursive https://github.com/jp-schneider/tiny-cuda-nn-float32.git  
```

Navigate to the cloned directory and build the package, for details we refer to the original tinycudann [readme](https://github.com/jp-schneider/tiny-cuda-nn-float32/blob/master/README.md). Make sure to build it locally, with the used CUDA toolkit on the PATH and properly set LD_LIBRARY_PATH. If you have multiple CUDA versions installed, make sure to use the one matching your pytorch installation and the one you intend to use with NAG.
The usual commands are:

```bash
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j
```
but check the [repository](https://github.com/jp-schneider/tiny-cuda-nn-float32/blob/master/README.md) since they might require further adjustments based on your system.

After you build tinycudann successfully, you can install the pytorch bindings to the current poetry environment using the following command. Make sure the poetry environment is activated when running the command:

```bash
cd [path/to/tiny-cuda-nn-float32]/bindings/torch
python setup.py install
```

### Install Development Dependencies (Optional)

If you plan to contribute to the development of NAG, run Notebooks or want to run the development tools (e.g., linters, formatters, type checkers), you can install the development dependencies by running:

```bash
poetry install --with dev
```


> **Thats it!** You have successfully installed the Neural Atlas Graphs (NAG) repository and its dependencies. You can now proceed to set up the datasets as described in the [Datasets](./datasets.md) document.
