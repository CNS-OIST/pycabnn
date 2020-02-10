# Pycabnn

**Pycabnn** (**Py**thon tool for **C**onstructing an **A**natomical **B**asis of a **N**eural **N**etwork; pronounce it as "pie cabin") is a Python 3 package to aid building a physiologically realistic neural network model. Pycabnn is for setting up a structural basis of a model, such as placing neurons in space and determine their connectivity, based on anatomical constraints.  It is primarily developed for a model of the granular layer in the cerebellar cortex [1]. However, we tried to make it as adaptable as possible to other circuit models.

For a detailed explanation about algorithms used/implemented, please check out our paper [2].



## Getting Started

### Prerequisites

Pycabnn is written in pure Python 3 and depends on the following packages:

* numpy
* scipy
* scikit-learn
* pytables (for saving results in HDF5 files)
* ipyparallel (for utilizing multiple CPUs)
* cloudpickle (when used with ipyparallel)

You will also need to install the followings to run example scripts:

* matplotlib (for plotting)
* NEURON (for reading a parameter file)

### Installing

We haven't written proper setup.py for installation yet. For usage, please check out example scripts:
#### 1. Population generation




#### 2. Connectivity generation
Run `run_connector.py` as:
```bash
python run_connector.py -i INPUT_PATH -o OUTPUT_PATH -p PARAM_PATH all
```

We include some test data in the `test_data` directory: Use `./test_data/cell_position` for `INPUT_PATH` and `./test_data/params` for `PARAM_PATH`. `OUTPUT_PATH` can be anywhere. This script will generate the connectivity data as tables in HDF5 and text files.



## Authors

* [**Ines Wichert**](https://github.com/inesw) - *Connectivity*

* [**Sanghun Jee**](https://github.com/Alexji9494) - *Cell population*

* [**Sungho Hong**](http://shhong.github.io) - *Enjoyed working with two great interns!*

  

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details



## Acknowledgments

* Erik De Schutter
* Ivan Raikov
* Peter Bratby
