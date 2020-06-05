# Pycabnn

**Pycabnn** (**Py**thon tool for **C**onstructing an **A**natomical **B**asis of a **N**eural **N**etwork; pronounce it as "pie cabin") is a Python 3 package to aid building a physiologically realistic neural network model. Pycabnn is for setting up a structural basis of a model, such as placing neurons in space and determine their connectivity, based on anatomical constraints. It is primarily developed for a model of the granular layer in the cerebellar cortex [1]. However, we tried to make it as adaptable as possible to other circuit models.

For a detailed explanation about algorithms used/implemented, please check out our paper [2].

## Getting Started

### Prerequisites

Pycabnn is written in pure Python 3 and depends on the following packages:

- numpy
- scikit-learn
- joblib
- tqdm (for progress bars)
- pandas and pytables (for saving results in HDF5 files)
- ipyparallel and cloudpickle (for utilizing multiple CPUs)

You will also need to install the followings to run example scripts:

- matplotlib (for plotting)
- NEURON (for reading a parameter file)
- Jupyter notebook (for reading notebooks)

### Installation

We do not have proper setup.py for installation yet. For usage, please check out example scripts:

#### 1. Cell position generation

Run `generate_cell_position.py` as:

```shell
python generate_cell_position.py -p PARAM_PATH -o OUTPUT_PATH all
```

We included some parameter data in `test_data/params` for `PARAM_PATH`. The results will be save in a [".npz" file](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html), `cell_positions.npz`, within `OUTPUT_PATH`, along with text files.

#### 2. Connectivity generation

Run `run_connector.py` as:

```shell
python run_connector.py -i INPUT_PATH -o OUTPUT_PATH -p PARAM_PATH all
```

We included some test data in the `test_data` directory: Use `test_data/cell_position` for `INPUT_PATH` and `test_data/params` for `PARAM_PATH`. `OUTPUT_PATH` can be anywhere. This script will generate the connectivity data as tables in HDF5 and text files.

## Authors

- [**Ines Wichert**](https://github.com/inesw) - _Connectivity generation_

- [**Sanghun Jee**](https://github.com/Alexji9494) - _Cell position generation_

- [**Sungho Hong**](http://shhong.github.io) - _Project conception and supervision_

- [**Erik De Schutter**](https://groups.oist.jp/cnu) - _Project supervision_

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details

## Acknowledgments

- Ivan Raikov
- Peter Bratby

## References

1. Sudhakar, S.K., Hong, S., Raikov, I., Publio, R., Lang, C., Close, T., Guo, D., Negrello, M., and De Schutter, E. (2017). Spatiotemporal network coding of physiological mossy fiber inputs by the cerebellar granular layer. PLoS Comput. Biol. _13_, e1005754.
2. Wichert I., Jee S., De Schutter, E., and Hong S. (2020) Pycabnn: Efficient and extensible software to construct an anatomical basis for a physiologically realistic neural network model. _In preparation_.

---

Written by Sungho Hong, Computational Neuroscience Unit, Okinawa Institute of Science Technology

March, 2020
