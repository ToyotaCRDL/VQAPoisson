# Variational quantum algorithm based on the minimimum potential energy for solving the Poisson equation

[![python](https://img.shields.io/badge/python-v3.7.4-blue)](https://www.python.org/downloads/release/python-374/)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://opensource.org/licenses/Apache-2.0)

This codes solves the one-dimensional Poisson equation based on the variational quantum algorithm.

## Requirement

|  Software  |  Version  |
| :----: | :----: |
|  python  |  3.7.4  |
|  qiskit  |  0.23.6  |
| qiskit-aer | 0.7.5 |
| qiskit-aqua | 0.8.2 |
| numpy | 1.19.1 |
| scipy | 1.6.1 |

To run jupyter notebook,
|  Software  |  Version  |
| :----: | :----: |
| matplotlib | 3.4.2 |
| tqdm | 4.60.0 |
 
## Usage
 
See [sample.ipynb](/sample.ipynb) as a sample code.

```bash
from vqa_poisson import VQAforPoisson

num_qubits = ... # int
num_layers = ... # int
bc = ... # str
oracle_f = ... # qiskit.QuantumCircuit
qins = ... # qiskit.aqua.QuantumInstance
vqa = VQAforPoisson(num_qubits, num_layers, bc, oracle_f=oracle_f, qinstance=qins)
x0 = ... # numpy.ndarray
res = vqa.minimize(x0)
```
  
## Citing the library

If you find it useful to use this module in your research, please cite the following paper.
```
Yuki Sato, Ruho Kondo, Satoshi Koide, Hideki Takamatsu, and Nobuyuki Imoto, Variational quantum algorithm based on the minimum potential energy for solving the Poisson equation, Physical Review A, 104: 052409, 2021.
```

In bibtex format:
```
@article{sato2021vqa,
  author  = {Sato, Yuki and Kondo, Ruho and Koide, Satoshi and Takamatsu, Hideki and Imoto, Nobuyuki},
  title   = {Variational quantum algorithm based on the minimum potential energy for solving the Poisson equation},
  journal = {Physical Review A},
  year    = {2021},
  volume  = {104},
  issue   = {5},
  pages   = {052409},
}
```
 
# License

This project is licensed under the Apache License Version 2.0 - see the [LICENSE.txt](/LICENSE.txt) file for details
