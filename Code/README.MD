# Code guidelines

This is implementation of our methods on Python (3.6.9). 

It enables to run simulated distributed optimization with master node on any number of workers. Here you can find implementations of NL1, NL2, CNL, DINGO, BFGS, and gradient type methods (DCGD, DIANA, ADIANA).

### Installation

To install requirements
```sh
$ pip install -r Requirements.txt
```

###  Example Notebooks
To run Newton type methods (NEWTON-LEARN and DINGO) see [example notebook](https://github.com/Intelligent-Systems-Phystech/Islamov-BS-Thesis/blob/main/Code/NEWTON-LEARN/NL_DINGO_example_notebook.ipynb)     
To run BFGS see [example notebook](https://github.com/Intelligent-Systems-Phystech/Islamov-BS-Thesis/blob/main/Code/BFGS/BFGS_example_notebook.ipynb)      
To run gradient type methods see [example notebook](https://github.com/Intelligent-Systems-Phystech/Islamov-BS-Thesis/blob/main/Code/Gradient-type-methods/gradient_type_methods_example_notebook.ipynb)

If you want to learn implementation in more details, please read brief descriptions of methods and oracles.

***Remark:*** We used implementation of BFGS which was written by other people.

Here you can find implementations of the following methods:
- Newton's method
- NEWTON-STAR (this paper)
- NEWTON-LEARN 1 (this paper)
- NEWTON-LEARN 2 (this paper)
- CUBIC-NEWTON-LEARN (this paper)
- [DINGO](https://arxiv.org/abs/1901.05134)
- [BFGS](https://www.ams.org/journals/mcom/1967-21-099/S0025-5718-1967-0224273-2/S0025-5718-1967-0224273-2.pdf)
- [ADIANA](https://arxiv.org/abs/2002.11364)
- [DIANA](https://arxiv.org/abs/1901.09269)
- [DCGD](https://arxiv.org/abs/1806.06573)

Please read original papers in order to get a closer look at state-of-the-art methods in distributed optimization like ADIANA or DIANA.

### License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
