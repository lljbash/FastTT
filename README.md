# FastTT
This package performs a faster tensor train (TT) decomposition for large sparse data.
It can be several times to several tens time faster than the widely-used TT-SVD algorithm while keeping same accuracy. The speedup ratio depends on the sparsity of the data.

## Prerequisites

- [g++](https://gcc.gnu.org/) version >= 7 or latest Clang
- [Xerus](https://www.libxerus.org/)
- [Boost](https://www.boost.org/)
- [cxxopts](https://github.com/jarro2783/cxxopts)

If you used this package, please cite the following paper:

[1] Lingjie Li, Wenjian Yu, and Kim Batselier, "Faster tensor train decomposition for sparse data," arXiv#1908.02721
