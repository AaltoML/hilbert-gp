# Hilbert Space Methods for Reduced-Rank Gaussian Process Regression

[Arno Solin](http://arno.solin.fi) · [Simo Särkkä](https://users.aalto.fi/~ssarkka/)

Codes for the paper:

* Arno Solin and Simo Särkkä (2019). **Hilbert Space Methods for Reduced-Rank Gaussian Process Regression**. Accepted for publication in *Statistics and Computing*. [[preprint on arXiv](https://arxiv.org/abs/1401.5508)]

## Summary

This paper proposes a novel scheme for reduced-rank Gaussian process regression. The method is based on an approximate series expansion of the covariance function in terms of an eigenfunction expansion of the Laplace operator in a compact subset of R^d. On this approximate eigenbasis the eigenvalues of the covariance function can be expressed as simple functions of the spectral density of the Gaussian process, which allows the GP inference to be solved under a computational cost scaling as O(nm^2) (initial) and O(m^3) (hyperparameter learning) with m basis functions and n data points. Furthermore, the basis functions are independent of the parameters of the covariance function, which allows for very fast hyperparameter learning. The approach also allows for rigorous error analysis with Hilbert space theory, and we show that the approximation becomes exact when the size of the compact subset and the number of eigenfunctions go to infinity. We also show that the convergence rate of the truncation error is independent of the input dimensionality provided that the differentiability order of the covariance function is increases appropriately, and for the squared exponential covariance function it is always bounded by ∼1/m regardless of the input dimensionality. The expansion generalizes to Hilbert spaces with an inner product which is defined as an integral over a specified input density. The method is compared to previously proposed methods theoretically and through empirical tests with simulated and real data.

## Matlab implementation

We provide a proof-of-concept implementation written in Mathworks Matlab. The demo runs a simple 1D GP regression task on noisy observations of the sinc function.

## License

Copyright 2013-2019 Arno Solin and Simo Särkkä

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
