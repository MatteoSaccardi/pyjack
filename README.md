# pyjack

`pyjack` is a lightweight Python package for handling observables with jackknife resampling, allowing error propagation and operations like arithmetic, powers, and universal functions.

## Features
- Create observables from raw data
- Automatic jackknife sample generation
- Arithmetic operations in python with proper error propagation
- Support for functions like `exp`, `log`, `sqrt`, etc.
- Plotting autocorrelation
- Perform fits, see [this paper](https://arxiv.org/pdf/2209.14188)

## Installation
```bash
pip install git+https://github.com/MatteoSaccardi/pyjack.git
```

## Upgrade
To upgrade to the latest version:
```bash
pip install --upgrade git+https://github.com/MatteoSaccardi/pyjack.git
```

## Current best install
```bash
pip install -U git+https://github.com/MatteoSaccardi/pyjack.git@main
```

## Authors

Copyright (C) 2025, Matteo Saccardi

## Usage
The syntax is inspired by the much more complete Python library [`pyobs`](https://github.com/mbruno46/pyobs).

Check out the [tests folder](https://github.com/MatteoSaccardi/pyjack/tree/main/tests) for more examples.

```python
import pyjack
import numpy

my_data = numpy.random.randn(100,10)

obs = pyjack.observable(description='My Observable')
obs.create(my_data)
print(obs)

# Arithmetic operations casted for observables

squared = obs ** 2
exponential = pyjack.exp(obs)

# Possibility to perform fits

numpy.random.seed(42)
data = numpy.random.randn(100,10)*0.3+4
obs = pyjack.observable()
obs.create(data)
fitfunc = 'params0+params1*x'
initial_guess = [3,0]
W = 'diag'
fit1 = pyjack.jackfit(fitfunc,W,initial_guess)
fit1.fit(numpy.arange(obs.data.shape[1]),obs,max_iter=1000,tol=1e-8,num_samples=10000)
# [jackfit.fit] Fit did converge: [LevenbergMarquardt.minimize] Convergence with tolerance 1e-08 reached after 2 iterations. Exiting successfully
# [jackfit.fit] chi2obs = 2.4941482973456255
# [jackfit.fit] chi2exp = 7.8693896829803505 +- 1.1128997617269722
# [jackfit.fit] p-value = 0.9614 +- 0.0019264929632660877
print(fit1.params)
# pyjack(4.001(17), 0.0009(33), description=Best parameters of fit)
fit1.plot()

```

## License
GPL-2.0 License
