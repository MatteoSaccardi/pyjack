# pyjobs

`pyjobs` is a lightweight Python package for handling observables with jackknife resampling, allowing error propagation and operations like arithmetic, powers, and universal functions.

## Features
- Create observables from raw data
- Automatic jackknife sample generation
- Arithmetic operations with proper error propagation
- Support for functions like `exp`, `log`, `sqrt`, etc.
- Plotting autocorrelation

## Installation
```bash
pip install git+https://github.com/MatteoSaccardi/pyjobs.git
```

## Upgrade
To upgrade to the latest version:
```bash
pip install --upgrade git+https://github.com/MatteoSaccardi/pyjobs.git
```

## Usage
```python
import pyjobs
import numpy

my_data = numpy.random.randn(100,10)

obs = pyjobs.observable(description='My Observable')
obs.create(my_data)
print(obs)

# Arithmetic operations casted for observables

squared = obs ** 2
exponential = pyjobs.exp(obs)

# Possibility to perform fits

import numpy, matplotlib.pyplot as plt, pyjobs
data = numpy.random.randn(100,10)*0.3+4
obs = pyjobs.observable()
obs.create(data)
fitfunc = 'params0+params1*x'
initial_guess = [3,0]
W = 'diag'
fit1 = pyjobs.jackfit(fitfunc,W,initial_guess)
fit1.fit(numpy.arange(obs.data.shape[1]),obs,max_iter=1000,tol=1e-8,num_samples=10000)
>>> [jackfit.fit] Fit did converge: [LevenbergMarquardt.minimize] Convergence with tolerance 1e-08 reached after 10 iterations. Exiting successfully
>>> [jackfit.fit] chi2obs = 6.198204612587487
>>> [jackfit.fit] chi2exp = 7.880612900146689 +- 1.1144869643199817
>>> [jackfit.fit] p-value = 0.6162 +- 0.004863344652820856
print(fit1.params)
>>> pyjobs(mean=[3.98990459e+00 3.43646323e-04], std=None, description=Best parameters of fit)
fit1.plot()

```

## License
MIT License
