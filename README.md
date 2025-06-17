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

squared = obs ** 2
exponential = pyjobs.exp(obs)
```

## License
MIT License
