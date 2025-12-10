import numpy
import pyjack

def _apply_ufunc(func, x, *args, **kwargs):
    if x.creator == 'create_from_cov':
        x_sampled = x.sample(1000)
        new_x = pyjack.observable(description=x.description, label=x.label)
        new_x.create(x_sampled)
        return new_x._new(func(new_x.jack_samples, *args, **kwargs))
    return x._new(func(x.jack_samples, *args, **kwargs))

def sqrt(x):    return _apply_ufunc(numpy.sqrt,  x)
def log(x):     return _apply_ufunc(numpy.log,   x)
def log10(x):   return _apply_ufunc(numpy.log10, x)
def exp(x):     return _apply_ufunc(numpy.exp,   x)
def sin(x):     return _apply_ufunc(numpy.sin,   x)
def cos(x):     return _apply_ufunc(numpy.cos,   x)
def tan(x):     return _apply_ufunc(numpy.tan,   x)
def arcsin(x):  return _apply_ufunc(numpy.arcsin,   x)
def arccos(x):  return _apply_ufunc(numpy.arccos,   x)
def arctan(x):  return _apply_ufunc(numpy.arctan,   x)
def sinh(x):    return _apply_ufunc(numpy.sinh,  x)
def cosh(x):    return _apply_ufunc(numpy.cosh,  x)
def tanh(x):    return _apply_ufunc(numpy.tanh,  x)
def arcsinh(x): return _apply_ufunc(numpy.arcsinh,  x)
def arccosh(x): return _apply_ufunc(numpy.arccosh,  x)
def arctanh(x): return _apply_ufunc(numpy.arctanh,  x)
def abs(x):     return _apply_ufunc(abs,         x)
def fabs(x):    return _apply_ufunc(numpy.fabs,  x)
def conj(x):    return _apply_ufunc(numpy.conj,  x)
def real(x):    return _apply_ufunc(numpy.real,  x)
def imag(x):    return _apply_ufunc(numpy.imag,  x)

# Recall that _apply_ufunc acts on the jackknife samples, hence the axis of the observable must be shifted by 1
def sum(x, axis=1):           return _apply_ufunc(numpy.sum, x, axis=axis+1)
def mean(x, axis=1):          return _apply_ufunc(numpy.mean, x, axis=axis+1)
def flip(x, axis=1):          return _apply_ufunc(numpy.flip, x, axis=axis+1)
def roll(x, axis=1, shift=0): return _apply_ufunc(numpy.roll, x, axis=axis+1, shift=shift)
def squeeze(x, axis=None):    return _apply_ufunc(numpy.squeeze, x, axis=axis+1)
def concatenate(x, axis=1):   return _apply_ufunc(numpy.concatenate, x, axis=axis+1)
def transpose(x, axes=None):  return _apply_ufunc(numpy.transpose, x, axes=axes)
def dot(x, y):                return _apply_ufunc(numpy.dot, x, y)

def increase_statistics(obs, new_data):
    '''
    Increase the statistics of the observable by appending new data to the existing data.
    Apply only to primary observables!

    Parameters
    ----------
    new_data : numpy.ndarray
        The new data to be appended to the existing data.

    Notes
    -----
    This method recomputes the jackknife samples and the statistical properties of the observable.
    '''
    if obs.primary is False:
        print('[pyjack.increase_statistics]Warning: increase_statistics should only be applied to primary observables')
    increased_obs = observable(description=obs.description, label=obs.label)
    increased_data = numpy.append(obs.data, new_data, axis=0)
    increased_obs.create(increased_data)
    return increased_obs

# def sum(obs,axis=0):
#     new_jack_samples = obs.jack_samples.sum(axis=axis)
#     new_obs = observable(description=obs.description, label=obs.label)
#     new_obs.create_from_jack_samples(new_jack_samples)
#     new_obs.primary = False
#     return new_obs

# def mean(self,axis=0):
#     new_jack_samples = obs.jack_samples.mean(axis=axis)
#     new_obs = observable(description=obs.description, label=obs.label)
#     new_obs.create_from_jack_samples(new_jack_samples)
#     new_obs.primary = False
#     return new_obs

# def flip(self,axis=0):
#     return self._new(self.jack_samples.flip(axis=axis))

# def roll(self,axis=0,shift=1):
#     return self._new(numpy.roll(self.jack_samples,axis=axis,shift=shift))

# def remove_tensor(self,axis=None):
#     return self._new(self.jack_samples.squeeze(axis=axis))

# def concatenate(self, new_data):
#     '''
#     Concatenate two arrays along the last axis, ensuring compatibility.

#     Parameters
#     ----------
#     new_data : numpy.ndarray
#         The new observable array to be concatenated. It must have the same first dimensions as `data`.

#     Returns
#     -------
#     numpy.ndarray
#         The concatenated array with the same first dimension as the inputs and combined size along the last axis.

#     Raises
#     ------
#     ValueError
#         If `data` and `new_data` do not have the same first dimension or if their shapes beyond the first and last dimensions do not match.

#     Examples
#     --------
#     >>> concatenate(numpy.zeros((100,10,4)), numpy.zeros((100,10)))
#     array with shape (100,10,5)
#     >>> concatenate(numpy.zeros((100,10)), numpy.zeros((100)))
#     array with shape (100,11)
#     >>> concatenate(numpy.zeros((100,10,4)), numpy.zeros((100)))
#     ValueError: data and new_data must have the same first dimension
#     '''
#     data = self.data
#     if data.shape[0] != new_data.shape[0]:
#         raise ValueError('data and new_data must have the same first dimension')
#     while new_data.ndim < data.ndim:
#         new_data = new_data[..., numpy.newaxis]
#     if data.shape[1:-1] != new_data.shape[1:-1]:
#         raise ValueError('Shape mismatch beyond the first and last dimensions')
#     new_data = numpy.concatenate((data, new_data), axis=-1)
#     self.data = new_data
#     self.jack_samples = self.compute_jack_samples(new_data)
#     self.compute_stats_from_jack_samples(self.jack_samples)