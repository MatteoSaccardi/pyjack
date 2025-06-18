import numpy

def _apply_ufunc(func, x):
    return x._new(func(x.jack_samples))

def sqrt(x):  return _apply_ufunc(numpy.sqrt,  x)
def log(x):   return _apply_ufunc(numpy.log,   x)
def log10(x): return _apply_ufunc(numpy.log10, x)
def exp(x):   return _apply_ufunc(numpy.exp,   x)
def sin(x):   return _apply_ufunc(numpy.sin,   x)
def cos(x):   return _apply_ufunc(numpy.cos,   x)
def tan(x):   return _apply_ufunc(numpy.tan,   x)
def sinh(x):  return _apply_ufunc(numpy.sinh,  x)
def cosh(x):  return _apply_ufunc(numpy.cosh,  x)
def tanh(x):  return _apply_ufunc(numpy.tanh,  x)
def abs(x):   return _apply_ufunc(abs,         x)
def fabs(x):  return _apply_ufunc(numpy.fabs,  x)
def conj(x):  return _apply_ufunc(numpy.conj,  x)
def real(x):  return _apply_ufunc(numpy.real,  x)
def imag(x):  return _apply_ufunc(numpy.imag,  x)
