'''
import numpy; from pyjobs import pyjobs; import pyobs
data = numpy.random.randn(100,10); obs1 = pyjobs.observable(); obs1.create(data); obs11 = pyobs.observable(); obs11.create('a',data.reshape(-1),shape=(10,))
'''

import numpy
import matplotlib.pyplot as plt
from .utils import pretty_print, save, load

plt.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def create_jack_samples(data):
    '''
    Compute jackknife samples of input data.

    Parameters
    ----------
    data : array_like
        Input data, shape (N, ...).

    Returns
    -------
    jack_samples : array_like
        Jackknife samples, shape (N, ...).
    '''
    return numpy.array([
            numpy.mean(numpy.delete(data, i, axis=0), axis=0)
            for i in range(data.shape[0])
        ])

class observable:
    def __init__(self, description=None, label=None):
        '''
        Initialize an observable object with optional description and label.

        Parameters
        ----------
        description : str, optional
            A brief description of the observable (default is None).
        label : str, optional
            A label for the observable, used for identification (default is None).

        Attributes
        ----------
        description : str
            Description of the observable.
        label : str
            Label of the observable.
        data : numpy.ndarray or None
            The input data associated with the observable.
        jack_samples : numpy.ndarray or None
            Jackknife samples derived from the input data.
        N : int or None
            The number of configurations in the input data.
        shape : tuple or None
            The shape of the observable without the configuration axis.
        mean : numpy.ndarray or None
            The mean of the jackknife samples.
        err : numpy.ndarray or None
            The standard deviation of the jackknife samples.
        cov : numpy.ndarray or None
            The covariance matrix of the jackknife samples.
        tau_int : numpy.ndarray or None
            The integrated autocorrelation time of the observable.
        '''
        self.description = description
        self.label = label
        
        self.data = None
        self.jack_samples = None
        self.N = None
        self.shape = None
        self.mean = None
        self.err = None
        self.cov = None
        self.tau_int = None
    
    def create(self, data, axis=0):
        '''
        Creates the observable with data and computes statistical properties.

        Parameters:
        data (array-like): The input data to be used for the observable. It is assumed
                        to have configurations as the first axis unless specified otherwise.
        axis (int, optional): The axis of the data that represents configurations. If not
                            zero, the data is moved to have configurations along the first axis.

        Attributes:
        data (numpy.ndarray): The input data with configurations along the first axis.
        N (int): The number of configurations.
        shape (tuple): The shape of the observable without the configuration axis.
        jack_samples (numpy.ndarray): Jackknife samples, excluding one configuration at a time.
        mean (numpy.ndarray): The mean of the jackknife samples.
        err (numpy.ndarray): The standard deviation of the jackknife samples.
        cov (numpy.ndarray): The covariance matrix of the jackknife samples.
        tau_int (numpy.ndarray): The integrated autocorrelation time of the observable.
        '''
        if axis != 0:
            data = numpy.moveaxis(data, axis, 0)
        self.data = data
        self.N = data.shape[0]
        self.shape = data.shape[1:] # shape of observable without config axis
        # Compute jackknife samples: shape (N, ...) where
        # jack_samples[i] = mean of data excluding data[i]
        self.jack_samples = create_jack_samples(data)
        self.compute_stats_from_jack_samples(self.jack_samples)
    
    def compute_stats_from_jack_samples(self, jack_samples):
        '''
        Computes statistical properties from jackknife samples.

        Parameters:
        jack_samples (numpy.ndarray): Jackknife samples, excluding one configuration at a time.

        Attributes:
        mean (numpy.ndarray): The mean of the jackknife samples.
        err (numpy.ndarray): The standard deviation of the jackknife samples.
        cov (numpy.ndarray): The covariance matrix of the jackknife samples.
        tau_int (numpy.ndarray): The integrated autocorrelation time of the observable.
        '''
        self.jack_samples = jack_samples
        self.N = self.jack_samples.shape[0]
        
        self.mean = numpy.mean(self.jack_samples, axis=0)
        diffs = self.jack_samples - self.mean
        self.err = numpy.sqrt((self.N - 1) / self.N * numpy.sum(numpy.abs(diffs)**2, axis=0))

        reshaped = self.jack_samples.reshape(self.N, -1)
        diffs = reshaped - reshaped.mean(axis=0)
        self.cov = (self.N - 1) / self.N * (diffs.T @ numpy.conj(diffs))

        # Autocorrelation time
        self.tau_int = numpy.zeros(reshaped.shape[1])
        for i in range(reshaped.shape[1]):
            v = reshaped[:, i] - numpy.mean(reshaped[:, i])
            norm = numpy.dot(v, numpy.conj(v)).real
            acf = []
            for lag in range(1, self.N // 2):
                c_lag = numpy.dot(v[:-lag], numpy.conj(v[lag:])).real
                acf_val = c_lag / norm
                if acf_val < 0:
                    break
                acf.append(acf_val)
            self.tau_int[i] = 0.5 + numpy.sum(acf)
        self.tau_int = self.tau_int.reshape(self.shape)

    def create_from_jack_samples(self, jack_samples):
        '''
        Initialize observable from jackknife samples.
        
        Parameters:
        jack_samples (numpy.ndarray): Jackknife samples, excluding one configuration at a time.
        
        Attributes:
        mean (numpy.ndarray): The mean of the jackknife samples.
        err (numpy.ndarray): The standard deviation of the jackknife samples.
        cov (numpy.ndarray): The covariance matrix of the jackknife samples.
        tau_int (numpy.ndarray): The integrated autocorrelation time of the observable.
        '''
        self.jack_samples = jack_samples
        self.compute_stats_from_jack_samples(self.jack_samples)

    def create_from_cov(self, mean, cov):
        '''
        Create observable from mean and covariance matrix.
        
        Parameters
        ----------
        mean : array_like
            Mean of observable, shape (...)
        cov : array_like
            Covariance matrix of observable, shape (..., ...)
        err : array_like
            Standard error of observable, shape (...)
        '''
        self.mean = mean
        self.cov = cov
        self.err = numpy.sqrt(numpy.diag(cov))

    def error(self):
        '''
        Standard error of observable.
        
        Returns
        -------
        error : array_like
            Standard error of observable, shape (...)
        '''
        return self.err

    def covariance_matrix(self):
        '''
        Covariance matrix of observable.
        
        Returns
        -------
        cov : array_like
            Covariance matrix of observable, shape (..., ...)
        '''
        return self.cov

    def autocorr_time(self):
        '''
        Autocorrelation time of observable.
        
        Returns
        -------
        tau_int : array_like
            Autocorrelation time of observable, shape (...)
        '''
        return self.tau_int

    def plot_autocorrelation(self, which_obs=[0]):
        N = self.N
        data = self.data.reshape(N,-1)
        
        plt.figure(figsize=(10,5))

        for idx,iobs in enumerate(which_obs):
            mean_obs = numpy.mean(data[:,iobs])
            diffs0 = data[:,iobs] - mean_obs
            gamma0 = diffs0 @ diffs0 / N
            gammas = [ gamma0 ]
            # tauint = 0.5
            for t in range(1, int(N/2)):
                diffs_i = data[0:N-t,iobs] - mean_obs
                diffs_i_plus_t = data[t:,iobs] - mean_obs
                gamma_t = diffs_i @ diffs_i_plus_t / ( N - t )
                gammas.append(gamma_t)
                # tauint += gamma_t / gamma0
            plt.plot(gammas, color=f'C{idx}', label=rf'Obs {iobs}, $\tau_{{\mathrm{{int}}}} = {self.tau_int[iobs]:.2f}$')

        plt.legend()
        plt.xlabel(r'$t$')
        plt.title(rf'$\Gamma(t) = \frac{{1}}{{N-t}} \sum_{{i=1}}^{{N-t}} (\mathcal{{O}}_i - \overline{{\mathcal{{O}}}}) (\mathcal{{O}}_{{i+t}} - \overline {{\mathcal{{O}}}})$ , $N={N}$')
        plt.tight_layout()
        plt.draw()
        plt.show()

    def save(self, filename):
        save(self, filename)
    
    def load(filename):
        return load(filename)

    def __repr__(self):
        if self.mean is not None and self.err is not None:
            mean_flat = self.mean.flatten()
            err_flat = self.err.flatten()
            formatted = ', '.join(pretty_print(x, dx) for x, dx in zip(mean_flat, err_flat))
            return f'pyjobs({formatted}, description={self.description})'
        return f'pyjobs(mean={self.mean}, err={self.err}, description={self.description})'

    def _new(self, new_jack_samples):
        new_obs = observable(description=self.description, label=self.label)
        new_obs.create_from_jack_samples(new_jack_samples)
        return new_obs
    
    # Arithmetic operations
    def __add__(self, other):
        if isinstance(other, observable):
            new_jack_samples = self.jack_samples + other.jack_samples
        else:
            new_jack_samples = self.jack_samples + other
        return self._new(new_jack_samples)

    def __radd__(self, other): return self + other

    def __sub__(self, other):
        if isinstance(other, observable):
            new_jack_samples = self.jack_samples - other.jack_samples
        else:
            new_jack_samples = self.jack_samples - other
        return self._new(new_jack_samples)

    def __rsub__(self, other): return self._new(other - self.jack_samples)

    def __mul__(self, other):
        if isinstance(other, observable):
            new_jack_samples = self.jack_samples * other.jack_samples
        else:
            new_jack_samples = self.jack_samples * other
        return self._new(new_jack_samples)

    def __rmul__(self, other): return self * other

    def __truediv__(self, other):
        if isinstance(other, observable):
            new_jack_samples = self.jack_samples / other.jack_samples
        else:
            new_jack_samples = self.jack_samples / other
        return self._new(new_jack_samples)

    def __rtruediv__(self, other): return self._new(other / self.jack_samples)

    def __pow__(self, power):
        if isinstance(power, observable):
            new_jack_samples = self.jack_samples ** power.jack_samples
        else:
            new_jack_samples = self.jack_samples ** power
        return self._new(new_jack_samples)

    def __matmul__(self, other):
        if isinstance(other, observable):
            new_jack_samples = self.jack_samples @ other.jack_samples
        else:
            new_jack_samples = self.jack_samples @ other
        return self._new(new_jack_samples)

    def __rmatmul__(self, other):
        if isinstance(other, observable):
            new_jack_samples = other.jack_samples @ self.jack_samples
        else:
            new_jack_samples = other @ self.jack_samples
        return self._new(new_jack_samples)
    
    # Universal functions
    @staticmethod
    def _apply_ufunc(func, x):
        return x._new(func(x.jack_samples))

    @classmethod
    def sqrt(cls, x): return cls._apply_ufunc(numpy.sqrt, x)
    @classmethod
    def log(cls, x): return cls._apply_ufunc(numpy.log, x)
    @classmethod
    def log10(cls, x): return cls._apply_ufunc(numpy.log10, x)
    @classmethod
    def exp(cls, x): return cls._apply_ufunc(numpy.exp, x)
    @classmethod
    def sin(cls, x): return cls._apply_ufunc(numpy.sin, x)
    @classmethod
    def cos(cls, x): return cls._apply_ufunc(numpy.cos, x)
    @classmethod
    def tan(cls, x): return cls._apply_ufunc(numpy.tan, x)
    @classmethod
    def sinh(cls, x): return cls._apply_ufunc(numpy.sinh, x)
    @classmethod
    def cosh(cls, x): return cls._apply_ufunc(numpy.cosh, x)
    @classmethod
    def tanh(cls, x): return cls._apply_ufunc(numpy.tanh, x)
    @classmethod
    def abs(cls, x): return cls._apply_ufunc(numpy.abs, x)
    @classmethod
    def conj(cls, x): return cls._apply_ufunc(numpy.conj, x)
    @classmethod
    def real(cls, x): return cls._apply_ufunc(numpy.real, x)
    @classmethod
    def imag(cls, x): return cls._apply_ufunc(numpy.imag, x)