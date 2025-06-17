'''
import numpy; from pyjobs import pyjobs; import pyobs
data = numpy.random.randn(100,10); obs1 = pyjobs.observable(); obs1.create(data); obs11 = pyobs.observable(); obs11.create('a',data.reshape(-1),shape=(10,))
'''

import numpy
import matplotlib.pyplot as plt

def pretty_print(x, dx, further_digits=1):
    '''
    Return a string representation of x with uncertainty dx.

    Parameters
    ----------
    x : float
        The value to be printed.
    dx : float
        The uncertainty associated to x.
    further_digits : int, optional
        The number of digits to be printed beyond the uncertainty (default: 1).

    Returns
    -------
    string
        A string representation of x with uncertainty dx.
    '''
    if dx == 0:
        return f'{x}(0)'
    decimal_places = max(0, -int(numpy.floor(numpy.log10(dx)))) + further_digits
    return f'{x:.{decimal_places}f}({int(dx * 10**decimal_places)})'

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
        std : numpy.ndarray or None
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
        self.std = None
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
        std (numpy.ndarray): The standard deviation of the jackknife samples.
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
        std (numpy.ndarray): The standard deviation of the jackknife samples.
        cov (numpy.ndarray): The covariance matrix of the jackknife samples.
        tau_int (numpy.ndarray): The integrated autocorrelation time of the observable.
        '''
        self.jack_samples = jack_samples
        self.N = self.jack_samples.shape[0]
        
        self.mean = numpy.mean(self.jack_samples, axis=0)
        diffs = self.jack_samples - self.mean
        self.std = numpy.sqrt((self.N - 1) / self.N * numpy.sum(numpy.abs(diffs)**2, axis=0))

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
        std (numpy.ndarray): The standard deviation of the jackknife samples.
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
        '''
        self.mean = mean
        self.cov = cov

    def error(self):
        '''
        Standard error of observable.
        
        Returns
        -------
        error : array_like
            Standard error of observable, shape (...)
        '''
        return self.std

    def covariance(self):
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

    def plot_autocorr_and_iat(self, want_return=False):
        '''
        Plot the autocorrelation function (ACF) and integrated autocorrelation time (IAT).

        The ACF is computed up to max lag = N/2. 
        The IAT is the cumulative sum of ACF plus 0.5 as per definition.

        Returns
        -------
        None
        '''
        reshaped = self.jack_samples.reshape(self.N, -1)
        centered = reshaped - reshaped.mean(axis=0)
        max_lag = self.N // 2
        acfs = []
        for i in range(centered.shape[1]):
            v = centered[:, i]
            norm = numpy.dot(v, numpy.conj(v)).real
            acf = []
            for lag in range(1, max_lag):
                c_lag = numpy.dot(v[:-lag], numpy.conj(v[lag:])).real
                acf_val = c_lag / norm
                if acf_val < 0:  # stop sum if autocorr negative (common practice)
                    break
                acf.append(acf_val)
            acfs.append(acf)
        
        # Pad acfs so they have the same length (max length among all)
        max_len = max(len(a) for a in acfs)
        acfs_padded = numpy.array([a + [0]*(max_len - len(a)) for a in acfs])
        avg_acf = numpy.mean(acfs_padded, axis=0)

        # Compute integrated autocorrelation time as function of lag
        iat = 0.5 + numpy.cumsum(avg_acf)

        # Plot
        fig, ax1 = plt.subplots()

        lags = range(1, len(avg_acf)+1)
        ax1.plot(lags, avg_acf, 'b-o', label='Autocorrelation')
        ax1.set_xlabel('Lag')
        ax1.set_ylabel('Autocorrelation', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(lags, iat, 'r--', label='Integrated Autocorr. Time')
        ax2.set_ylabel('Integrated Autocorrelation Time', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        plt.title(self.label or 'Autocorrelation & Integrated Autocorr. Time')
        fig.tight_layout()
        plt.show()

        if want_return:
            return avg_acf, iat



    def __repr__(self):
        if self.mean is not None and self.std is not None:
            mean_flat = self.mean.flatten()
            std_flat = self.std.flatten()
            formatted = ', '.join(pretty_print(x, dx) for x, dx in zip(mean_flat, std_flat))
            return f'pyjobs({formatted}, description={self.description})'
        return f'pyjobs(mean={self.mean}, std={self.std}, description={self.description})'

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