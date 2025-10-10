import numpy
import matplotlib.pyplot as plt
from .utils import pretty_print, save, load

plt.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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
        
        self.primary = True
        self.creator = None
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
        self.primary = True
        self.creator = 'create'
        if axis != 0:
            data = numpy.moveaxis(data, axis, 0)
        self.data = data
        self.N = data.shape[0]
        self.shape = data.shape[1:] # shape of observable without config axis
        # Compute jackknife samples: shape (N, ...) where
        # jack_samples[i] = mean of data excluding data[i]
        self.jack_samples = self.compute_jack_samples(data)
        self.compute_stats_from_jack_samples(self.jack_samples)

    def compute_jack_samples(self, data=None):
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
        if data is None:
            data = self.data
        return numpy.array([
                numpy.mean(numpy.delete(data, i, axis=0), axis=0)
                for i in range(data.shape[0])
            ])
        
    def compute_stats_from_jack_samples(self, jack_samples=None):
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
        if jack_samples is None:
            jack_samples = self.jack_samples
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

    def create_from_jack_samples(self, jack_samples=None):
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
        self.creator = 'create_from_jack_samples'
        if jack_samples is None:
            jack_samples = self.jack_samples
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
        self.creator = 'create_from_cov'
        self.mean = mean
        self.cov = cov
        if isinstance(cov, (int,float)):
            self.err = numpy.sqrt(cov)
        else:
            self.err = numpy.sqrt(numpy.diagonal(cov))

    def sample(self, N=1000, seed=42):
        '''
        Generate jackknife samples from mean and covariance matrix.

        Parameters
        ----------
        N : int, optional
            Number of jackknife samples to generate. Default is 1000.

        Returns
        -------
        data_samples : array_like
            Jackknife samples, shape (N, ...)
        '''
        numpy.random.seed(seed)

        obs = observable(description=self.description, label=self.label)

        mean = numpy.atleast_1d(self.mean)
        shape = mean.shape
        flat_mean = mean.ravel()
        dim = flat_mean.shape[0]

        # Flattened covariance matrix
        cov = numpy.atleast_2d(self.cov)
        if cov.shape != (dim, dim):
            raise ValueError(f'[pyjack.observable.sample] Covariance shape {cov.shape} incompatible with mean shape {shape}')

        # Generate fluctuations with zero mean and given covariance
        fluctuations = numpy.random.multivariate_normal(
            mean=numpy.zeros(dim),
            cov=cov * (N - 1),
            size=N
        )

        # Enforce jackknife constraint (zero mean of fluctuations)
        fluctuations -= fluctuations.mean(axis=0)
        
        # Construct jackknife samples
        data_samples = flat_mean[None, :] + fluctuations
        data_samples = data_samples.reshape((N,) + shape)
        obs.create(data_samples)
        return obs

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
    
    def increase_statistics(self, new_data):
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
        if self.primary is False:
            print('[observable.increase_statistics] Warning: increase_statistics should only be applied to primary observables')
        increased_data = numpy.append(self.data, new_data, axis=0)
        self.create(increased_data)

    def __repr__(self):
        if self.mean is not None and self.err is not None:
            if isinstance(self.mean,(int,float)):
                mean_flat = [ self.mean ]
                err_flat = [ self.err.flatten() ]
            else:
                mean_flat = self.mean.flatten()
                err_flat = self.err.flatten()
            formatted = ', '.join(pretty_print(x, dx) for x, dx in zip(mean_flat, err_flat))
            return f'pyjack({formatted}, description={self.description})'
        return f'pyjack(mean={self.mean}, err={self.err}, description={self.description})'

    def _new(self, new_jack_samples):
        new_obs = observable(description=self.description, label=self.label)
        new_obs.create_from_jack_samples(new_jack_samples)
        new_obs.primary = False
        return new_obs

    # Arithmetic operations
    def __add__(self, other):
        if isinstance(other, observable):
            if other.creator == 'create_from_cov':
                other_sampled = other.sample(self.N)
                new_jack_samples = self.jack_samples + other_sampled.jack_samples
            else:
                new_jack_samples = self.jack_samples + other.jack_samples
        else:
            new_jack_samples = self.jack_samples + other
        return self._new(new_jack_samples)

    def __radd__(self, other): return self + other

    def __sub__(self, other):
        if isinstance(other, observable):
            if other.creator == 'create_from_cov':
                other_sampled = other.sample(self.N)
                new_jack_samples = self.jack_samples - other_sampled.jack_samples
            else:
                new_jack_samples = self.jack_samples - other.jack_samples
        else:
            new_jack_samples = self.jack_samples - other
        return self._new(new_jack_samples)

    def __rsub__(self, other): return self._new(other - self.jack_samples)

    def __mul__(self, other):
        if isinstance(other, observable):
            if other.creator == 'create_from_cov':
                other_sampled = other.sample(self.N)
                new_jack_samples = self.jack_samples * other_sampled.jack_samples
            else:
                new_jack_samples = self.jack_samples * other.jack_samples
        else:
            new_jack_samples = self.jack_samples * other
        return self._new(new_jack_samples)

    def __rmul__(self, other): return self * other

    def __truediv__(self, other):
        if isinstance(other, observable):
            if other.creator == 'create_from_cov':
                other_sampled = other.sample(self.N)
                new_jack_samples = self.jack_samples / other_sampled.jack_samples
            else:
                new_jack_samples = self.jack_samples / other.jack_samples
        else:
            new_jack_samples = self.jack_samples / other
        return self._new(new_jack_samples)

    def __rtruediv__(self, other): return self._new(other / self.jack_samples)

    def __pow__(self, other):
        if isinstance(other, observable):
            if other.creator == 'create_from_cov':
                other_sampled = other.sample(self.N)
                new_jack_samples = self.jack_samples ** other_sampled.jack_samples
            else:
                new_jack_samples = self.jack_samples ** other.jack_samples
        else:
            new_jack_samples = self.jack_samples ** other
        return self._new(new_jack_samples)

    def __matmul__(self, other):
        if isinstance(other, observable):
            if other.creator == 'create_from_cov':
                other_sampled = other.sample(self.N)
                new_jack_samples = self.jack_samples @ other_sampled.jack_samples
            else:
                new_jack_samples = self.jack_samples @ other.jack_samples
        else:
            new_jack_samples = self.jack_samples @ other
        return self._new(new_jack_samples)

    def __rmatmul__(self, other):
        if isinstance(other, observable):
            if other.creator == 'create_from_cov':
                other_sampled = other.sample(self.N)
                new_jack_samples = other_sampled.jack_samples @ self.jack_samples
            else:
                new_jack_samples = other.jack_samples @ self.jack_samples
        else:
            new_jack_samples = other @ self.jack_samples
        return self._new(new_jack_samples)

    def __getitem__(self,key):
        if self.jack_samples is not None:
            if not isinstance(key, tuple):
                key = (key,)
            new_jack_samples = numpy.array(self.jack_samples)[(slice(None),)+key]
            new_obs = observable(description=self.description, label=self.label)
            new_obs.create_from_jack_samples(new_jack_samples)
        else:
            mean = numpy.array(self.mean)[key]
            cov = numpy.array(self.cov)[key,key]
            new_obs = observable(description=self.description, label=self.label)
            new_obs.create_from_cov(mean,cov)
        new_obs.primary = False
        return new_obs
    
    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            key = (key,)

        # convert key to flat indices
        idx = numpy.arange(self.mean.size).reshape(self.mean.shape)[key].ravel() # .ravel equivalent to .reshape(-1), although more flexible
        not_idx = numpy.setdiff1d(numpy.arange(self.mean.size), idx)

        if isinstance(value, observable):
            # update jackknife samples (when available), means, errors and covariance

            try:
                self.jack_samples[(slice(None),) + key] = value.jack_samples
                self.compute_stats_from_jack_samples(self.jack_samples)
            except:
                self.jack_samples[(slice(None),) + key] = value.mean[None, ...]
                self.mean[key] = value.mean
                self.err[key] = value.err
                self.cov[numpy.ix_(idx, idx)] = value.cov
                self.cov[numpy.ix_(idx, not_idx)] = 0.0
                self.cov[numpy.ix_(not_idx, idx)] = 0.0
                return self

        elif isinstance(value, (numpy.ndarray, float, int)):
            self.jack_samples[(slice(None),)+key][:][key] = value[None, ...]
            self.mean[key] = value
            self.cov[numpy.ix_(idx, idx)] = 0.0
            self.cov[numpy.ix_(idx, not_idx)] = 0.0
            self.cov[numpy.ix_(not_idx, idx)] = 0.0
            self.err[key] = 0.0

        else:
            raise TypeError(f'[pyojack.observable.__setitem__] Type {type(value)} not supported.')

        

    
'''
# Comparison with pyobs
import numpy; from pyjack import pyjack; import pyobs
data = numpy.random.randn(100,10); obs1 = pyjack.observable(); obs1.create(data); obs11 = pyobs.observable(); obs11.create('a',data.reshape(-1),shape=(10,))
'''