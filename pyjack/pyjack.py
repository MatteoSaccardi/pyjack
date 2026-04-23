import numpy
import matplotlib.pyplot as plt
from .utils import pretty_print, save, load
from statistics import NormalDist

plt.rcParams.update({'font.size': 16})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import socket
from datetime import datetime

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
        # self.resampling_method = None
        self.data = None
        self.jack_samples = None
        self.N = None
        # self.shape = None
        self.mean = None
        self.err = None
        self.cov = None
        self.tau_int = None

        hostname = socket.gethostname()
        try:
            ip_address = socket.gethostbyname(hostname)
        except socket.gaierror:
            ip_address = 'unresolved'

        self.creation_info = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'hostname': hostname,
            'ip': ip_address
        }
    
    def create(self, data, axis=0, binsize=1, method='jackknife', n_resamples=1000, seed=None):
        '''
        Create the observable from raw data and compute resampling statistics.

        Parameters
        ----------
        data : array_like
            Input data. Configurations are assumed to lie on the first axis unless
            `axis` is specified otherwise.
        axis : int, optional
            Axis containing configurations. It is moved to the front before further
            processing.
        binsize : int, optional
            Size of the blocks used to bin the data before resampling.
        method : {'jackknife', 'bootstrap'}, optional
            Resampling method. The default is `'jackknife'` to preserve backward
            compatibility with existing code.
        n_resamples : int, optional
            Number of bootstrap replicas when `method='bootstrap'`.
        seed : int or None, optional
            Seed for bootstrap resampling. Ignored for jackknife.

        Attributes
        ----------
        data : numpy.ndarray
            The input data with configurations along the first axis.
        N : int
            The number of configurations or replicas used internally by the observable.
        shape : tuple
            The shape of the observable without the configuration axis.
        jack_samples : numpy.ndarray
            Internal replica array. For jackknife observables these are jackknife
            samples; for bootstrap observables these are bootstrap replicas. The
            name is preserved for backward compatibility.
        mean : numpy.ndarray
            The mean of the resampling replicas.
        err : numpy.ndarray
            The standard deviation of the observable estimated from the active
            resampling method.
        cov : numpy.ndarray
            The covariance matrix estimated from the active resampling method.
        tau_int : numpy.ndarray
            The integrated autocorrelation time of the observable.
        '''
        self.primary = True
        self.creator = 'create'
        self.resampling_method = method
        if axis != 0:
            data = numpy.moveaxis(data, axis, 0)
        
        if binsize < 1:
            raise ValueError('[pyjack.observable.create] binsize must be >= 1')

        # Binning happens before either jackknife or bootstrap, so the existing
        # jackknife workflow is preserved and the bootstrap path naturally becomes
        # a block bootstrap over the binned configurations.
        N_orig = data.shape[0]
        if binsize > 1:
            n_bins = int(numpy.ceil(N_orig / binsize))
            data = numpy.array([numpy.mean(b, axis=0) for b in numpy.array_split(data, n_bins, axis=0)])

        self.data = data
        self.N = data.shape[0]

        if method == 'jackknife':
            self.jack_samples = self.compute_jack_samples(data)
            self.compute_stats_from_jack_samples(self.jack_samples)
        elif method == 'bootstrap':
            self.jack_samples = self.compute_bootstrap_samples(data, n_resamples=n_resamples, seed=seed)
            self.compute_stats_from_bootstrap_samples(self.jack_samples)
        else:
            raise ValueError(f"[pyjack.observable.create] Unknown method '{method}'. Use 'jackknife' or 'bootstrap'.")

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
        
        # OLD CODE
        # return numpy.array([
        #         numpy.mean(numpy.delete(data, i, axis=0), axis=0)
        #         for i in range(data.shape[0])
        #     ])

        # Sum all bins, then subtract each bin to get the 'leave-one-out' sum
        total_sum = numpy.sum(data, axis=0)
        # Jackknife sample i is (Total - bin_i) / (N_bins - 1)
        return (total_sum - data) / (data.shape[0] - 1)

    def compute_bootstrap_samples(self, data=None, n_resamples=1000, seed=None):
        '''
        Compute bootstrap replicas by resampling the binned configurations with replacement.

        Parameters
        ----------
        data : array_like, optional
            Input data with shape `(N, ...)`, where `N` is the number of binned
            configurations.
        n_resamples : int, optional
            Number of bootstrap replicas to generate.
        seed : int or None, optional
            Seed used to initialize the bootstrap random number generator.

        Returns
        -------
        bootstrap_samples : array_like
            Bootstrap replicas with shape `(n_resamples, ...)`.
        '''
        if data is None:
            data = self.data

        if n_resamples is None or n_resamples < 1:
            raise ValueError('[pyjack.observable.compute_bootstrap_samples] n_resamples must be >= 1')

        rng = numpy.random.default_rng(seed)
        n_cfg = data.shape[0]
        indices = rng.integers(0, n_cfg, size=(n_resamples, n_cfg))
        return numpy.mean(data[indices], axis=1)
        
    def compute_stats_from_jack_samples(self, jack_samples=None):
        '''
        Computes statistical properties from jackknife samples.

        Parameters
        ----------
        jack_samples : numpy.ndarray
            Jackknife samples, excluding one configuration at a time.

        Attributes
        ----------
        mean : numpy.ndarray
            The mean of the jackknife samples.
        err : numpy.ndarray
            The standard deviation of the jackknife samples.
        cov : numpy.ndarray
            The covariance matrix of the jackknife samples.
        tau_int : numpy.ndarray
            The integrated autocorrelation time of the observable.
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

    def compute_stats_from_bootstrap_samples(self, bootstrap_samples=None):
        '''
        Compute statistical properties from bootstrap replicas.

        Parameters
        ----------
        bootstrap_samples : array_like, optional
            Bootstrap replicas with shape `(Nrep, ...)`.

        Notes
        -----
        Bootstrap replicas are treated as ordinary Monte Carlo draws of the estimator,
        so errors and covariances use the usual sample formulas with `ddof=1`.
        '''
        if bootstrap_samples is None:
            bootstrap_samples = self.jack_samples

        self.jack_samples = bootstrap_samples
        self.N = self.jack_samples.shape[0]

        self.mean = numpy.mean(self.jack_samples, axis=0)
        self.err = numpy.std(self.jack_samples, axis=0, ddof=1)

        reshaped = self.jack_samples.reshape(self.N, -1)
        self.cov = numpy.cov(reshaped, rowvar=False, ddof=1)

        # Bootstrap replicas do not define an autocorrelation time. When primary data
        # are available, estimate tau_int from the underlying binned time series instead.
        if self.data is not None:
            self.tau_int = self.compute_tau_int_from_data(self.data)
        else:
            self.tau_int = numpy.full(self.shape, numpy.nan)

    def compute_tau_int_from_data(self, data=None):
        '''
        Estimate the integrated autocorrelation time from a configuration time series.

        Parameters
        ----------
        data : array_like, optional
            Time-ordered data with configurations on the first axis.

        Returns
        -------
        tau_int : array_like
            Integrated autocorrelation time with the observable shape.
        '''
        if data is None:
            data = self.data

        reshaped = numpy.asarray(data).reshape(data.shape[0], -1)
        tau_int = numpy.zeros(reshaped.shape[1])
        for i in range(reshaped.shape[1]):
            v = reshaped[:, i] - numpy.mean(reshaped[:, i])
            norm = numpy.dot(v, numpy.conj(v)).real
            if norm == 0:
                tau_int[i] = 0.5
                continue

            acf = []
            for lag in range(1, reshaped.shape[0] // 2):
                c_lag = numpy.dot(v[:-lag], numpy.conj(v[lag:])).real
                acf_val = c_lag / norm
                if acf_val < 0:
                    break
                acf.append(acf_val)
            tau_int[i] = 0.5 + numpy.sum(acf)

        return tau_int.reshape(data.shape[1:])

    def compute_bca_interval(self, level=0.68):
        '''
        Compute a bias-corrected and accelerated (BCa) bootstrap confidence interval.

        Parameters
        ----------
        level : float, optional
            Central coverage probability, e.g. `0.68` or `0.95`.

        Returns
        -------
        tuple of array_like
            Lower and upper BCa bounds with the observable shape.

        Notes
        -----
        BCa intervals improve on plain percentile intervals by correcting for:

        - bias in the bootstrap distribution (`z0`)
        - skewness / nonlinearity through the acceleration parameter (`a`)

        The acceleration is estimated from a jackknife over the underlying binned
        primary data. Therefore BCa is only available when the observable still
        retains primary data.
        '''
        if getattr(self, 'resampling_method', 'jackknife') != 'bootstrap':
            print('[pyjack.observable.compute_bca_interval] Warning: BCa is defined here only for bootstrap observables. Falling back to normal interval.')
            return self.confidence_interval(level=level, method='normal')

        if self.data is None:
            print("[pyjack.observable.compute_bca_interval] Warning: BCa requires underlying primary data. Falling back to percentile interval.")
            return self.confidence_interval(level=level, method='percentile')

        alpha = 0.5 * (1.0 - level)
        normal = NormalDist()

        samples = self.jack_samples.reshape(self.N, -1)
        theta_hat = numpy.asarray(self.mean).reshape(-1)

        # Bias-correction term from the fraction of bootstrap replicas below the
        # observed estimate. Clipping avoids infinities in the inverse normal CDF.
        prop_less = numpy.mean(samples < theta_hat[None, :], axis=0)
        eps = 0.5 / self.N
        prop_less = numpy.clip(prop_less, eps, 1.0 - eps)
        z0 = numpy.array([normal.inv_cdf(p) for p in prop_less])

        # Acceleration comes from the classical jackknife influence values computed
        # on the underlying binned primary data.
        jk = self.compute_jack_samples(self.data).reshape(self.data.shape[0], -1)
        jk_mean = numpy.mean(jk, axis=0)
        delta = jk_mean[None, :] - jk
        num = numpy.sum(delta ** 3, axis=0)
        den = 6.0 * numpy.sum(delta ** 2, axis=0) ** 1.5
        accel = numpy.divide(num, den, out=numpy.zeros_like(num), where=den != 0)

        z_alpha_low = normal.inv_cdf(alpha)
        z_alpha_high = normal.inv_cdf(1.0 - alpha)

        def adjusted_prob(z_alpha):
            numer = z0 + z_alpha
            denom = 1.0 - accel * numer
            adjusted = z0 + numer / denom
            probs = numpy.array([normal.cdf(val) for val in adjusted])
            return numpy.clip(probs, 0.0, 1.0)

        p_low = adjusted_prob(z_alpha_low)
        p_high = adjusted_prob(z_alpha_high)

        lower = numpy.array([
            numpy.quantile(samples[:, i], p_low[i]) for i in range(samples.shape[1])
        ])
        upper = numpy.array([
            numpy.quantile(samples[:, i], p_high[i]) for i in range(samples.shape[1])
        ])

        return lower.reshape(self.shape), upper.reshape(self.shape)

    def create_from_jack_samples(self, jack_samples=None):
        '''
        Initialize observable from jackknife samples.
        
        Parameters
        ----------
        jack_samples : numpy.ndarray
            Jackknife samples, excluding one configuration at a time.
        
        Attributes
        ----------
        mean : numpy.ndarray
            The mean of the jackknife samples.
        err : numpy.ndarray
            The standard deviation of the jackknife samples.
        cov : numpy.ndarray
            The covariance matrix of the jackknife samples.
        tau_int : numpy.ndarray
            The integrated autocorrelation time of the observable.
        '''
        self.creator = 'create_from_jack_samples'
        self.resampling_method = 'jackknife'
        if jack_samples is None:
            jack_samples = self.jack_samples
        self.jack_samples = jack_samples
        self.compute_stats_from_jack_samples(self.jack_samples)
        self.data = self.data_from_jack()

    def create_from_bootstrap_samples(self, bootstrap_samples=None):
        '''
        Initialize observable from bootstrap replicas.

        Parameters
        ----------
        bootstrap_samples : array_like
            Bootstrap replicas with shape `(Nrep, ...)`.

        Attributes
        ----------
        mean : numpy.ndarray
            The mean of the bootstrap replicas.
        err : numpy.ndarray
            The standard deviation estimated from the bootstrap replicas.
        cov : numpy.ndarray
            The covariance matrix estimated from the bootstrap replicas.
        tau_int : numpy.ndarray
            The integrated autocorrelation time estimated from the underlying data
            when available, otherwise `nan`.
        '''
        self.creator = 'create_from_bootstrap_samples'
        self.resampling_method = 'bootstrap'
        if bootstrap_samples is None:
            bootstrap_samples = self.jack_samples
        self.jack_samples = bootstrap_samples
        self.data = None
        self.compute_stats_from_bootstrap_samples(self.jack_samples)

    def create_from_cov(self, mean, cov):
        '''
        Create observable from mean and covariance matrix.
        
        Parameters
        ----------
        mean : array_like
            Mean of observable, shape (...)
        cov : array_like
            Covariance matrix of observable, shape (..., ...)

        Attributes
        ----------
        mean : numpy.ndarray
            Central value of the observable.
        err : numpy.ndarray
            Standard deviation extracted from the covariance.
        cov : numpy.ndarray
            Covariance matrix of the observable.
        '''
        self.creator = 'create_from_cov'
        self.resampling_method = None
        self.mean = mean
        self.cov = cov
        if isinstance(cov, (int,float)):
            self.err = numpy.sqrt(cov)
        else:
            self.err = numpy.sqrt(numpy.diagonal(cov))

    def sample(self, N=1000, seed=42, method=None):
        '''
        Generate resampling replicas from mean and covariance matrix.

        Parameters
        ----------
        N : int, optional
            Number of replicas to generate. Default is 1000.
        seed : int, optional
            Seed used for random generation.
        method : {'jackknife', 'bootstrap'} or None, optional
            Replica type to generate. If omitted, uses the observable method when
            available and falls back to `'jackknife'` for backward compatibility.

        Returns
        -------
        observable
            A new observable generated from synthetic replicas consistent with the
            current mean and covariance.
        '''
        obs = observable(description=self.description, label=self.label)
        target_method = method or self.resampling_method or 'jackknife'

        mean = numpy.atleast_1d(self.mean)
        shape = mean.shape
        flat_mean = mean.ravel()
        dim = flat_mean.shape[0]

        # Flattened covariance matrix
        cov = numpy.atleast_2d(self.cov)
        if cov.shape != (dim, dim):
            raise ValueError(f'[pyjack.observable.sample] Covariance shape {cov.shape} incompatible with mean shape {shape}')

        rng = numpy.random.default_rng(seed)

        if target_method == 'jackknife':
            fluctuations = rng.multivariate_normal(
                mean=numpy.zeros(dim),
                cov=cov * (N - 1),
                size=N
            )
            fluctuations -= fluctuations.mean(axis=0)
            data_samples = flat_mean[None, :] + fluctuations
            data_samples = data_samples.reshape((N,) + shape)
            obs.create(data_samples, method='jackknife')
        elif target_method == 'bootstrap':
            data_samples = rng.multivariate_normal(
                mean=flat_mean,
                cov=cov,
                size=N
            )
            data_samples = data_samples.reshape((N,) + shape)
            obs.create_from_bootstrap_samples(data_samples)
        else:
            raise ValueError(f"[pyjack.observable.sample] Unknown method '{target_method}'. Use 'jackknife' or 'bootstrap'.")
        return obs
    
    def data_from_jack(self):
        '''
        Reconstruct original data from jackknife samples.
        '''
        jk_samples = self.jack_samples
        jk = numpy.asarray(jk_samples)
        N = jk.shape[0]
        mean_full = jk.mean(axis=0)
        data = N * mean_full - (N - 1) * jk
        return data

    def confidence_interval(self, level=0.68, method='normal'):
        '''
        Compute a confidence interval for the observable.

        Parameters
        ----------
        level : float, optional
            Central coverage probability, e.g. `0.68` or `0.95`.
        method : {'normal', 'percentile', 'bca'}, optional
            Interval construction method.

            - `'normal'`: symmetric interval `mean +/- z * err`
            - `'percentile'`: bootstrap percentile interval from the stored replicas
            - `'bca'`: bias-corrected and accelerated bootstrap interval

        Returns
        -------
        tuple of array_like
            Lower and upper bounds with the observable shape.
        '''
        if not (0.0 < level < 1.0):
            raise ValueError('[pyjack.observable.confidence_interval] level must be between 0 and 1')

        alpha = 0.5 * (1.0 - level)

        # Jackknife naturally supports the usual symmetric error interval. If a
        # bootstrap-style interval is requested, keep the workflow alive but report
        # the method switch explicitly.
        if getattr(self, 'resampling_method', 'jackknife')  == 'jackknife' and method != 'normal':
            print(f"[pyjack.observable.confidence_interval] Warning: method='{method}' is not available for jackknife observables. Using method='normal' instead.")
            method = 'normal'

        if method == 'normal':
            z = NormalDist().inv_cdf(1.0 - alpha)
            return self.mean - z * self.err, self.mean + z * self.err

        if method == 'percentile':
            if getattr(self, 'resampling_method', 'jackknife') != 'bootstrap':
                print("[pyjack.observable.confidence_interval] Warning: percentile intervals require method='bootstrap'. Using method='normal' instead.")
                return self.confidence_interval(level=level, method='normal')
            lower, upper = numpy.quantile(self.jack_samples, [alpha, 1.0 - alpha], axis=0)
            return lower, upper

        if method == 'bca':
            return self.compute_bca_interval(level=level)

        raise ValueError(f"[pyjack.observable.confidence_interval] Unknown interval method '{method}'.")


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
        if self.data is None:
            if getattr(self, 'resampling_method', 'jackknife')  == 'bootstrap':
                raise ValueError('[pyjack.observable.plot_autocorrelation] Bootstrap-derived observables do not retain reconstructible time-series data.')
            self.data = self.data_from_jack()
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

    @property
    def shape(self):
        try:
            return self.jack_samples.shape[1:]
        except:
            return self.mean.shape
    
    def __len__(self):
        return self.jack_samples.shape[1]
    
    def save(self, filename):
        save(filename, self)
    
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
        if getattr(self, 'resampling_method', 'jackknife') == 'bootstrap':
            self.create(increased_data, method='bootstrap', n_resamples=self.N)
        else:
            self.create(increased_data)

    def __repr__(self):
        if self.mean is not None and self.err is not None:
            if numpy.ndim(self.mean) == 0:
                mean_flat = [numpy.asarray(self.mean).item()]
                err_flat = [numpy.asarray(self.err).item()]
            else:
                mean_flat = self.mean.flatten()
                err_flat = self.err.flatten()
            formatted = ', '.join(pretty_print(x, dx) for x, dx in zip(mean_flat, err_flat))
            return f'pyjack({formatted}, description={self.description})'
        return f'pyjack(mean={self.mean}, err={self.err}, description={self.description})'

    def _new(self, new_jack_samples):
        new_obs = observable(description=self.description, label=self.label)
        if getattr(self, 'resampling_method', 'jackknife') == 'bootstrap':
            new_obs.create_from_bootstrap_samples(new_jack_samples)
        else:
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
            if getattr(self, 'resampling_method', 'jackknife') == 'bootstrap':
                new_obs.create_from_bootstrap_samples(new_jack_samples)
            else:
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
                if getattr(self, 'resampling_method', 'jackknife') == 'bootstrap':
                    self.compute_stats_from_bootstrap_samples(self.jack_samples)
                else:
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
