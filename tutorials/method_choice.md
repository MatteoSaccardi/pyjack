# Choosing Jackknife or Bootstrap

This note is a short conceptual guide for choosing the resampling method and
interval type in `pyjack`.

It is not meant as a full derivation. The goal is practical guidance for
everyday use.

## When jackknife is a good default

Use jackknife when:

- your estimator is reasonably smooth as a function of the data
- you mainly care about mean, error, covariance, and fits
- you want something cheaper than a large bootstrap run

**In this library, jackknife remains the default.**

## When bootstrap is worth using

Use bootstrap when:

- you want interval estimates derived directly from the replica distribution
- the estimator is nonlinear enough that asymmetry may matter
- you want percentile or BCa intervals
- you are willing to pay more computational cost for more flexible uncertainty summaries

In `pyjack`, bootstrap is opt-in with `method='bootstrap'`.

## Why `binsize` matters

Both jackknife and bootstrap operate on the binned data inside `pyjack`.

This matters because Monte Carlo time series are often autocorrelated. If you
resample raw configurations as if they were independent when they are not, the
error estimate can be too optimistic.

The more principled way to choose a sensible `binsize` is to inspect the
autocorrelation of the observable. In `pyjack`, you can access this through
`obs.autocorr_time()` and visualize it with `obs.plot_autocorrelation()`.

Practical rules:

- increase `binsize` until the estimated uncertainties become reasonably stable;
- do not make `binsize` so large that only a handful of bins remain.

There is no universal magic number. The right value depends on the
autocorrelation structure of your data.

## How to choose `n_resamples`

For bootstrap, `n_resamples` controls the number of replica estimates.

Typical practical guidance:

- exploratory work: `500` to `1000`;
- more stable percentile or BCa intervals: `2000` or more.

Larger values reduce Monte Carlo noise in the interval estimates, but they also
cost more time and memory.

## Which interval should I use?

This discussion refers specifically to the function
`obs.confidence_interval(level=..., method=...)`.

That function returns a pair `(low, high)` with the lower and upper confidence
bounds for each component of the observable at the requested confidence level.

### `method='normal'`

This returns a symmetric interval built from the observable mean and standard
error:

```python
mean +/- z(level) * err
```

This is the natural choice for jackknife observables in the current library.
It is also available for bootstrap observables when you want a simple
Gaussian-style summary.

### `method='percentile'`

This is a bootstrap-specific interval. It uses the lower and upper quantiles of
the stored bootstrap replicas.

Use it when you want the interval to come directly from the bootstrap replica
distribution rather than from a symmetric approximation.

### `method='bca'`

This is also a bootstrap-specific interval. BCa stands for *bias-corrected and
accelerated*.

Use it when you want a bootstrap interval that improves on the plain percentile
interval by correcting for:

- bias in the bootstrap distribution;
- skewness or nonlinearity of the estimator.

In this library, this is the most refined interval currently implemented for
bootstrap observables.

For jackknife observables, `percentile` and `bca` are not natural
constructions in this implementation. If you request them,
`pyjack` prints a warning and falls back to `method='normal'`.

## Minimal examples

Jackknife:

```python
obs = pyjack.observable()
obs.create(data)
low, high = obs.confidence_interval(level=0.68, method='normal')
```

Bootstrap:

```python
obs = pyjack.observable()
obs.create(data, method='bootstrap', binsize=5, n_resamples=2000, seed=42)
low, high = obs.confidence_interval(level=0.68, method='bca')
```

## Practical recommendations

If you are unsure:

- choose one method (jackknife is simpler, bootstrap is more flexible and general)
- inspect the effect of binning
- switch to bootstrap when you specifically need distribution-based intervals
- use BCa when the bootstrap interval looks asymmetric or when you want the
  most careful interval currently implemented here

Use the dedicated workflow notebooks for runnable examples:

- `tutorial_jackknife.ipynb`
- `tutorial_bootstrap.ipynb`
