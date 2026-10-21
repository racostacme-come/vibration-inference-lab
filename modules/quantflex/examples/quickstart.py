"""Recover a sub-step harmonic and apply the declared energy bias correction."""

import numpy as np

from quantflex import HarmonicModel, elastic_energy, quantize, steady_coefficients

times = np.arange(640) / 40
model = HarmonicModel(times, 2 * np.pi)
beta = steady_coefficients(1, 0.4, 100, 2 * np.pi, 0.012)
signal = model.design @ beta
step = 1e-3
dither = np.random.default_rng(42).uniform(-step / 2, step / 2, signal.shape)
observed = quantize(signal, step, dither=dither)
assert not observed.saturated.any()
estimate = model.fit(observed.values)
covariance = model.covariance(step**2 / 12)
print("Exact [DC, cosine, sine] (m):", beta)
print("Estimated coefficients (m):", estimate)
print("Corrected energy (J):", elastic_energy(estimate, 100, covariance=covariance))
print("One realization is uncertain; unbiasedness describes repeated experiments.")
