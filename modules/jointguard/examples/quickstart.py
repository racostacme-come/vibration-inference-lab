"""Score the same marginal predictions with and without dependence."""

import numpy as np

from jointguard import Gaussian, energy_score, expected_scores

truth = Gaussian([0, 0], [[1, 0.9], [0.9, 1]])
independent = Gaussian([0, 0], np.eye(2))
print(expected_scores(independent, truth))
print("Extension risk:", independent.exceedance([-1, 1], threshold=1))
rng = np.random.default_rng(42)
x = rng.standard_normal((128, 2))
print("Ensemble energy score:", energy_score(x, [[0.2, 0.1]]))
