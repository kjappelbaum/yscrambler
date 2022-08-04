# -*- coding: utf-8 -*-
from collections import defaultdict

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

"""Permutation test of the null hypothesis that there is no relationship between features and labels"""


def predict_yscramble(model, X, y, repeats, scorers=(("mse", mean_squared_error))):
    """Bootstrap resample for `repeats` times, then train model and predict, then shuffle y, train model and predict"""
    errors_unshuffled = defaultdict(list)
    errors_shuffled = defaultdict(list)

    for _ in range(repeats):
        X_resampled, y_resampled = resample(X, y)
        model.fit(X_resampled, y_resampled)
        for name, scorer in scorers:
            predictions = model.predict(X)
            errors_unshuffled[name].append(scorer(y, predictions))
        y_shuffled = y[:]
        np.random.shuffle(y_shuffled)
        model.fit(X, y_shuffled)
        for name, scorer in scorers:
            predictions = model.predict(X)
            errors_shuffled[name].append(scorer(y, predictions))

    return errors_unshuffled, errors_shuffled
