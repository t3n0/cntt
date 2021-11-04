"""Pollo."""
import numpy as np


def pollo(k, a1, a2):
    """Pollo."""
    band = np.sqrt(
        3
        + 2 * np.cos(np.dot(k, a1))
        + 2 * np.cos(np.dot(k, a2))
        + 2 * np.cos(np.dot(k, (a2 - a1)))
    )
    return band


print("pol")

k = 10
a1 = 30
a2 = 24

print(pollo(k, a1, a2))
