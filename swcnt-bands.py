#
# Copyright (c) 2021 Stefano Dal Forno.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
