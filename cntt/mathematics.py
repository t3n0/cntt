'''
CNTT / C-entity
-----------

CNTT, also spelled C-entity, is a software for computing, displaying and
manipulating the physical properties of single-walled carbon nanotubes (SWCNTs).

CONTACTS:
---------

email: tenobaldi@gmail.com
github: https://github.com/t3n0/cntt

LICENSE:
--------

Copyright (c) 2021-2022 Stefano Dal Forno.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''


import numpy as np
import warnings


def findFunctionExtrema(x, y, which = 'max'):
    '''
    Finds the extrema of a given function y = f(x).
    Cusp points, like the Dirac point, are ignored.

    Parameters:
    -----------
        x (float)       array, x values

        y (float)       array, y = f(x) values

        which (str)     'max' or 'min'
    
    Returns:
    --------
        mask (bool)     array, boolean mask with True value at the extrema
    '''
    with warnings.catch_warnings(record=True) as caught_warnings:
        prime = np.gradient(y, x)
        threshold = np.max(prime) - np.min(prime)
        if which == 'max':
            mask1 = (np.roll(prime, 1) > 0) * (prime < 0) # derivative change sign, might be maximum
        elif which == 'min':
            mask1 = (np.roll(prime, 1) < 0) * (prime > 0) # derivative change sign, might be minimum
        mask2 = np.abs(np.roll(prime, 1) - prime) > 0.3*threshold # delta derivative too big, might be dirac point
        mask = mask1 & (~ mask2) # tilde is negation, not mask2
        # print(caught_warnings) # when masks with np.nan are generated, there can be some RuntimeWarnings
        return mask


def findFunctionListExtrema(funcList, which='max'):
    '''
    Finds the extrema of a list of functions y = f(x).
    Cusp points, like the Dirac point, are ignored.

    Partameters
    -----------
        funcList
            (array)     list of functions [[x1, y1], [x2, y2], ...]

        which (str)     'max' or 'min'
    
    Returns:
    --------
        masks (bool)    array, list of boolean masks with True value at the extrema
    '''
    masks = []
    for func in funcList:
        mask = findFunctionExtrema(func[0], func[1], which)
        masks.append(mask)
    return masks


def findFunctionZeros(y):
    '''
    Returns a boolean array mask corresponding to the zeros of the given function.

    Partameters
    -----------
        y (float)       1D array, function
    
    Returns:
    --------
        masks (bool)    array, boolean mask with True value at the zeros
    '''
    mask = (np.roll(y, 1) * y) < 0
    return mask


def minimumPbcNorm(vec, k1, k2):
    '''
    Returns the minimum norm of vector 'vec' in a PBC lattice
    defined by the base vectors k1 and k2
    '''
    norm = np.linalg.norm(vec)
    for n in range(-1,2):
        for m in range(-1,2):
            newvec = vec + n*k1 + m*k2
            newnorm = np.linalg.norm(newvec)
            if newnorm < norm:
                norm = newnorm
    return norm