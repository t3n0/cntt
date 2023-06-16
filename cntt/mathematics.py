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

Copyright (c) 2021-2023 Stefano Dal Forno.

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


import warnings
import numpy as np


def fourierInterpolation(x, y, newx):
    '''
    Returns the Fourier interpolation of a function y = f(x).
    Note! The function is assumed to be periodic.
    This means, the list of x values must NOT include the last point:
    e.g. use x = np.linspace(minx, maxx, lenx, endpoint = False) !!
    
    Parameters:
    -----------
        x: float
            1D-array, x domain
        
        y: float
            1D-array, values of function f(x)
        
        newx: float
            1D-array, new (finer) x domain over which
            to compute the new values of the function
    
    Returns:
    --------
        newy: float
            1D-array with values of function f(newx)
    '''
    lenx = len(x)
    lennewx = len(newx)
    lendiff = lennewx - lenx
    if lenx % 2 == 0:
        pads = (int(np.floor(lendiff/2)), int(np.ceil(lendiff/2)))
    elif lenx % 2 == 1:
        pads = (int(np.ceil(lendiff/2)), int(np.floor(lendiff/2)))
    yFT     = np.fft.fft(y)
    yFT_pad = np.pad(np.fft.fftshift(yFT), pads, 'constant', constant_values = (0.0, 0.0))
    yFT_pad = np.fft.ifftshift(yFT_pad)
    newy    = np.fft.ifft(yFT_pad).real
    return lennewx / lenx * newy


def findFunctionExtrema(x, y, which = 'max'):
    '''
    Finds the extrema of a given function y = f(x).
    Cusp points, like the Dirac point, are ignored.

    Parameters:
    -----------
        x: float
            1d array, x values

        y: float
            1d array, y = f(x) values

        which: str (optional)
            type of extrema [ 'max' | 'min' ]
            default = max
    
    Returns:
    --------
        mask: bool
            1d array, boolean mask with True value at the extrema
    '''
    with warnings.catch_warnings(record=True) as caught_warnings:
        prime = np.gradient(y, x)
        threshold = np.max(prime) - np.min(prime)
        if which == 'max':
            mask1 = ((np.roll(prime, 1) > 0) * (prime <= 0)) # derivative change sign, might be maximum
            #mask1 = mask1 & (np.roll(prime, 2) > 0) * (np.roll(prime, -1) < 0) # check also the nearest 2 points
        elif which == 'min':
            mask1 = ((np.roll(prime, 1) < 0) * (prime >= 0)) # derivative change sign, might be minimum
            #mask1 = mask1 & (np.roll(prime, 2) < 0) * (np.roll(prime, -1) > 0) # check also the nearest 2 points
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
        funcList: list
            list of functions [f1, f2, ...]
            where f1 = [x_list, y_list]

        which: str (optional)
            type of extrema [ 'max' | 'min' ]
            default = max
    
    Returns:
    --------
        masks: bool
            2d array, list of boolean masks with True value at the extrema
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
        y: float
            1D array, function
    
    Returns:
    --------
        masks: bool
        1d array, boolean mask with True value at the zeros
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
    

def winding(point, polygon):
    '''
    Returns True is the point is inside the polygon.
    '''
    angle = 0
    for i in range(len(polygon)):
        aux1 = polygon[i]-point
        aux2 = polygon[(i+1) % len(polygon)]-point
        newangle = np.arctan2(aux1[0]*aux2[1]-aux1[1]*aux2[0], aux1[0]*aux2[0]+aux1[1]*aux2[1])
        angle += newangle
    return not -0.1<angle/2/np.pi<0.1
