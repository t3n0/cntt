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


def grapheneTBBands(k, a1, a2, gamma=3.0):
    '''
    Usual band structure tight-binding approximation for graphene.

    Parameters:
    -----------
        k (float)       (kx, ky) point or array of kpoints

        a1 (float)      (a1x, a1y) lattice vector

        a2 (float)      (a2x, a2y) lattice vector

        gamma (float)   onsite energy parameter
                        optional, default = 3.0 eV

    Returns:
    --------
        band (float)    band structure at the given kpoint or array of kpoints
    '''
    band = gamma * np.sqrt(3 + 2 * np.cos(np.dot(k, a1)) + 2 * np.cos(np.dot(k, a2)) + 2 * np.cos(np.dot(k, (a2 - a1))) + 1e-6)
    # 1e-6 to avoid sqrt of negative number (maybe some subtraction cancelation is happening)
    return band


def tightBindingElectronBands(cnt, name, sym, gamma=3.0, fermi=0.0):
    '''
    Computes the band structure of the given CNT.
    '''
    attrCuts = f'bzCuts{sym.capitalize()}'
    attrNorm = f'norm{sym.capitalize()}'
    attrBands = f'electronBands{sym.capitalize()}'
    if hasattr(cnt, attrCuts):
        bzCuts = getattr(cnt, attrCuts)
        bzNorm = getattr(cnt, attrNorm)
        subN, ksteps, _ = bzCuts.shape
        bz = np.linspace(-0.5, 0.5, ksteps) * bzNorm
        bands = np.zeros( (subN, 2, 2, ksteps) ) # bands = E_n^mu(k), bands[mu index, n index, k/energy index, grid index]
        bands[:,:,0,:] = bz
        for mu, cut in enumerate(bzCuts):
            upperBand =   grapheneTBBands(cut, cnt.a1, cnt.a2, gamma) - fermi
            lowerBand = - grapheneTBBands(cut, cnt.a1, cnt.a2, gamma) - fermi
            bands[mu, 0, 1, :] = lowerBand
            bands[mu, 1, 1, :] = upperBand
        getattr(cnt, attrBands)[name] = bands
    else:
        print(f'Cutlines "{sym}" not defined.')