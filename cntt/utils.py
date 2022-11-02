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


import argparse
import numpy as np
from cntt.physics import unitFactors


def textParams(cnt):
    '''
    Returns a text with the main physical properties and parameters
    of a given carbon nanotube.

    Parameters:
    -----------
        cnt (swcnt):    cnt object

    Returns:
    --------
        text (str):     physical properties and parameters
    '''
    eFactor, lFactor, invLFactor = unitFactors(cnt)
    text = (
        f"n, m = {cnt.n},{cnt.m}\n"
        f"Diameter = {lFactor * np.linalg.norm(cnt.C)/np.pi:.2f} {cnt.unitL}\n"
        f"C = {cnt.n:+d} a1 {cnt.m:+d} a2\n"
        f"T = {cnt.p:+d} a1 {cnt.q:+d} a2\n"
        f"t1 = {cnt.u1:+d} a1 {cnt.v1:+d} a2\n"
        f"t2 = {cnt.u2:+d} a1 {cnt.v2:+d} a2\n"
        f"NU = {cnt.NU}\n"
        f"D = {cnt.D}\n"
        f"BZ_lin = {invLFactor * cnt.normLin:.2f} {cnt.unitInvL}\n"
        f"BZ_hel = {invLFactor * cnt.normHel:.2f} {cnt.unitInvL}\n"
        f"K_ort = {invLFactor * cnt.normOrt:.2f} {cnt.unitInvL}"
    )
    return text


def save_file(*args, path, header=""):
    data = []
    for arg in args:
        if (arg.dtype == float) or (arg.dtype == int):
            data.append(arg)
        elif arg.dtype == complex:
            data.append(arg.real)
            data.append(arg.imag)
    data = np.array(data)
    np.savetxt(path, data.T, header=header)


def getArgs():
    parser = argparse.ArgumentParser(description="Calculate the direct space, reciprocal space, electron and exciton band structure of a given (n,m) CNT")
    # group = parser.add_mutually_exclusive_group()
    # group.add_argument("-v", "--verbose", action="store_true")
    # group.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("n", help="(n,m) carbon nanotube n paramenter", type=int)
    parser.add_argument("m", help="(n,m) carbon nanotube m paramenter", type=int)
    parser.add_argument("-o", "--outdir", help="output destination folder")
    args = parser.parse_args()
    return args
