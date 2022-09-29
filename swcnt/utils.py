#
# Copyright (c) 2021-2022 Stefano Dal Forno.
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

import argparse
import numpy as np


# physical constants
Ha2eV = 27.2113825435 # 1 Ha = 27.2 eV
Ry2eV = 13.6056980659 # 1 Ry = 13.6 eV
Bohr2nm = 0.0529177249 # 1 bohr = 0.053 nm
Angstrom2nm = 0.1  # 1 A = 0.1 nm


def unitFactors(cnt):
    if cnt.unitE == 'eV': eFactor = 1
    if cnt.unitE == 'Ha': eFactor = 1/Ha2eV
    if cnt.unitE == 'Ry': eFactor = 1/Ry2eV
    if cnt.unitL == 'nm':
        lFactor = 1
        invLFactor = 1
    if cnt.unitL == 'bohr':
        lFactor = 1/Bohr2nm
        invLFactor = Bohr2nm
    if cnt.unitL == 'angstrom':
        lFactor = 1/Angstrom2nm
        invLFactor = Angstrom2nm
    return eFactor, lFactor, invLFactor


def changeUnits(cnt, factor, *args):
    newValues = []
    for arg in args:
        newValues.append(factor * getattr(cnt, arg))
    return newValues


def textParams(cnt):
    text = (
        f"n, m = {cnt.n},{cnt.m}\n"
        f"Diameter = {np.linalg.norm(cnt.C)/np.pi:.2f} nm\n"
        f"C = {cnt.n:+d} a1 {cnt.m:+d} a2\n"
        f"T = {cnt.p:+d} a1 {cnt.q:+d} a2\n"
        f"t1 = {cnt.u1:+d} a1 {cnt.v1:+d} a2\n"
        f"t2 = {cnt.u2:+d} a1 {cnt.v2:+d} a2\n"
        f"NU = {cnt.NU}\n"
        f"D = {cnt.D}\n"
        f"BZ_lin = {cnt.normLin:.2f} nm-1\n"
        f"BZ_hel = {cnt.normHel:.2f} nm-1\n"
        f"K_ort = {cnt.normOrt:.2f} nm-1"
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
    parser = argparse.ArgumentParser(description="Calculate the direct space, reciprocal space and band structure of a given (n,m) CNT")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true")
    group.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("n", help="(n,m) carbon nanotube n paramenter", type=int)
    parser.add_argument("m", help="(n,m) carbon nanotube m paramenter", type=int)
    parser.add_argument("-o", "--outdir", help="output destination folder")
    args = parser.parse_args()
    return args


def minVector2AtomUnitCell(l, k, n):
    # seq is [0, 1, -1, 2, -2, 3, -3, ...]
    # maybe there is a smarter python way to do it
    seq = [0]
    for i in range(1, 2 * (n + l + 1)):
        seq.append(seq[-1] + i * (-1) ** (i + 1))
    for u in seq:
        v = 1 / l + k / l * u
        if v.is_integer():
            return u, int(v)


def findMinima(x,y):
    prime = np.gradient(y, x)
    second = np.gradient(prime, x)
    # todo avoid dirac points in metallic cnts
    threshold = np.max(prime) - np.min(prime)
    mask1 = (np.roll(prime, 1) < 0) * (prime > 0) # derivative change sign, might be minimum
    mask2 = np.abs(np.roll(prime, 1) - prime) > 0.3*threshold # delta derivative too big, might be dirac point
    mask = mask1 & (~ mask2) # tilde is negation, not mask2
    xMin = x[mask]
    yMin = y[mask]
    secondMin = second[mask]
    return xMin, yMin, secondMin, mask


def bzCuts(k1, k2, N, ksteps):
    kmesh = np.linspace(-0.5, 0.5, ksteps)
    k1grid = np.outer(kmesh, k1)
    cuts = []
    for mu in range(0, N):
        cut = k1grid + mu * k2
        cuts.append(cut)
    return np.array(cuts)


def grapheneTBBands(cuts, a1, a2, gamma=3.0, fermi=0.0):
    bands = []
    for cut in cuts:
        band = gamma * np.sqrt(3 + 2 * np.cos(np.dot(cut, a1)) + 2 * np.cos(np.dot(cut, a2)) + 2 * np.cos(np.dot(cut, (a2 - a1))))
        bands.append(band - fermi)
        bands.append(-band - fermi)
    return np.array(bands)


def tightBindingElectronBands(cnt, name, sym='hel', gamma=3.0, fermi=0.0):
    attrCuts = f'bzCuts{sym.capitalize()}'
    attrNorm = f'norm{sym.capitalize()}'
    attrBands = f'electronBands{sym.capitalize()}'
    if hasattr(cnt, attrCuts):
        bzCuts = getattr(cnt, attrCuts)
        bzNorm = getattr(cnt, attrNorm)
        subN, ksteps, _ = bzCuts.shape
        bz = np.linspace(-0.5, 0.5, ksteps) * bzNorm
        bands = grapheneTBBands(bzCuts, cnt.a1, cnt.a2, gamma, fermi)
        data = np.zeros((2*subN, 2, ksteps))
        data[:,0,:] = bz
        data[:,1,:] = bands
        getattr(cnt, attrBands)[name] = data
    else:
        print(f'Cutlines "{sym}" not defined.')


def effectiveMassExcitonBands(cnt, name, deltaK=10.0):
    pass

def findMinDelta(vec, k1, k2):
    norm = np.linalg.norm(vec)
    for n in range(-1,2):
        for m in range(-1,2):
            newvec = vec + n*k1 + m*k2
            newnorm = np.linalg.norm(newvec)
            if newnorm < norm:
                norm = newnorm
    return norm


def excBands(helPos, invMass, energy, deltak, kstep):
    kmesh = np.linspace(-0.5*deltak, 0.5*deltak, kstep)
    band = 0.5 * invMass * kmesh ** 2 + energy
    return kmesh-helPos, band


def opt_mat_elems(k, a1, a2, n, m):
    N = n ** 2 + n * m + m ** 2
    elem = (((n - m) * np.cos(np.dot(k, (a2 - a1))) - (2 * n + m) * np.cos(np.dot(k, a1)) + (n + 2 * m) * np.cos(np.dot(k, a2))) / 2 / np.sqrt(N) / _bands(k, a1, a2))
    return elem


def _bands(k, a1, a2):
    band = np.sqrt(3 + 2 * np.cos(np.dot(k, a1)) + 2 * np.cos(np.dot(k, a2)) + 2 * np.cos(np.dot(k, (a2 - a1))))
    return band