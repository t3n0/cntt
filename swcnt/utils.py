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
import warnings


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


def bzCuts(k1, k2, N, ksteps):
    kmesh = np.linspace(-0.5, 0.5, ksteps)
    k1grid = np.outer(kmesh, k1)
    cuts = []
    for mu in range(0, N):
        cut = k1grid + mu * k2
        cuts.append(cut)
    return np.array(cuts)


def grapheneTBBands(k, a1, a2, gamma=3.0):
    band = gamma * np.sqrt(3 + 2 * np.cos(np.dot(k, a1)) + 2 * np.cos(np.dot(k, a2)) + 2 * np.cos(np.dot(k, (a2 - a1))))
    return band


def tightBindingElectronBands(cnt, name, sym='hel', gamma=3.0, fermi=0.0):
    attrCuts = f'bzCuts{sym.capitalize()}'
    attrNorm = f'norm{sym.capitalize()}'
    attrBands = f'electronBands{sym.capitalize()}'
    if hasattr(cnt, attrCuts):
        bzCuts = getattr(cnt, attrCuts)
        bzNorm = getattr(cnt, attrNorm)
        subN, ksteps, _ = bzCuts.shape
        bz = np.linspace(-0.5, 0.5, ksteps) * bzNorm
        bands = np.zeros( (subN, 2, 2, ksteps) ) # bands = E_n^mu(k), bands[mu index, n index, k or energy index, grid index]
        bands[:,:,0,:] = bz
        for mu, cut in enumerate(bzCuts):
            upperBand =   grapheneTBBands(cut, cnt.a1, cnt.a2, gamma) - fermi
            lowerBand = - grapheneTBBands(cut, cnt.a1, cnt.a2, gamma) - fermi
            bands[mu, 0, 1, :] = lowerBand
            bands[mu, 1, 1, :] = upperBand
        getattr(cnt, attrBands)[name] = bands
    else:
        print(f'Cutlines "{sym}" not defined.')


def valeCondBands(bands):
    condBands = []
    valeBands = []
    for band in bands:
        cond = np.where(band[1]>0, band, np.nan)
        vale = np.where(band[1]<0, band, np.nan)
        if not np.all(np.isnan(cond)): # skip if there are no states at this energy
            condBands.append( cond )
        if not np.all(np.isnan(vale)): # skip if there are no states at this energy
            valeBands.append( vale )
    condBands = np.array(condBands)
    valeBands = np.array(valeBands)
    return valeBands, condBands


def effectiveMassExcitonBands(cnt, name, which, deltaK = 10.0, bindEnergy = 0.0):
    '''
    Calculates the exciton energy dispersion in the effective mass approximation.
    '''
    
    condValleys = cnt.condKpointValleys[which]
    condInvMasses = cnt.condInvMasses[which]
    condEnergyZeros = cnt.condEnergyZeros[which]
    condKpointZeros = cnt.condKpointZeros[which]

    valeValleys = cnt.valeKpointValleys[which]
    valeInvMasses = cnt.valeInvMasses[which]
    valeEnergyZeros = cnt.valeEnergyZeros[which]
    valeKpointZeros = cnt.valeKpointZeros[which]

    counter = 0
    excBands = {}

    kSteps = int(deltaK / cnt.normLin * cnt.kStepsLin)

    for mu in range(len(condInvMasses)):                # cut index
        for n in range(len(condInvMasses[mu])):         # band index
            for i in range(len(condInvMasses[mu][n])):  # valley index
                condInvMass = abs( condInvMasses[mu][n][i] )
                condEnergy = condEnergyZeros[mu][n][i]
                condKpoint = condKpointZeros[mu][n][i]
                condValley = condValleys[mu][n][i]
                for nu in range(len(valeInvMasses)):                # cut index
                    for m in range(len(valeInvMasses[nu])):         # band index
                        for j in range(len(valeInvMasses[nu][m])):  # valley index
                            valeInvMass = abs( valeInvMasses[nu][m][j] )
                            valeEnergy = valeEnergyZeros[nu][m][j]
                            valeKpoint = valeKpointZeros[nu][m][j]
                            valeValley = valeValleys[nu][m][j]

                            counter += 1

                            print(counter, mu,n,i,nu,m,j, condEnergy, valeEnergy)

                            deltaNorm = minimumPbcNorm(condValley - valeValley, cnt.k1H, cnt.k2H)
                            kpoint = (condKpoint - valeKpoint + cnt.normHel / 2) % cnt.normHel - cnt.normHel / 2
                            invMass = condInvMass * valeInvMass / (condInvMass + valeInvMass)
                            energy = condEnergy - valeEnergy - bindEnergy

                            if deltaNorm < 1e-4:
                                # parallel excitons
                                excBands[f"para.{mu}.{n}.{i}.{nu}.{m}.{j}"] = excitonBandDispersion(kpoint, invMass, energy, deltaK, kSteps)
                            elif 0.6*cnt.normKC < deltaNorm < 1.4*cnt.normKC:
                                # perpendicular excitons
                                excBands[f"perp.{mu}.{n}.{i}.{nu}.{m}.{j}"] = excitonBandDispersion(kpoint, invMass, energy, deltaK, kSteps)
                            else:
                                # dark excitons
                                excBands[f"dark.{mu}.{n}.{i}.{nu}.{m}.{j}"] = excitonBandDispersion(kpoint, invMass, energy, deltaK, kSteps)
    cnt.excitonBands[name] = excBands



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


def excitonBandDispersion(helPos, invMass, energy, deltak, kstep):
    kmesh = np.linspace(-0.5*deltak, 0.5*deltak, kstep)
    band = 0.5 * invMass * kmesh ** 2 + energy
    return kmesh-helPos, band


# def opt_mat_elems(k, a1, a2, n, m):
#     N = n ** 2 + n * m + m ** 2
#     elem = (((n - m) * np.cos(np.dot(k, (a2 - a1))) - (2 * n + m) * np.cos(np.dot(k, a1)) + (n + 2 * m) * np.cos(np.dot(k, a2))) / 2 / np.sqrt(N) / _bands(k, a1, a2))
#     return elem


