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
import cntt.mathematics as mathematics


# physical constants
Ha2eV = 27.2113825435   # 1 Ha = 27.2 eV
Ry2eV = 13.6056980659   # 1 Ry = 13.6 eV
Bohr2nm = 0.0529177249  # 1 bohr = 0.053 nm
Angstrom2nm = 0.1       # 1 A = 0.1 nm


def unitFactors(cnt):
    '''
    Provides the conversion (multiplicative) factors for the unit of measures.

    Parameters:
    -----------
        cnt (swcnt):    cnt object

    Returns:
    --------
        eFactor
            (float):    energy conversion factor
        lFactor
            (float):    length conversion factor
        invLFactor
            (float):    inverse length conversion factor
    '''
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
    '''
    Convinence function to convert the values of a set of data to different
    unit of measure.
    '''
    newValues = []
    for arg in args:
        newValues.append(factor * getattr(cnt, arg))
    return newValues


def minVector2AtomUnitCell(l, k, n):
    '''
    Returns the u,v pair of integer number defining the CNT 2-atom unit cell.
    There are infinite choice of u,v (i.e. infinite slanted parallelograms
    of same area).
    We pick the pair with minimum (integer) u such that v is also integer.
    '''
    # seq is [0, 1, -1, 2, -2, 3, -3, ...]
    # maybe there is a smarter python way to do it
    seq = [0]
    for i in range(1, 2 * (n + l + 1)):
        seq.append(seq[-1] + i * (-1) ** (i + 1))
    for u in seq:
        v = 1 / l + k / l * u
        if v.is_integer():
            return u, int(v)


def densityOfStates(bands, energySteps):
    energyMin = np.min(bands[:,1]) - 0.1*abs(np.min(bands[:,1]))
    energyMax = np.max(bands[:,1]) + 0.1*abs(np.max(bands[:,1]))
    energyGrid = np.linspace(energyMin, energyMax, energySteps)
    dos = np.zeros(len(energyGrid))
    for n in range(len(bands)):
        prime = np.gradient(bands[n,1], bands[n,0])
        for i, energy in enumerate(energyGrid):
            maskZeros = mathematics.findFunctionZeros(bands[n,1] - energy)
            primeZeros = prime[maskZeros]
            #if 0.0 in primeZeros:
                #mask = np.where(primeZeros == 0.0)
            mask = np.where(np.abs(primeZeros) < 1e-6)
            # todo add verbosity when we manually rescale the divergencies
            #print('Divide by zero in van hove, set it to 1000, lol')
            # van Hove singularities set to 1000, because, why not!? lol
            primeZeros[mask] = 1/1000
            dos[i] += np.sum( 1 / np.abs(primeZeros) )
    return energyGrid, dos


def bzCuts(k1, k2, N, ksteps, min=-0.5, max=0.5):
    kmesh = np.linspace(min, max, ksteps)
    k1grid = np.outer(kmesh, k1)
    cuts = []
    for mu in range(0, N):
        cut = k1grid + mu * k2
        cuts.append(cut)
    return np.array(cuts)


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


def effectiveMassExcitonBands(cnt, which, name=None, deltaK = 10.0, bindEnergy = 0.0):
    '''
    Calculates the exciton energy dispersion in the effective mass approximation.
    '''
    if name == None:
        name = which
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

                            #print(counter, mu,n,i,nu,m,j, condEnergy, valeEnergy)

                            deltaNorm = mathematics.minimumPbcNorm(condValley - valeValley, cnt.k1H, cnt.k2H)
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


def excitonBandDispersion(helPos, invMass, energy, deltak, kstep):
    kmesh = np.linspace(-0.5*deltak, 0.5*deltak, kstep)
    band = 0.5 * invMass * kmesh ** 2 + energy
    return kmesh-helPos, band


# def opt_mat_elems(k, a1, a2, n, m):
#     N = n ** 2 + n * m + m ** 2
#     elem = (((n - m) * np.cos(np.dot(k, (a2 - a1))) - (2 * n + m) * np.cos(np.dot(k, a1)) + (n + 2 * m) * np.cos(np.dot(k, a2))) / 2 / np.sqrt(N) / _bands(k, a1, a2))
#     return elem
