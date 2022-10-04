'''
swcnt-bands
-----------

swcnt-bands is a software for computing, displaying and manipulating
the physical properties of single-walled carbon nanotubes (SWCNTs).

CONTACTS:
---------

email: tenobaldi@gmail.com
github: https://github.com/t3n0/swcnt-bands

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

import matplotlib.pyplot as plt
import numpy as np
import swcnt.utils as utils
import swcnt.plotting as plotting
import os


class Swcnt(object):
    """
    Base class for computing the physical properties
    of single-walled carbon nanotubes (SWCNTs).
    Properties include:
        - direct and reciprocal lattice structure;
        - electron, phonon and exciton energy dispersions;
        - density of states (DOS) and joint density of states (JDOS);
        - absorption spectra;
        - and more (soon).
    """
    def __init__(self, n, m, a0 = 0.2461):
        '''
        Constructor for the Swcnt class.

        Parameters:
        -----------
            n (int):        carbon nanotube n number

            m (int):        carbon nanotube m number

            a0 (float):     carbon nanotube lattice constant (nm)
                            optional, default = 0.2461 nm
        '''
        # units
        self.unitL = 'nm'
        self.unitE = 'eV'
        self.unitInvL = 'nm-1'

        # graphene constants
        self.a0 = a0 # nm
        self.b0 = 4 * np.pi / np.sqrt(3) / self.a0 # nm-1
        self.ac = self.a0/np.sqrt(3)
        self.bc = self.b0/np.sqrt(3)

        # carbon nanotube parameters
        self.n, self.m = n, m
        self.R = np.gcd(2 * m + n, 2 * n + m)
        self.D = np.gcd(m, n)
        self.N = n ** 2 + n * m + m ** 2
        self.NU = 2 * self.N // self.R
        self.p, self.q = (2 * m + n) // self.R, -(2 * n + m) // self.R

        # graphene lattice vectors
        self.a1 = self.a0 * np.array([np.sqrt(3) / 2, 1 / 2])
        self.a2 = self.a0 * np.array([np.sqrt(3) / 2, -1 / 2])
        self.b1 = self.b0 * np.array([1 / 2, np.sqrt(3) / 2])
        self.b2 = self.b0 * np.array([1 / 2, -np.sqrt(3) / 2])

        # CNT lattice vectors
        self.C = self.n * self.a1 + self.m * self.a2
        self.T = self.p * self.a1 + self.q * self.a2
        self.u1, self.v1 = utils.minVector2AtomUnitCell(self.p, self.q, self.n)
        self.alpha = (self.n * self.R - 2 * self.N * self.u1) / (2 * self.m + self.n)
        self.t1 = self.u1 * self.a1 + self.v1 * self.a2
        self.u2, self.v2 = utils.minVector2AtomUnitCell(self.n // self.D, self.m // self.D, self.p)
        self.beta = (self.D * (2 * self.m + self.n) / self.R / self.n + self.NU * self.u2 / self.n)
        self.t2 = self.u2 * self.a1 + self.v2 * self.a2

        # CNT reciprocal lattice vectors
        self.KC = (-self.q * self.b1 + self.p * self.b2) / self.NU
        self.KT = (self.m * self.b1 - self.n * self.b2) / self.NU
        self.k1L = self.NU * self.KC
        self.k2L = self.KT + self.alpha * self.KC
        self.k1H = self.beta * self.KT + self.D * self.KC
        self.k2H = -self.NU / self.D * self.KT

        # CNT linear and helical BZs
        self.normKC = np.linalg.norm(self.KC)
        self.normLin = np.linalg.norm(self.KT)
        self.normHel = np.linalg.norm(self.k2H)
        self.normOrt = abs(self.beta)/self.D*self.normLin

        # CNT data containers for electron and exciton bands, DOS, JDOS, etc
        self.electronBandsLin = {}
        self.electronBandsHel = {}
        self.condKpointValleys = {}
        self.valeKpointValleys = {}
        self.electronDOS = {}
        self.excitonBands = {}
        self.excitonDOS = {}

    def setUnits(self, energy, length):
        '''
        Set the energy and length unit of measure.
        This is just a wrapper for the output.
        The code performs all calculations in eV and nm.

        Parameters:
        -----------
            energy (str):   energy units, either 'eV', 'Ha', 'Ry'

            length (str):   length units, either 'nm', 'bohr', 'Angstrom'
        '''
        self.unitE = energy
        self.unitL = length
        self.unitInvL = length + '-1'


    def calculateCuttingLines(self, ksteps=51):
        '''
        Calculate the linear and helical cutting lines in the zone-folding scheme.

        Parameters:
        -----------
            ksteps (int):   number of kpoints for the discretisation
                            optional, default = 51
        '''
        kstepsLin = ksteps
        kstepsHel = int(self.normHel / self.normLin * kstepsLin)
        self.bzCutsLin = utils.bzCuts(self.KT, self.KC, self.NU, kstepsLin)
        self.bzCutsHel = utils.bzCuts(self.k2H, self.k1H / self.D, self.D, kstepsHel)


    def calculateElectronBands(self, calc, name, sym, **kwargs):
        '''
        Calculate the electron bands energy dispersion using different methods.

        Parameters:
        -----------
            calc (str):     specify the method for the band calculation
                            can be either 'TB' or 'DFT' (more in the future)

            name (str):     unique name to identify the resulting bands

            sym (str):      linear or helical symmetry, either 'lin' or 'hel'

            **kwargs:       key-value arguments depends on the calculation to
                            be performed.

                            TB calculation:
                                gamma (float):      TB on-site parameter,
                                                    optional, default = 3.0 eV
                                
                                fermi (float):      position of the Fermi energy
                                                    wrt the graphene Fermi level
                                                    optional, default = 0.0 eV
                            
                            DFT calculation:
                                something, not yet defined
        '''
        if calc == 'TB':
            utils.tightBindingElectronBands(self, name, sym, **kwargs)
        elif calc == 'DFT':
            pass
        elif calc == 'something else':
            pass
        else:
            print(f'Calculation {calc} not implemented.')


    def calculateKpointValleys(self, which=None):
        '''
        Calculate the (kx,ky) position of the minima and maxima of the electron band dispersion.

        Note! The K valley extrema are calculated in the helical coordinate system.

        Parameters:
        -----------
            which (str)     name of the helical electron band to use for the calculation
                            optional, default = all
        '''
        if which == None:
            keys = self.electronBandsHel.keys()
        else:
            keys = [which]
        bzCuts = self.bzCutsHel
        for key in keys:
            condXY = []
            valeXY = []
            bands = self.electronBandsHel[key]
            vale, cond = utils.condValeBands(bands)
            _, _, _, condMasks = utils.findFunctionListExtrema(cond, 'min')
            _, _, _, valeMasks = utils.findFunctionListExtrema(vale, 'max')
            for condMask, cut in zip(condMasks, bzCuts):
                condXY.append( cut[condMask] )
            for Mask, cut in zip(condMasks, bzCuts):
                condXY.append( cut[condMask] )




    def calculateExcitonBands(self, calc, name, which, **kwargs):
        '''
        Calculate the exciton bands energy dispersion using different methods.
        
        Note! Excitons are always computed in helical coordinates. This means
        that the underlying electron bands should also have helical symmetry.

        Parameters:
        -----------
            calc (str):     specify the method for the band calculation
                            can be either 'EffMass' or 'something' (more in the future)

            name (str):     unique name to identify the resulting bands

            which (str):    name of the helical electron band to use for the calculation

            **kwargs:       key-value arguments depend on the calculation to
                            be performed.

                            EffMass calculation:
                                deltaK (float):     width of the exciton parabolic bands
                                                    optional, default = 10 nm-1
        '''
        if calc == 'EffMass' or calc == 'EM':
            utils.effectiveMassExcitonBands(self, name, which, **kwargs)
        elif calc == 'DFT':
            pass
        elif calc == 'something else':
            pass
        else:
            print(f'Calculation {calc} not implemented.')


    def calculateDOS(self, which, ensteps=1000):
        '''
        Calcualte the density of states for a given particle energy dispersion.
        By default, the DOS is calculated on the helical symmetry.

        Parameters:
        -----------
            which (str):    particle, either 'el', 'ex', 'ph'

            enesteps (int): energy steps for the discretisation
                            optional, default = 1000
        '''
        pass


    def calculateJDOS(self, which, ensteps=1000):
        '''
        Calcualte the joint density of states for a given particle energy dispersion.
        By default, the JDOS is calculated on the helical symmetry.

        Parameters:
        -----------
            which (str):    particle, either 'el', 'ex', 'ph'

            enesteps (int): energy steps for the discretisation
                            optional, default = 1000
        '''
        pass


    def _calculateExcitonBands(self, bindEnergy = 0.05, deltak = 10.0, kstep = 20):
        # CNT band minima, energies and masses
        self.bandMinHel = []
        self.bandMinXy = []
        self.bandInvMasses = []
        self.bandEnergy = []
        for mu in range(0, self.D):
            band = self.bandHel[mu]
            xMin, yMin, secondMin, mask = utils.findMinima(band[0], band[1])
            if len(xMin) > 0:
                self.bandMinHel.append(xMin)
                self.bandInvMasses.append(secondMin)
                self.bandEnergy.append(yMin)
                self.bandMinXy.append(self.bzCutsHel[mu][mask])
        if len(self.bandMinHel) > 0:
            self.bandMinHel = np.array(self.bandMinHel)
            self.bandInvMasses = np.array(self.bandInvMasses)
            self.bandEnergy = np.array(self.bandEnergy)
            self.bandMinXy = np.array(self.bandMinXy)
        else:
            print('No excitons found.')
            return

        # dic = [bz, band]
        self.excParaBands = {}
        self.excPerpBands = {}
        self.excDarkBands = {}
        nMu, nMin = self.bandMinHel.shape
        for mu in range(nMu):
            for nu in range(nMu):
                for i in range(nMin):
                    for j in range(nMin):
                        deltaNorm = utils.findMinDelta(self.bandMinXy[mu, i] - self.bandMinXy[nu, j], self.k1H, self.k2H)
                        helPos = (self.bandMinHel[mu, i] - self.bandMinHel[nu, j] + self.normHel / 2) % self.normHel - self.normHel / 2
                        invMass = self.bandInvMasses[mu, i] * self.bandInvMasses[nu, j] / (self.bandInvMasses[mu, i] + self.bandInvMasses[nu, j])
                        energy = self.bandEnergy[mu, i] + self.bandEnergy[nu, j] - bindEnergy
                        if deltaNorm < 1e-4:
                            # parallel excitons
                            self.excParaBands[f"{mu}.{i}.{j}"] = utils.excBands(helPos, invMass, energy, deltak, kstep)
                        elif 0.6*self.normKC < deltaNorm < 1.4*self.normKC:
                            # perpendicular excitons
                            self.excPerpBands[f"{mu}.{nu}.{i}.{j}"] = utils.excBands(helPos, invMass, energy, deltak, kstep)
                        else:
                            # dark excitons
                            self.excDarkBands[f"{mu}.{nu}.{i}.{j}"] = utils.excBands(helPos, invMass, energy, deltak, kstep)


    def saveToDirectory(self, dirpath):
        for mu in range(0, self.NU):
            path = os.path.join(dirpath, f"bandLin{mu:03d}.txt")
            utils.save_file(self.bandLin[mu][0], self.bandLin[mu][1], path=path)
        for mu in range(0, self.D):
            path = os.path.join(dirpath, f"bandHel{mu:03d}.txt")
            utils.save_file(self.bandHel[mu][0], self.bandHel[mu][1], path=path)
        for k in self.excParaBands:
            path = os.path.join(dirpath, f"excPara{k}.txt")
            utils.save_file(*self.excParaBands[k], path=path)
        for k in self.excPerpBands:
            path = os.path.join(dirpath, f"excPerp{k}.txt")
            utils.save_file(*self.excPerpBands[k], path=path)
        for k in self.excDarkBands:
            path = os.path.join(dirpath, f"excDark{k}.txt")
            utils.save_file(*self.excDarkBands[k], path=path)


    def plot(self, path=None):
        '''
        Plots all relevant quantities and properties of the nanotube object.

        This is a *convenience* function that displays the carbon nanotube
        parameters, direct and reciprocal lattices, K and K' valley minima,
        linear and helical electron bands, and the helical exciton and
        phonon bands.

        For more versatile plotting routines, the user might want to import
        the `swcnt.plotting` module.

        Parameters:
        -----------
            path (str):     destination path where to save the figure
                            optional, default = shows the figure
        '''
        if path == None:
            fig = plt.figure(figsize=(16, 9))
        else:
            fig = plt.figure(figsize=(16, 9), dpi=300)

        # figure and axes
        fig.suptitle(f"CNT ({self.n},{self.m})")
        ax1 = fig.add_axes([0.23, 0.63, 0.35, 0.32])
        ax2 = fig.add_axes([0.63, 0.63, 0.35, 0.32])
        ax3 = fig.add_axes([0.05, 0.25, 0.2, 0.3])
        ax4 = fig.add_axes([0.25, 0.25, 0.58, 0.3])
        ax5 = fig.add_axes([0.83, 0.25, 0.15, 0.3])
        ax6 = fig.add_axes([0.25, 0.05, 0.58, 0.2])
        #ax7 = fig.add_axes([0.83, 0.05, 0.15, 0.2])

        plotting.dirLat(self, ax1)
        plotting.recLat(self, ax2)
        plotting.electronBands(self, 'lin', ax3)
        plotting.electronBands(self, 'hel', ax4)
        
        ax3.set_title("Linear")
        ax4.set_title("Helical")
        ax3.get_shared_y_axes().join(ax3, ax4) # "Matplotlib fai ribrezzo ai popoli del mondo!"
        ax4.set_yticklabels([])                # "Tu metti in subbuglio il mio sistema di donna sensibile!"
        ax4.set_ylabel('')                     # (Alida Valli)
        ax4.get_shared_x_axes().join(ax4, ax6)

        # # ax6 plot excitons
        # for i,k in enumerate(self.excParaBands):
        #     if i == 0:
        #         label='Para'
        #     else:
        #         label='_'
        #     ax6.plot(*self.excParaBands[k],'r', label=label)
        # for i,k in enumerate(self.excPerpBands):
        #     if i == 0:
        #         label='Perp'
        #     else:
        #         label='_'
        #     ax6.plot(*self.excPerpBands[k],'y', label=label)
        # for i,k in enumerate(self.excDarkBands):
        #     if i == 0:
        #         label='Dark'
        #     else:
        #         label='_'
        #     ax6.plot(*self.excDarkBands[k],'grey', label=label)

        # ax5 ax7 plot DOS and absorption
        # todo

        # -------------------------------------------------
        # labels, ticks, range, grids, legends, texts
        
        
        ax5.set_title("DOS")
        ax6.set_ylabel("Energy (eV)")
        ax6.set_xlabel("k (nm-1)")
        
        ax5.set_yticklabels([])
        ax4.set_xticklabels([])
        ax5.set_xticklabels([])
        #ax7.set_yticklabels([])
        # plt.text(0.05, 0.9, self.textParams(), ha="left", va="top", transform=fig.transFigure)

        # minElEn, maxElEn = 1.1 * np.min(-self.bandHel), 1.1 * np.max(self.bandHel)
        # minBzHel, maxBzHel = 1.1 * np.min(self.bzHel), 1.1 * np.max(self.bzHel)
        # for ax in [ax4, ax6]:
        #     ax.set_xlim(minBzHel, maxBzHel)
        #     ax.locator_params(axis='x', nbins=20)
        # for ax in [ax3, ax4, ax5]:
        #     ax.set_ylim(minElEn, maxElEn)
        #     ax.locator_params(axis='y', nbins=10)
        # for ax in [ax3, ax4, ax5, ax6, ax7]:    
        #     ax.grid(linestyle='--')

        # ax6.vlines(self.normOrt,0,6,linestyles ="dashed", colors ="k")
        # ax6.vlines(-self.normOrt,0,6,linestyles ="dashed", colors ="k")
        # ax6.legend(loc=(-0.2, 0.0))

        # save if path is not None
        if path == None:
            plt.show()
        else:
            fig.savefig(path)
            plt.close()
