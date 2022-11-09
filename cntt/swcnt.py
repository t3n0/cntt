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


import matplotlib.pyplot as plt
import numpy as np
import os

import cntt.utils as utils
import cntt.plotting as plotting
import cntt.physics as physics
import cntt.tightbinding as tightbinding
import cntt.dft as dft
import cntt.mathematics as mathematics



class Swcnt(object):
    """
    Base class for computing the physical properties
    of single-walled carbon nanotubes (SWCNTs).
    Properties include:
        - direct and reciprocal lattice structure;
        - electron and exciton energy dispersions;
        - density of states (DOS) and joint density of states (JDOS);
        - absorption spectra (soon);
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
        self.cellVolume = self.a0**2 * np.sqrt(3) / 2
        self.bzVolume   = self.b0**2 * np.sqrt(3) / 2

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
        self.u1, self.v1 = physics.minVector2AtomUnitCell(self.p, self.q, self.n)
        self.alpha = (self.n * self.R - 2 * self.N * self.u1) / (2 * self.m + self.n)
        self.t1 = self.u1 * self.a1 + self.v1 * self.a2
        self.u2, self.v2 = physics.minVector2AtomUnitCell(self.n // self.D, self.m // self.D, self.p)
        self.beta = (self.D * (2 * self.m + self.n) / self.R / self.n + self.NU * self.u2 / self.n)
        self.t2 = self.u2 * self.a1 + self.v2 * self.a2
        self.atomA = 1/3*(self.a1 + self.a2)
        self.atomB = 2/3*(self.a1 + self.a2)
        self.atomAlin = 1/3*(self.T + self.t1)
        self.atomBlin = 2/3*(self.T + self.t1)
        self.atomAhel = 1/3*(self.C/self.D + self.t2)
        self.atomBhel = 2/3*(self.C/self.D + self.t2)

        # CNT crystallographic constants
        self.normT  = np.linalg.norm(self.T)
        self.normCD = np.linalg.norm(self.C)/self.D
        self.normt1 = np.linalg.norm(self.t1)
        self.normt2 = np.linalg.norm(self.t2)
        self.cosTt1 = np.dot(self.T, self.t1)/self.normT/self.normt1
        self.cosCt2 = np.dot(self.C/self.D, self.t2)/self.normCD/self.normt2

        # CNT reciprocal lattice vectors
        self.KC = (-self.q * self.b1 + self.p * self.b2) / self.NU
        self.KT = (self.m * self.b1 - self.n * self.b2) / self.NU
        self.k1L = self.NU * self.KC
        self.k2L = self.KT + self.alpha * self.KC
        self.k1H = self.beta * self.KT + self.D * self.KC
        self.k2H = -self.NU / self.D * self.KT

        # CNT linear and helical BZs
        self.normKT = np.linalg.norm(self.KT)
        self.normKC = np.linalg.norm(self.KC)
        self.normLin = np.linalg.norm(self.KT)
        self.normHel = np.linalg.norm(self.k2H)
        self.normOrt = abs(self.beta)/self.D*self.normLin

        # CNT data containers for electron and exciton bands, DOS, JDOS, etc
        self.electronBandsLin = {}
        self.electronBandsHel = {}

        self.condKpointValleys = {}
        self.condEnergyZeros = {}
        self.condKpointZeros = {}
        self.condInvMasses = {}

        self.valeKpointValleys = {}
        self.valeEnergyZeros = {}
        self.valeKpointZeros = {}
        self.valeInvMasses = {}

        self.excitonBands = {}
        self.excitonDOS = {}
        self.electronDOS = {}

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
        self.kStepsLin = ksteps
        self.kStepsHel = int(self.normHel / self.normLin * self.kStepsLin)
        self.bzCutsLin = physics.bzCuts(self.KT, self.KC, self.NU, self.kStepsLin)
        self.bzCutsHel = physics.bzCuts(self.k2H, self.k1H / self.D, self.D, self.kStepsHel)


    def calculateElectronBands(self, calc, name, sym='hel', **kwargs):
        '''
        Calculate the electron bands energy dispersion using different methods.

        Parameters:
        -----------
            calc (str):     specify the method for the band calculation
                            can be either 'TB' or 'DFT' (more in the future)

            name (str):     unique name to identify the resulting bands

            sym (str):      linear or helical symmetry, either 'lin' or 'hel'
                            optional, default = 'hel'

            **kwargs:       Optional key-value arguments.
                            They depends on the calculation to be performed.

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
            tightbinding.tightBindingElectronBands(self, name, sym, **kwargs)
        elif calc == 'DFT':
            dft.dftElectronBands(self, name, sym, **kwargs)
        elif calc == 'something else':
            pass
        else:
            print(f'Calculation {calc} not implemented.')


    def calculateKpointValleys(self, which='all'):
        '''
        Calculate the (kx,ky) position of the minima and maxima of the electron band dispersion.

        Note! The K valley extrema are calculated in the helical coordinate system.

        Parameters:
        -----------
            which (str)     name of the helical electron band to use for the calculation
                            optional, defalut = all
        '''
        if which == 'all':
            keys = self.electronBandsHel.keys()
        else:
            keys = [which]
        
        for which in keys:
            bzCuts = self.bzCutsHel
            bands = self.electronBandsHel[which]
            subN, _, _, _ = bands.shape

            self.condKpointValleys[which] = []
            self.condInvMasses[which] = []
            self.condEnergyZeros[which] = []
            self.condKpointZeros[which] = []

            self.valeKpointValleys[which] = []
            self.valeInvMasses[which] = []
            self.valeEnergyZeros[which] = []
            self.valeKpointZeros[which] = []

            for mu in range(0, subN): # angular momentum
                vales, conds = physics.valeCondBands(bands[mu])
                condMasks = mathematics.findFunctionListExtrema(conds, 'min')
                valeMasks = mathematics.findFunctionListExtrema(vales, 'max')
                cuts = bzCuts[mu]
                
                # valence band properties
                energyZeros = []
                kpointZeros = []
                invMasses = []
                kValley = []
                for vale, mask in zip(vales, valeMasks):
                    prime = np.gradient(vale[1], vale[0])
                    second = np.gradient(prime, vale[0])
                    energyZeros.append( vale[1][mask] )
                    kpointZeros.append( vale[0][mask] )
                    invMasses.append( second[mask] )
                    kValley.append( cuts[mask] )
                self.valeKpointValleys[which].append( kValley )
                self.valeEnergyZeros[which].append( energyZeros )
                self.valeKpointZeros[which].append( kpointZeros )
                self.valeInvMasses[which].append( invMasses )
                
                # conduction band properties
                energyZeros = []
                kpointZeros = []
                invMasses = []
                kValley = []
                for cond, mask in zip(conds, condMasks):
                    prime = np.gradient(cond[1], cond[0])
                    second = np.gradient(prime, cond[0])
                    energyZeros.append( cond[1][mask] )
                    kpointZeros.append( cond[0][mask] )
                    invMasses.append( second[mask] )
                    kValley.append( cuts[mask] )
                self.condKpointValleys[which].append( kValley )
                self.condEnergyZeros[which].append( energyZeros )
                self.condKpointZeros[which].append( kpointZeros )
                self.condInvMasses[which].append( invMasses )


    def calculateExcitonBands(self, calc, which, name=None, **kwargs):
        '''
        Calculate the exciton bands energy dispersion using different methods.
        
        Note! Excitons are always computed in helical coordinates. This means
        that the underlying electron bands should also have helical symmetry.

        Parameters:
        -----------
            calc (str):     specify the method for the band calculation
                            can be either 'EffMass' or 'something else' (more in the future)

            which (str):    name of the helical electron band to use for the calculation

            name (str):     unique name to identify the resulting bands
                            optional, default = same as which

            **kwargs:       key-value arguments depend on the calculation to
                            be performed.

                            EffMass calculation:
                                deltaK (float):     width of the exciton parabolic bands
                                                    optional, default = 10.0 nm-1
                                
                                bindEnergy
                                        (float):    binding energy between electron and
                                                    hole states
                                                    optional, default = 0.0 eV
        '''
        if calc == 'EffMass' or calc == 'EM':
            physics.effectiveMassExcitonBands(self, which, name, **kwargs)
        elif calc == 'DFT':
            pass
        elif calc == 'something else':
            pass
        else:
            print(f'Calculation {calc} not implemented.')


    def calculateDOS(self, which, name = 'all', enSteps=1000):
        '''
        Calcualte the density of states for a given particle energy dispersion.
        By default, the DOS is calculated on the helical symmetry.

        Parameters:
        -----------
            which (str):    particle, either 'electron', 'exciton', 'phonon'

            name (str):     name of the specific band to use for DOS
                            optional, default = all

            enSteps (int):  energy steps for the discretisation
                            optional, default = 1000
        '''
        if which == 'el' or which == 'electron':
            if name == 'all':
                bandNames = self.electronBandsHel.keys()
            else:
                bandNames = [name]
            for name in bandNames:
                bands = self.electronBandsHel[name]
                cutIdx, bandIdx, axIdx, gridIdx = bands.shape
                en, dos = physics.densityOfStates(bands.reshape((cutIdx * bandIdx, axIdx, gridIdx)), enSteps)
                self.electronDOS[name] = [en, dos/self.normHel]
        elif which == 'ex' or which == 'exciton':
            if name == 'all':
                bandNames = self.excitonBands.keys()
            else:
                bandNames = [name]
            for name in bandNames:
                bands = self.excitonBands[name]
                allBands = []
                for key in bands:
                    allBands.append(bands[key])
                allBands = np.array(allBands)
                en, dos = physics.densityOfStates(allBands, enSteps)
                self.excitonDOS[name] = [en, dos]
        else:
            print(f'Particle {which} not recognized.')
        


    def calculateJDOS(self, which, enSteps=1000):
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
        the `cntt.plotting` module.

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
        ax7 = fig.add_axes([0.83, 0.05, 0.15, 0.2])

        # plot all axes
        plotting.dirLat(self, ax1)
        plotting.recLat(self, ax2)
        plotting.electronBands(self, ax3, 'lin')
        plotting.electronBands(self, ax4, 'hel')
        plotting.excitonBands(self, ax6)
        plotting.electronDOS(self, ax5, True)
        plotting.excitonDOS(self, ax7, True)

        # format some text and layout
        ax3.set_title("Linear")
        ax4.set_title("Helical")
        ax5.set_title("DOS")
        for ax in [ax4, ax5, ax7]:
            ax.set_ylabel('')
        for ax in [ax4, ax5]:
            ax.set_xlabel('')

        # "Matplotlib fai ribrezzo ai popoli del mondo!"
        # "Tu metti in subbuglio il mio sistema di donna sensibile!"
        # (Alida Valli)

        ax3.get_shared_y_axes().join(ax3, ax4, ax5)
        ax6.get_shared_y_axes().join(ax6, ax7)                                   
        ax4.get_shared_x_axes().join(ax4, ax6)
        
        ax4.set_yticklabels([]) 
        ax5.set_yticklabels([])
        ax4.set_xticklabels([])
        ax5.set_xticklabels([])
        ax7.set_yticklabels([])

        # cnt parameters text
        plt.text(0.05, 0.9, utils.textParams(self), ha="left", va="top", transform=fig.transFigure)

        # ticks locators and grids
        for ax in [ax4, ax6]:
            ax.locator_params(axis='x', nbins=10)
        for ax in [ax3, ax4, ax5]:
            ax.locator_params(axis='y', nbins=6)
        for ax in [ax6, ax7]:
            ax.locator_params(axis='y', nbins=6)
        for ax in [ax3, ax4, ax5, ax6, ax7]:    
            ax.grid(linestyle='--')


        # save if path is not None
        if path == None:
            plt.show()
        else:
            fig.savefig(path)
            plt.close()
