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


import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
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
            n: int
                carbon nanotube n number

            m: int
                carbon nanotube m number

            a0: float (optional)
                carbon nanotube lattice constant (nm)
                default = 0.2461 nm
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
        #self.N = n ** 2 + n * m + m ** 2
        self.N = 2 * (n ** 2 + n * m + m ** 2) // self.R
        self.t1, self.t2 = (2 * m + n) // self.R, -(2 * n + m) // self.R

        # graphene lattice vectors
        self.a1 = self.a0 * np.array([np.sqrt(3) / 2, 1 / 2])
        self.a2 = self.a0 * np.array([np.sqrt(3) / 2, -1 / 2])
        self.b1 = self.b0 * np.array([1 / 2, np.sqrt(3) / 2])
        self.b2 = self.b0 * np.array([1 / 2, -np.sqrt(3) / 2])

        # CNT Supercell lattice vectors (C, T)
        self.C = self.n * self.a1 + self.m * self.a2
        self.T = self.t1 * self.a1 + self.t2 * self.a2

        # Linear 2 atom unit cell (Tl, T)
        self.l1, self.l2 = physics.twoAtomUnitCell(self.t1, self.t2, self.n)
        self.Tl = self.l1 * self.a1 + self.l2 * self.a2
        alpha = self.n * self.l2 - self.m * self.l1

        # Helical 2 atom unit cell (Th, C/D)
        self.h1, self.h2 = physics.twoAtomUnitCell(self.n // self.D, self.m // self.D, self.t1)
        self.Th = self.h1 * self.a1 + self.h2 * self.a2
        beta = self.h2*self.t1 - self.h1*self.t2

        # self.atomA = 1/3*(self.a1 + self.a2)
        # self.atomB = 2/3*(self.a1 + self.a2)
        # self.atomAlin = 1/3*(self.T + self.t1)
        # self.atomBlin = 2/3*(self.T + self.t1)
        # self.atomAhel = 1/3*(self.C/self.D + self.t2)
        # self.atomBhel = 2/3*(self.C/self.D + self.t2)

        # CNT crystallographic constants
        
        # self.normt1 = np.linalg.norm(self.t1)
        # self.normt2 = np.linalg.norm(self.t2)
        # self.cosTt1 = np.dot(self.T, self.t1)/self.normT/self.normt1
        # self.cosCt2 = np.dot(self.C/self.D, self.t2)/self.normCD/self.normt2

        # CNT reciprocal lattice vectors
        self.K1 = (-self.t2 * self.b1 + self.t1 * self.b2) / self.N  # K1.T = 0
        self.K2 = (self.m * self.b1 - self.n * self.b2) / self.N     # K2.C = 0

        self.k1L = self.N * self.K1
        self.k2L = self.K2 + self.K1 * alpha
        
        self.k1H = self.K1 * self.D + self.K2 * beta
        self.k2H = - self.K2 * self.N / self.D

        # CNT norms
        self.normT = la.norm(self.T)
        self.normC = la.norm(self.C)
        self.normK1 = la.norm(self.K1)
        self.normK2 = la.norm(self.K2)
        self.bzLin = la.norm(self.K2)
        self.bzHel = la.norm(self.k2H)
        self.normOrt = self.bzLin * abs(beta) / self.D

        # CNT data containers for electron and exciton bands, DOS, JDOS, etc
        self.electronBands = {}

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
            energy: str
                energy units [ 'eV' | 'Ha' | 'Ry' ]

            length: str
                length units [ 'nm' | 'bohr' | 'Angstrom' ]
        '''
        self.unitE = energy
        self.unitL = length
        self.unitInvL = length + '-1'


    def calculateCuttingLines(self, ksteps=100, sym='helical'):
        '''
        Calculate the cutting lines in the zone-folding scheme.

        Parameters:
        -----------
            ksteps: int (optional)
                number of kpoints for the discretisation (helical sym)
                default = 100

            sym: str (optional)
                linear or helical symmetry [ 'linear' | 'helical']
                default = 'helical'
        '''
        self.kSteps = ksteps
        # self.kStepsLin = int(self.bzLin / self.bzHel * self.kStepsHel)
        if sym == 'linear' or sym == 'lin':
            self.sym = 'linear'
            self.cuttingLines = physics.bzCuts(self.K2, self.K1, self.N, self.kSteps)
            self.kGrid = np.linspace(0.0, 1.0, self.kSteps, endpoint=False) * self.bzLin
            self.subN = self.N
        else:
            self.sym = 'helical'
            self.cuttingLines = physics.bzCuts(self.k2H, self.k1H / self.D, self.D, self.kSteps)
            self.kGrid = np.linspace(0.0, 1.0, self.kSteps, endpoint=False) * self.bzHel
            self.subN = self.D


    def calculateElectronBands(self, calc, **kwargs):
        '''
        Calculate the electron bands energy dispersion using different methods.

        Parameters:
        -----------
            calc: str
                specify the method for the band calculation
                [ 'TB' | 'DFT' ]

            **kwargs: (optional)
                Key-value arguments that depend on the calculation to be performed.

                calc = 'TB' calculation:
                ------------------------
                    gamma: float (optional)
                        TB on-site parameter
                        default = 3.0 eV
                    
                    fermi: float (optional)
                        position of the Fermi energy wrt the graphene Fermi level
                        default = 0.0 eV
                
                calc = 'DFT' calculation:
                -------------------------
                    nprocs: int (optional)
                        number of processors to use
                        default = 1
                    
                    from_file: bool (optional)
                        if True, read bands from a previous calculation
                        default = False

                    fourier_interp: bool (optional)
                        if True, the code will attempt  to interpolate the dft bands
                        to the BZ kpoint grid
                        default = False

                    pseudo_dir: str (optional)
                        directory containing the pseudopotential
                        default = './pseudo_dir'
                    
                    ecutwfc: float (optional)
                        wavefunction cutoff for the planewave expansion
                        default = 20 Ry
                    
                    ecutrho: float (optional)
                        density cutoff for the planewave expansion
                        default = 200 Ry
                    
                    nbnd: int (optional)
                        number of electron bands to calculate
                        default = 8
                    
                    deltan: int (optional)
                        number of bands around the Fermi energy to consider
                        default = 1

                    clat: float (optional)
                        spacing between adjecent graphene sheets
                        default = 1 nm
                    
                    kpoints: int (optional)
                        number of kpoints for the BZ sampling in the self-consistent calculation
                        default = 12
        '''
        if calc == 'TB':
            self.electronBands = tightBindingElectronBands(self, **kwargs)
        elif calc == 'DFT':
            dft.dftElectronBands(self, **kwargs)
        elif calc == 'something else':
            pass
        else:
            print(f'Calculation {calc} not implemented.')


    # def calculateKpointValleys(self):
    #     '''
    #     Calculate the (kx,ky) position of the minima and maxima of the electron band dispersion.
    #     Note! The K valley extrema are calculated in the helical coordinate system.
    #     '''
    #     bzCuts = self.bzCutsHel
    #     bands = self.electronBands
    #     subN, _, _, _ = bands.shape

    #     self.condKpointValleys = []
    #     self.valeKpointValleys = []

    #     for mu in range(0, subN): # angular momentum
    #         vales, conds = physics.valeCondBands(bands[mu])
    #         condMasks = mathematics.findFunctionListExtrema(conds, 'min')
    #         valeMasks = mathematics.findFunctionListExtrema(vales, 'max')
    #         cuts = bzCuts[mu]
            
    #         # valence band properties
    #         kValley = []
    #         for mask in valeMasks:
    #             kValley.append( cuts[mask] )
    #         self.valeKpointValleys.append( kValley )
            
    #         # conduction band properties
    #         kValley = []
    #         for mask in condMasks:
    #             kValley.append( cuts[mask] )
    #         self.condKpointValleys.append( kValley )


    def calculateExcitonBands(self, calc, **kwargs):
        '''
        Calculate the exciton bands energy dispersion using different methods.
        
        Note! Excitons are always computed in helical coordinates. This means
        that the underlying electron bands should also have helical symmetry.

        Parameters:
        -----------
            calc: str
                specify the method for the band calculation
                [ 'EM' | 'BSE' | 'more in the future' ]

            **kwargs: (optional)
                key-value arguments depend on the calculation to be performed.

                calc = 'EM' calculation:
                ---------------------------                    
                    bindEnergy: float (optional)
                        binding energy between electron and hole states
                        default = 0.0 eV
        '''
        if calc == 'EffMass' or calc == 'EM':
            self.excitonBands = effMassExcitonBands(self, **kwargs)
            # physics.effectiveMassExcitonBands(self, **kwargs)
        elif calc == 'BSE':
            pass
        elif calc == 'more in the future':
            print('Pollo!')
        else:
            print(f'Calculation {calc} not implemented.')


    def calculateDOS(self, which, name = 'all', enSteps=1000):
        '''
        Calcualte the density of states for a given particle energy dispersion.
        By default, the DOS is calculated on the helical symmetry.

        Parameters:
        -----------
            which: str
                particle type [ 'electron' | 'exciton' ]

            name: str (optional)
                name of the specific band to use for DOS
                default = all

            enSteps: int (optional)
                energy steps for the discretisation
                default = 1000
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
            which: str
                particle type [ 'el' | 'ex' | 'ph' ]

            enesteps: int (optional)
                energy steps for the discretisation
                default = 1000
        '''
        pass


    def saveToDirectory(self, dirpath):
        WORK_DIR = os.getcwd()
        OUT_DIR = os.path.join(WORK_DIR, dirpath)
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
        
        print(OUT_DIR)
        # for mu in range(0, self.NU):
        #     path = os.path.join(dirpath, f"bandLin{mu:03d}.txt")
        #     utils.save_file(self.bandLin[mu][0], self.bandLin[mu][1], path=path)
        # for mu in range(0, self.D):
        #     path = os.path.join(dirpath, f"bandHel{mu:03d}.txt")
        #     utils.save_file(self.bandHel[mu][0], self.bandHel[mu][1], path=path)
        # for k in self.excParaBands:
        #     path = os.path.join(dirpath, f"excPara{k}.txt")
        #     utils.save_file(*self.excParaBands[k], path=path)
        # for k in self.excPerpBands:
        #     path = os.path.join(dirpath, f"excPerp{k}.txt")
        #     utils.save_file(*self.excPerpBands[k], path=path)
        # for k in self.excDarkBands:
        #     path = os.path.join(dirpath, f"excDark{k}.txt")
        #     utils.save_file(*self.excDarkBands[k], path=path)

        # path = os.path.join(dirpath, f"cicco.txt")
        # utils.save_file(np.array([1,2,3]), np.array([4,5,6]), path=path)

        excitons = self.excitonBands['DFT-65rough']
        
        print(len(excitons))
        for exc in excitons:
            # path = os.path.join(dirpath, f"{exc}.txt")
            # utils.save_file(*excitons[exc], path=path)
            #np.savetxt(path, excitons[exc])
            #print(exc, ' ', excitons[exc])
            print(exc)


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
            path: str (optional)
                destination path where to save the figure
                default = shows the figure
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


def tightBindingElectronBands(cnt: Swcnt, gamma=3.0, fermi=0.0):
    '''
    Computes the band structure of the given CNT with the tight binding method in the zone folding scheme.
    Bands are computed using the symmetry selected in the "Swcnt.cuttingLines" method.
    '''
    if hasattr(cnt, 'cuttingLines'):
        cuts = cnt.cuttingLines
        subN, ksteps, _ = cuts.shape
        bands = np.zeros( (subN, 2, ksteps) ) # bands = E_n^mu(k), bands[mu index, n index, grid index]
        for mu, cut in enumerate(cuts):
            upperBand =   tightbinding.grapheneTBBands(cut, cnt.a1, cnt.a2, gamma) - fermi
            lowerBand = - tightbinding.grapheneTBBands(cut, cnt.a1, cnt.a2, gamma) - fermi
            bands[mu, 0, :] = lowerBand
            bands[mu, 1, :] = upperBand
        return bands
    else:
        print(f'Cutting lines not defined.')


def effMassExcitonBands(cnt: Swcnt, bindEnergy=0.0):
    pass