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

import matplotlib.pyplot as plt
import numpy as np
import swcnt.utils as utils
import os


class Swcnt(object):

    def __init__(self, n, m, ksteps=100):

        # carbon nanotube parameters
        self.n, self.m = n, m
        self.R = np.gcd(2*m+n, 2*n+m)
        self.D = np.gcd(m, n)
        self.N = n**2 + n*m + m**2
        self.NU = 2*self.N//self.R
        self.p, self.q = (2*m+n)//self.R, -(2*n+m)//self.R

        # graphene lattice vectors

        self.a0 = 0.2461  # nm
        # self.a0 = 2.461 # Angstrom
        # self.a0 = 4.6511 # bohr'
        self.a1 = self.a0*np.array([np.sqrt(3)/2, 1/2])
        self.a2 = self.a0*np.array([np.sqrt(3)/2, -1/2])
        self.b0 = 4*np.pi/np.sqrt(3)/self.a0
        self.b1 = self.b0*np.array([1/2, np.sqrt(3)/2])
        self.b2 = self.b0*np.array([1/2, -np.sqrt(3)/2])

        # CNT lattice vectors
        self.C = self.n*self.a1 + self.m*self.a2
        self.T = self.p*self.a1 + self.q*self.a2
        self.u1, self.v1 = utils.minVector2AtomUnitCell(self.p, self.q, self.n)
        self.alpha = (self.n*self.R - 2*self.N*self.u1)/(2*self.m+self.n)
        self.t1 = self.u1*self.a1 + self.v1*self.a2
        self.u2, self.v2 = utils.minVector2AtomUnitCell(
            self.n//self.D, self.m//self.D, self.p)
        self.beta = self.D*(2*self.m+self.n)/self.R / \
            self.n + self.NU*self.u2/self.n
        self.t2 = self.u2*self.a1 + self.v2*self.a2

        # CNT reciprocal lattice vectors
        self.KC = (-self.q*self.b1 + self.p*self.b2)/self.NU
        self.KT = (self.m*self.b1 - self.n*self.b2)/self.NU
        self.k1L = self.NU*self.KC
        self.k2L = self.KT + self.alpha*self.KC
        self.k1H = self.beta*self.KT + self.D*self.KC
        self.k2H = -self.NU/self.D*self.KT

        # CNT linear and helical BZs
        normLin = np.linalg.norm(self.KT)
        normHel = np.linalg.norm(self.k2H)
        kstepsLin = ksteps
        kstepsHel = int(normHel/normLin*kstepsLin)
        self.bzLin, self.bzCutsLin, self.bandLin = utils.subBands(
            self.KT, self.KC, self.a1, self.a2, self.NU, kstepsLin)
        self.bzHel, self.bzCutsHel, self.bandHel = utils.subBands(
            self.k2H, self.k1H/self.D, self.a1, self.a2, self.D, kstepsHel)

    def plot(self, path=None):
        if path == None:
            fig = plt.figure(figsize=(16, 9))
        else:
            fig = plt.figure(figsize=(16, 9), dpi=300)

        fig.suptitle(f"CNT ({self.n},{self.m})")
        ax1 = fig.add_axes([0.23, 0.53, 0.35, 0.42])
        ax2 = fig.add_axes([0.63, 0.53, 0.35, 0.42])
        ax3 = fig.add_axes([0.05, 0.05, 0.2, 0.4])
        ax4 = fig.add_axes([0.25, 0.05, 0.58, 0.4])
        ax5 = fig.add_axes([0.83, 0.05, 0.15, 0.4])
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        minx, maxx, miny, maxy = utils.boundingRectangle(
            self.C, self.T, self.C + self.T, self.t1, self.t1 + self.T, self.t2, self.t2 + self.C/self.D)
        hexDir = utils.hexPatches(minx, maxx, miny, maxy, self.a0, lat='dir')
        ax1.set_xlim(minx, maxx)
        ax1.set_ylim(miny, maxy)
        minx, maxx, miny, maxy = utils.boundingRectangle(self.bzCutsLin[0, 0, :], self.bzCutsLin[0, -1, :], self.bzCutsLin[-1, 0, :],
                                                         self.bzCutsLin[-1, -1, :], self.bzCutsHel[0, 0, :], self.bzCutsHel[0, -1, :], self.bzCutsHel[-1, 0, :], self.bzCutsHel[-1, -1, :])
        hexRec = utils.hexPatches(minx, maxx, miny, maxy, self.b0, lat='rec')
        ax2.set_xlim(minx, maxx)
        ax2.set_ylim(miny, maxy)
        # labels
        ax1.set_xlabel('x (nm)')
        ax1.set_ylabel('y (nm)')
        ax2.set_xlabel('kx ({nm-1)')
        ax2.set_ylabel('ky (nm-1)')
        ax3.set_title('Linear')
        ax3.set_ylabel('Energy (eV)')
        ax4.set_title('Helical')
        ax5.set_title('DOS')
        ax4.set_yticks([])
        ax5.set_yticks([])
        plt.text(0.05, 0.9, self.textParams(), ha='left',
                 va='top', transform=fig.transFigure)

        # plot unic cells
        unitCell_la = np.array([[0.0, 0.0], self.C, self.C + self.T, self.T])
        unitCell_lh = np.array([[0.0, 0.0], self.t1, self.t1 + self.T, self.T])
        unitCell_ha = np.array(
            [[0.0, 0.0], self.C/self.D, self.C/self.D + self.t2, self.t2])
        cells = utils.cellPatches(
            [unitCell_la, unitCell_ha, unitCell_lh], ['g', 'b', 'r'])

        # plot hexagons
        ax1.add_collection(hexDir)
        ax2.add_collection(hexRec)
        ax1.add_collection(cells)

        # plot cutting lines
        lines = utils.linePatches(self.bzCutsLin[:, 0, 0], self.bzCutsLin[:, 0, 1], self.bzCutsLin[:, -1, 0] -
                                  self.bzCutsLin[:, 0, 0], self.bzCutsLin[:, -1, 1]-self.bzCutsLin[:, 0, 1], ec='r')
        ax2.add_collection(lines)
        lines = utils.linePatches(self.bzCutsHel[:, 0, 0], self.bzCutsHel[:, 0, 1], self.bzCutsHel[:, -1, 0] -
                                  self.bzCutsHel[:, 0, 0], self.bzCutsHel[:, -1, 1]-self.bzCutsHel[:, 0, 1], ec='b')
        ax2.add_collection(lines)
        for mu in range(0, self.D):
            ax2.plot(*self.excPos[mu].T, 'r.')

        # plot bands
        for mu in range(0, self.NU):
            ax3.plot(self.bzLin, self.bandLin[mu], 'r')
            ax3.plot(self.bzLin, -self.bandLin[mu], 'r')
        for mu in range(0, self.D):
            ax4.plot(self.bzHel, self.bandHel[mu], 'b')
            ax4.plot(self.bzHel, -self.bandHel[mu], 'b')
        mine, maxe = np.min(-self.bandHel), np.max(self.bandHel)
        for ax in [ax3, ax4, ax5]:
            ax.set_ylim(1.1*mine, 1.1*maxe)

        if path == None:
            plt.show()
        else:
            fig.savefig(path)
            plt.close()

    def saveData(self, dirpath):
        for mu in range(0, self.NU):
            path = os.path.join(dirpath, f'bandLin{mu:03d}.txt')
            utils.save_file(self.bzLin, self.bandLin[mu], path=path)
        for mu in range(0, self.D):
            path = os.path.join(dirpath, f'bandHel{mu:03d}.txt')
            utils.save_file(self.bzHel, self.bandHel[mu], path=path)

    def plotTransition(self, mu, pol):
        valBand = -utils.bands(self.bzCutsHel[mu], self.a1, self.a2)
        if pol == 'para':
            conBand = utils.bands(self.bzCutsHel[mu], self.a1, self.a2)
        elif pol == 'perp':
            conBand = utils.bands(self.bzCutsHel[mu]+self.KC, self.a1, self.a2)
        plt.plot(self.bzHel, valBand, 'r')
        plt.plot(self.bzHel, conBand, 'b')
        plt.show()

    def plotExcitons(self, mu, pol):
        valBand = -utils.bands(self.bzCutsHel[mu], self.a1, self.a2)
        if pol == 'para':
            conBand = utils.bands(self.bzCutsHel[mu], self.a1, self.a2)
        elif pol == 'perp':
            conBand = utils.bands(self.bzCutsHel[mu]+self.KC, self.a1, self.a2)
        fprime = np.gradient(valBand, self.bzHel)
        fsecond = np.gradient(fprime, self.bzHel)
        mask1 = (np.roll(fprime, 1) > 0)
        mask2 = (fprime < 0)
        print(mask1)
        print(mask2)
        print(mask1 * mask2)
        print(self.bzHel[mask1*mask2])
        plt.plot(self.bzHel, valBand, self.bzHel, fprime)
        plt.show()

    def calculateExcitons(self):
        self.excHelPos = []
        self.excPos = []
        self.excInvMasses = []
        self.excEnergy = []
        for mu in range(0, self.D):
            band = self.bandHel[mu]
            prime = np.gradient(band, self.bzHel)
            second = np.gradient(prime, self.bzHel)
            mask = (np.roll(prime, 1) < 0) * (prime > 0)
            self.excHelPos.append(self.bzHel[mask])
            self.excInvMasses.append(second[mask])
            self.excEnergy.append(band[mask])
            self.excPos.append(self.bzCutsHel[mu][mask])
        self.excHelPos = np.array(self.excHelPos)
        self.excInvMasses = np.array(self.excInvMasses)
        self.excEnergy = np.array(self.excEnergy)
        self.excPos = np.array(self.excPos)

        # excdic = {}
        # for mu in range(2):
        #     for nu in range(2):
        #         for i in range(5):
        #             for j in range(5):
        #                 e[mu, nu, i, j] = a[mu, i]+a[nu, j]
        #                 edic[f'E{mu}.{nu}.{i}.{j}'] = a[mu, i]+a[nu, j]

        # excPrams[1,1,0,2]=[pos, mass, energy]
        self.excParams = []
        masses = np.array(np.meshgrid(self.excHelPos.reshape(-1),
                                      self.excHelPos.reshape(-1))).T.reshape(-1, 2)
        print(masses)

    def textParams(self):
        text = (
            f'n, m = {self.n},{self.m}\n'
            f'Diameter = {np.linalg.norm(self.C)/np.pi:.2f} (nm)\n'
            f'C = {self.n:+d} a1 {self.m:+d} a2\n'
            f'T = {self.p:+d} a1 {self.q:+d} a2\n'
            f't1 = {self.u1:+d} a1 {self.v1:+d} a2\n'
            f't2 = {self.u2:+d} a1 {self.v2:+d} a2\n'
            f'NU = {self.NU}\n'
            f'D = {self.D}')
        return text
