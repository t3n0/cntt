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
    def __init__(self, n, m):

        # units
        self.a0 = 0.2461  # nm
        self.gamma = 3.0  # eV
        # self.a0 = 2.461 # Angstrom
        # self.a0 = 4.6511 # bohr'

        # carbon nanotube parameters
        self.n, self.m = n, m
        self.R = np.gcd(2 * m + n, 2 * n + m)
        self.D = np.gcd(m, n)
        self.N = n ** 2 + n * m + m ** 2
        self.NU = 2 * self.N // self.R
        self.p, self.q = (2 * m + n) // self.R, -(2 * n + m) // self.R

        # CNT lattice vectors
        self.a1 = self.a0 * np.array([np.sqrt(3) / 2, 1 / 2])
        self.a2 = self.a0 * np.array([np.sqrt(3) / 2, -1 / 2])
        self.b0 = 4 * np.pi / np.sqrt(3) / self.a0
        self.b1 = self.b0 * np.array([1 / 2, np.sqrt(3) / 2])
        self.b2 = self.b0 * np.array([1 / 2, -np.sqrt(3) / 2])
        self.ac = self.a0/np.sqrt(3)
        self.bc = self.b0/np.sqrt(3)

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

    def calculateElectronBands(self, ksteps=10):
        # CNT linear and helical BZs
        self.normKC = np.linalg.norm(self.KC)
        self.normLin = np.linalg.norm(self.KT)
        self.normHel = np.linalg.norm(self.k2H)
        self.normOrt = abs(self.beta)/self.D*self.normLin
        kstepsLin = ksteps
        kstepsHel = int(self.normHel / self.normLin * kstepsLin)
        self.bzLin, self.bzCutsLin, self.bandLin = utils.subBands(self.KT, self.KC, self.a1, self.a2, self.NU, kstepsLin)
        self.bzHel, self.bzCutsHel, self.bandHel = utils.subBands(self.k2H, self.k1H / self.D, self.a1, self.a2, self.D, kstepsHel)
        self.bandLin = self.bandLin * self.gamma
        self.bandHel = self.bandHel * self.gamma

    def calculateExcitonBands(self, bindEnergy = 0.05, deltak = 10.0, kstep = 20):
        # CNT band minima, energies and masses
        self.bandMinHel = []
        self.bandMinXy = []
        self.bandInvMasses = []
        self.bandEnergy = []
        for mu in range(0, self.D):
            band = self.bandHel[mu]
            prime = np.gradient(band, self.bzHel)
            second = np.gradient(prime, self.bzHel)
            # todo avoid dirac points in metallic cnts
            # mask1 = (np.roll(prime, 1) < 0) * (prime > 0)
            # mask2 = abs(np.roll(prime, 1) - prime) < 1e-1
            # mask = mask1 * mask2
            mask = (np.roll(prime, 1) < 0) * (prime > 0)
            self.bandMinHel.append(self.bzHel[mask])
            self.bandInvMasses.append(second[mask])
            self.bandEnergy.append(band[mask])
            self.bandMinXy.append(self.bzCutsHel[mu][mask])
        
        self.bandMinHel = np.array(self.bandMinHel)
        self.bandInvMasses = np.array(self.bandInvMasses)
        self.bandEnergy = np.array(self.bandEnergy)
        self.bandMinXy = np.array(self.bandMinXy)

        # dic = [bz, band]
        self.excParaBands = {}
        self.excPerpBands = {}
        self.excDarkBands = {}
        nMin = len(self.bandMinXy[0])
        for mu in range(self.D):
            for nu in range(self.D):
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

    def saveData(self, dirpath):
        for mu in range(0, self.NU):
            path = os.path.join(dirpath, f"bandLin{mu:03d}.txt")
            utils.save_file(self.bzLin, self.bandLin[mu], path=path)
        for mu in range(0, self.D):
            path = os.path.join(dirpath, f"bandHel{mu:03d}.txt")
            utils.save_file(self.bzHel, self.bandHel[mu], path=path)
        for k in self.excParaBands:
            path = os.path.join(dirpath, f"excPara{k}.txt")
            utils.save_file(*self.excParaBands[k], path=path)
        for k in self.excPerpBands:
            path = os.path.join(dirpath, f"excPerp{k}.txt")
            utils.save_file(*self.excPerpBands[k], path=path)
        for k in self.excDarkBands:
            path = os.path.join(dirpath, f"excDark{k}.txt")
            utils.save_file(*self.excDarkBands[k], path=path)

    def textParams(self):
        text = (
            f"n, m = {self.n},{self.m}\n"
            f"Diameter = {np.linalg.norm(self.C)/np.pi:.2f} nm\n"
            f"C = {self.n:+d} a1 {self.m:+d} a2\n"
            f"T = {self.p:+d} a1 {self.q:+d} a2\n"
            f"t1 = {self.u1:+d} a1 {self.v1:+d} a2\n"
            f"t2 = {self.u2:+d} a1 {self.v2:+d} a2\n"
            f"NU = {self.NU}\n"
            f"D = {self.D}\n"
            f"BZ_lin = {self.normLin:.2f} nm-1\n"
            f"BZ_hel = {self.normHel:.2f} nm-1\n"
            f"K_ort = {self.normOrt:.2f} nm-1"
        )
        return text

    def plot(self, path=None):
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

        # -------------------------------------------------------
        # ax1 and ax2 (unit cell and BZ)
        ax1.set_aspect("equal")
        ax2.set_aspect("equal")
        minx, maxx, miny, maxy = utils.boundingRectangle(self.C, self.T, self.C + self.T, self.t1, self.t1 + self.T, self.t2, self.t2 + self.C / self.D,)
        hexDir = utils.hexPatches(minx, maxx, miny, maxy, self.a0, lat="dir")
        ax1.set_xlim(minx, maxx)
        ax1.set_ylim(miny, maxy)
        minx, maxx, miny, maxy = utils.boundingRectangle(
            self.bzCutsLin[0, 0, :],
            self.bzCutsLin[0, -1, :],
            self.bzCutsLin[-1, 0, :],
            self.bzCutsLin[-1, -1, :],
            self.bzCutsHel[0, 0, :],
            self.bzCutsHel[0, -1, :],
            self.bzCutsHel[-1, 0, :],
            self.bzCutsHel[-1, -1, :],
        )
        hexRec = utils.hexPatches(minx, maxx, miny, maxy, self.b0, lat="rec")
        ax2.set_xlim(minx, maxx)
        ax2.set_ylim(miny, maxy)

        # create unit cells vectors
        dirVectors = utils.arrowPatches(self.C, self.T, self.t1, self.t2, color='grey')
        #recVectors = utils.arrowPatches(self.k1L, self.k2L, self.k1H, self.k2H, color='grey')

        unitCell_la = np.array([[0.0, 0.0], self.C, self.C + self.T, self.T])
        unitCell_lh = np.array([[0.0, 0.0], self.t1, self.t1 + self.T, self.T])
        unitCell_ha = np.array([[0.0, 0.0], self.C / self.D, self.C / self.D + self.t2, self.t2])
        cells = utils.cellPatches([unitCell_la, unitCell_ha, unitCell_lh], ["g", "b", "r"])

        # plot hexagons and unit cells
        ax1.add_collection(hexDir)
        ax2.add_collection(hexRec)
        ax1.add_collection(cells)
        ax1.add_collection(dirVectors)
        #ax2.add_collection(recVectors)

        # plot cutting lines
        lines = utils.linePatches(
            self.bzCutsLin[:, 0, 0],
            self.bzCutsLin[:, 0, 1],
            self.bzCutsLin[:, -1, 0] - self.bzCutsLin[:, 0, 0],
            self.bzCutsLin[:, -1, 1] - self.bzCutsLin[:, 0, 1],
            ec="r",
        )
        ax2.add_collection(lines)
        lines = utils.linePatches(
            self.bzCutsHel[:, 0, 0],
            self.bzCutsHel[:, 0, 1],
            self.bzCutsHel[:, -1, 0] - self.bzCutsHel[:, 0, 0],
            self.bzCutsHel[:, -1, 1] - self.bzCutsHel[:, 0, 1],
            ec="b",
        )
        ax2.add_collection(lines)

        # plot band minima
        for mu in range(0, self.D):
            ax2.plot(*self.bandMinXy[mu].T, "r.")

        # -------------------------------------------------
        # ax3 ax4 plot subbands
        for mu in range(0, self.NU):
            ax3.plot(self.bzLin, self.bandLin[mu], "r")
            ax3.plot(self.bzLin, -self.bandLin[mu], "r")
        for mu in range(0, self.D):
            ax4.plot(self.bzHel, self.bandHel[mu],'b')
            ax4.plot(self.bzHel, -self.bandHel[mu],'b')
        
        # ax6 plot excitons
        for i,k in enumerate(self.excParaBands):
            if i == 0:
                label='Para'
            else:
                label='_'
            ax6.plot(*self.excParaBands[k],'r', label=label)
        for i,k in enumerate(self.excPerpBands):
            if i == 0:
                label='Perp'
            else:
                label='_'
            ax6.plot(*self.excPerpBands[k],'y', label=label)
        for i,k in enumerate(self.excDarkBands):
            if i == 0:
                label='Dark'
            else:
                label='_'
            ax6.plot(*self.excDarkBands[k],'grey', label=label)

        # ax5 ax7 plot DOS and absorption
        # todo

        # -------------------------------------------------
        # labels, ticks, range, grids, legends, texts
        ax1.set_xlabel("x (nm)")
        ax1.set_ylabel("y (nm)")
        ax2.set_xlabel("kx (nm-1)")
        ax2.set_ylabel("ky (nm-1)")
        ax3.set_title("Linear")
        ax3.set_ylabel("Energy (eV)")
        ax3.set_xlabel("k (nm-1)")
        ax4.set_title("Helical")
        ax5.set_title("DOS")
        ax6.set_ylabel("Energy (eV)")
        ax6.set_xlabel("k (nm-1)")
        ax4.set_yticklabels([])
        ax5.set_yticklabels([])
        ax4.set_xticklabels([])
        ax5.set_xticklabels([])
        ax7.set_yticklabels([])
        plt.text(0.05, 0.9, self.textParams(), ha="left", va="top", transform=fig.transFigure)

        minElEn, maxElEn = 1.1 * np.min(-self.bandHel), 1.1 * np.max(self.bandHel)
        minBzHel, maxBzHel = 1.1 * np.min(self.bzHel), 1.1 * np.max(self.bzHel)
        for ax in [ax4, ax6]:
            ax.set_xlim(minBzHel, maxBzHel)
            ax.locator_params(axis='x', nbins=20)
        for ax in [ax3, ax4, ax5]:
            ax.set_ylim(minElEn, maxElEn)
            ax.locator_params(axis='y', nbins=10)
        for ax in [ax3, ax4, ax5, ax6, ax7]:    
            ax.grid(linestyle='--')

        ax6.vlines(self.normOrt,0,6,linestyles ="dashed", colors ="k")
        ax6.vlines(-self.normOrt,0,6,linestyles ="dashed", colors ="k")
        ax6.legend(loc=(-0.2, 0.0))

        # save or plot
        if path == None:
            plt.show()
        else:
            fig.savefig(path)
            plt.close()
