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
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection


def changeUnits():
    pass

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


def boundingRectangle(*args):
    vecs = [[0.0, 0.0]]
    for arg in args:
        vecs.append(arg)
    vecs = np.array(vecs)
    minx = np.min(vecs[:, 0])
    maxx = np.max(vecs[:, 0])
    miny = np.min(vecs[:, 1])
    maxy = np.max(vecs[:, 1])
    return minx, maxx, miny, maxy


def arrowPatches(*vec,color):
    patches = []
    for v in vec:
        arrow = mpatches.FancyArrow(0,0,v[0],v[1], width=0.005, length_includes_head=True, color=color)
        #ax.add_patch(arrow)
        patches.append(arrow)
    return PatchCollection(patches, match_original=True)

def cellPatches(cells, colors):
    patches = []
    for aux, c in zip(cells, colors):
        cell = mpatches.Polygon(aux, closed=True, color=c, alpha=0.2)
        patches.append(cell)
    cells = PatchCollection(patches, match_original=True)
    return cells


def linePatches(xs, ys, dxs, dys, ec="k", fc="w"):
    patches = []
    for x, y, dx, dy in zip(xs, ys, dxs, dys):
        line = mpatches.FancyArrow(x, y, dx, dy, width=0.01, head_width=0)
        patches.append(line)
    lines = PatchCollection(patches, edgecolor=ec, facecolor=fc)
    return lines


def hexPatches(minx, maxx, miny, maxy, c, lat="dir"):
    if lat == "dir":
        minNx = 2 * int(np.floor(minx / c / np.sqrt(3)))
        maxNx = int(np.ceil(2 * maxx / c / np.sqrt(3)))
        minNy = int(np.floor(miny / c)) - 1
        maxNy = int(np.ceil(maxy / c))
        rotation = np.pi / 6
    elif lat == "rec":
        minNx = int(np.floor(minx / c)) - 1
        maxNx = int(np.ceil(maxx / c))
        minNy = 2 * int(np.floor(miny / c / np.sqrt(3)))
        maxNy = int(np.ceil(2 * maxy / c / np.sqrt(3)))
        rotation = 0
    xs, ys = np.meshgrid(range(minNx, maxNx + 1), range(minNy, maxNy + 1))
    if lat == "dir":
        xs = xs * np.sqrt(3) / 2
        ys = ys.astype("float")
        ys[:, 1::2] += 0.5
    elif lat == "rec":
        ys = ys * np.sqrt(3) / 2
        xs = xs.astype("float")
        xs[1::2, :] += 0.5
    ys = ys.reshape(-1) * c
    xs = xs.reshape(-1) * c
    patches = []
    for x, y in zip(xs, ys):
        line = mpatches.RegularPolygon((x, y), numVertices=6, radius=c / np.sqrt(3), orientation=rotation)
        patches.append(line)
    hexs = PatchCollection(patches, edgecolor="k", facecolor=(1,1,1,0))
    return hexs


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


def grapheneTBBands(cuts, a1, a2, gamma=3.0):
    bands = []
    for cut in cuts:
        band = gamma * np.sqrt(3 + 2 * np.cos(np.dot(cut, a1)) + 2 * np.cos(np.dot(cut, a2)) + 2 * np.cos(np.dot(cut, (a2 - a1))))
        bands.append(band)
        bands.append(-band)
    return np.array(bands)


def _bands(k, a1, a2):
    band = np.sqrt(3 + 2 * np.cos(np.dot(k, a1)) + 2 * np.cos(np.dot(k, a2)) + 2 * np.cos(np.dot(k, (a2 - a1))))
    return band


def tightBindingElectronBands(cnt, name, sym='hel', gamma=3.0):
    attrCuts = f'bzCuts{sym.capitalize()}'
    attrNorm = f'norm{sym.capitalize()}'
    attrBands = f'electronBands{sym.capitalize()}'
    if hasattr(cnt, attrCuts):
        bzCuts = getattr(cnt, attrCuts)
        bzNorm = getattr(cnt, attrNorm)
        subN, ksteps, _ = bzCuts.shape
        bz = np.linspace(-0.5, 0.5, ksteps) * bzNorm
        bands = grapheneTBBands(bzCuts, cnt.a1, cnt.a2, gamma)
        data = np.zeros((2*subN, 2, ksteps))
        data[:,0,:] = bz
        data[:,1,:] = bands
        getattr(cnt, attrBands)[name] = data
        # cnt.bandsHel[name] = data
    else:
        print(f'Cutlines "{sym}" not defined.')    


def _subBands(k1, k2, a1, a2, N, ksteps):
    norm = np.linalg.norm(k1)
    kmesh = np.linspace(-0.5, 0.5, ksteps)
    bz = kmesh * norm
    bzCuts = []
    subBands = []
    outer = np.outer(kmesh, k1)
    for mu in range(0, N):
        cut = outer + mu * k2
        subBands.append( [bz, bands(cut, a1, a2)] )
        bzCuts.append(cut)
    bzCuts = np.array(bzCuts)
    subBands = np.array(subBands)
    return bzCuts, subBands




def opt_mat_elems(k, a1, a2, n, m):
    N = n ** 2 + n * m + m ** 2
    elem = (((n - m) * np.cos(np.dot(k, (a2 - a1))) - (2 * n + m) * np.cos(np.dot(k, a1)) + (n + 2 * m) * np.cos(np.dot(k, a2))) / 2 / np.sqrt(N) / bands(k, a1, a2))
    return elem

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


