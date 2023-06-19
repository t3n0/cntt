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


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.markers import MarkerStyle
from matplotlib.cm import get_cmap
import cntt.physics as physics
from cntt.swcnt import Swcnt


def show():
    plt.show()


def mycolors(i, n, which='cividis'):
    x = np.linspace(0, 1, n)
    cmap = get_cmap(which)
    rgba = cmap(x)
    return rgba[i]


def mylabel(i, label):
    if i == 0:
        return label
    else:
        return '_'


def dirLat(*cnts: Swcnt, ax=None, pad='on_top', cmap='Paired'):
    '''
    Plots the super cell, unit cell and lattice vectors of the given CNTs.
    If an <ax> is given, the plot is drawn on the given axis, otherwise a new figure is made.
    CNTs should be commensurate (i.e. same a0), otherwise plots might be inconsistent.

    Parameters:
    -----------
        cnts: Swcnt objects
            list or tuple of CNTs to plot

        ax: matplotlib axis (optional)
            axis where to draw the plot
            default, make a new figure
        
        pad: str or int (optional)
            defines the shift on x where to draw the CNTs
            can be either 'on_top', 'by_side' or any integer number of unitcells
            default 'on_top'
        
        cmap: str (optional)
            color map that specify the sequence of colors to use for the plots
            cmap options are listed on the matplotlib documentation
            default 'Paired'

    Returns:
    --------
        ax: matplotlib axis object
            axis where the plot has been drawn
    '''
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes([0.08, 0.08, 0.87, 0.87])
    
    # collect and build all CNTs cells
    cells = []
    for cnt in cnts:
        sc = cnt.getCell('sc')
        uc = cnt.getCell('uc')
        cells.append( sc )
        cells.append( uc )
    cells = np.array(cells)
    
    # define padding
    if pad == 'by_side':
        padx = np.rint(np.max(cells[:,:,0])/3/cnt.ac) * 3 *cnt.ac
    elif pad == 'on_top':
        padx = 0.0
    else:
        padx = pad * 3 *cnt.ac

    # padding all cells and define vectors
    vectors = []
    for i, cell in enumerate(cells):
        cell[:,0] += padx * (i//2) # i = 0, 0, 1, 1, 2, 2, 3, 3, ...
        origin = cell[0]
        delta1 = cell[1] - origin
        delta2 = cell[3] - origin
        vectors.append( [origin, delta1] )
        vectors.append( [origin, delta2] )
    vectors = np.array(vectors)

    colors = [mycolors(i, len(cells), cmap) for i in range(len(cells)) ]
    unitCells = cellPatches(cells, colors=colors)

    # build all CNTS lattice vectors
    latVecs = arrowPatches(vectors)

    # hexagons        
    if pad == 'on_top':
        minx, maxx, miny, maxy = boundingSquare(cells)
    else:
        minx, maxx, miny, maxy = boundingRectangle(cells)
    hexs = dirHexPatches(minx, maxx, miny, maxy, cnt.a0)

    # plot
    ax.add_collection(hexs)
    ax.add_collection(unitCells)
    ax.add_collection(latVecs)
    ax.set_aspect(1)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    return ax


def recLat(cnt: Swcnt, ax=None):
    # _, _, invLfactor = physics.unitFactors(cnt)
    # KT, k1L, k2L, k1H, k2H, b0 = physics.changeUnits(cnt, invLfactor, 'KT', 'k1L', 'k2L', 'k1H', 'k2H', 'b0')
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    # hexagons
    rc = cnt.getCell('rc')
    # minx, maxx, miny, maxy = boundingRectangle(*rc)
    minx, maxx, miny, maxy = boundingSquare(*rc)
    hexs = recHexPatches(minx, maxx, miny, maxy, cnt.b0)
    # cells
    # recCell_lh = np.array([[0.0, 0.0], k1L, k1L + k2L, k2L]) - 0.5*KT
    # recCell_ha = np.array([[0.0, 0.0], k1H, k1H + k2H, k2H]) + 0.5*cnt.NU/cnt.D*KT
    recCells = cellPatches([rc], ["r"])
    # lattice vectors
    #latVecs = arrowPatches(cnt.k1L, cnt.k2L, cnt.k1H, cnt.k2H, color='grey', d= cnt.KT)

    # if hasattr(cnt, 'bzCutsLin') and hasattr(cnt, 'bzCutsHel'):
    #     # plot linear cutting lines
    #     cuts = invLfactor * cnt.bzCutsLin
    #     cutsPatches = linePatches(cuts[:, 0, 0], cuts[:, 0, 1], cuts[:, -1, 0] - cuts[:, 0, 0], cuts[:, -1, 1] - cuts[:, 0, 1], ec="r")
    #     ax.add_collection(cutsPatches)
    #     # plot helical cutting lines
    #     cuts = invLfactor * cnt.bzCutsHel
    #     cutsPatches = linePatches(cuts[:, 0, 0], cuts[:, 0, 1], cuts[:, -1, 0] - cuts[:, 0, 0], cuts[:, -1, 1] - cuts[:, 0, 1], ec="b")
    #     ax.add_collection(cutsPatches)

    # # KpointValleys
    # for key in cnt.condKpointValleys:           # name of the calculation
    #     for cuts in cnt.condKpointValleys[key]: # mu index
    #         for xys in cuts:                    # band index
    #             for xy in xys:                  # extrema index
    #                 ax.scatter(xy[0], xy[1], s=50, c='white', edgecolor='black', marker=MarkerStyle("o", fillstyle="right"))
    # for key in cnt.valeKpointValleys:
    #     for cuts in cnt.valeKpointValleys[key]:
    #         for xys in cuts:
    #             for xy in xys:
    #                 ax.scatter(xy[0], xy[1], s=50, c='white', edgecolor='black', marker=MarkerStyle("o", fillstyle="left"))
    
    # plot
    ax.add_collection(hexs)
    ax.add_collection(recCells)
    #ax.add_collection(latVecs)
    ax.set_aspect("equal")
    ax.set_xlabel(f'kx ({cnt.unitInvL})')
    ax.set_ylabel(f'ky ({cnt.unitInvL})')
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    return ax


def electronBands(cnt, ax=None, sym='hel'):
    efactor, _, invLfactor = physics.unitFactors(cnt)
    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    dicBands = getattr(cnt, f'electronBands{sym.capitalize()}')
    NN = len(dicBands)
    for i, key in enumerate(dicBands):
        label = key
        bands = dicBands[key]
        subN, bandN, _, _ = bands.shape
        for mu in range(0, subN): # loop over the cuts
            for n in range(0, bandN): # loop over the bands for the given cut
                ax.plot(invLfactor*bands[mu, n, 0, :], efactor*bands[mu,n,1,:], color=mycolors(i,NN), label=label)
                ax.set_ylabel(f'Energy ({cnt.unitE})')
                ax.set_xlabel(f'k ({cnt.unitInvL})')
                label = '_'
    #energyExtrema(cnt, ax, 'cond')
    #energyExtrema(cnt, ax, 'vale')
    ax.axhline(0,ls='--',c='grey')
    ax.legend()


def energyExtrema(cnt, ax, which):
    efactor, _, invLfactor = physics.unitFactors(cnt)
    energies = getattr(cnt, f'{which}EnergyZeros') #cnt.condEnergyZeros
    kpoints = getattr(cnt, f'{which}KpointZeros') #cnt.condKpointZeros
    NN = len(kpoints)
    for i, key in enumerate(kpoints):                                         # name of the calculation
        for kcuts, ecuts in zip(kpoints[key], energies[key]):   # mu index
            count = 0
            for ks, es in zip(kcuts, ecuts):                    # band index
                for k, e in zip(ks, es):                        # extrema index
                    plt.text(k * invLfactor, e * efactor, f'{count}', ha="center", va="center")
                    ax.scatter(k * invLfactor, e * efactor, s=250, color=mycolors(i,NN), alpha=0.3)
                    count += 1


def excitonBands(cnt, ax=None):
    efactor, _, invLfactor = physics.unitFactors(cnt)
    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    NN = len(cnt.excitonBands)
    maxEnergy = 1.0
    for i, name in enumerate(cnt.excitonBands):
        label = name
        for key in cnt.excitonBands[name]:
            newMax = np.max(cnt.excitonBands[name][key][1])
            if newMax > maxEnergy: maxEnergy = newMax
            ax.plot(invLfactor*cnt.excitonBands[name][key][0], efactor*cnt.excitonBands[name][key][1], color=mycolors(i,NN), label=label)
            ax.set_ylabel(f'Energy ({cnt.unitE})')
            ax.set_xlabel(f'k ({cnt.unitInvL})')
            label = '_'
    ax.set_ylim(0.0, efactor*maxEnergy)
    ax.set_xlim(-cnt.normHel/2, cnt.normHel/2)
    ax.vlines(cnt.normOrt,0,efactor*maxEnergy,linestyles ="dashed", colors ="k")
    ax.vlines(-cnt.normOrt,0,efactor*maxEnergy,linestyles ="dashed", colors ="k")
    ax.legend()


def electronDOS(cnt, ax=None, swapAxes=False):
    efactor, _, _ = physics.unitFactors(cnt)
    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    NN = len(cnt.electronDOS)
    for i, name in enumerate(cnt.electronDOS.keys()):
        en = cnt.electronDOS[name][0]
        dos = cnt.electronDOS[name][1]
        maxDos = np.max(dos)
        #integral = np.trapz(dos, en)
        #print(name, integral)
        if swapAxes:
            ax.plot(dos, efactor * en, color=mycolors(i,NN), label=name)
            ax.set_xlabel(f'DOS')
            ax.set_ylabel(f'Energy ({cnt.unitE})')
            ax.hlines(0,0, maxDos, linestyles ="dashed", colors ="grey")
        else:
            ax.plot(efactor * en, dos, color=mycolors(i,NN), label=name)
            ax.set_ylabel(f'DOS')
            ax.set_xlabel(f'Energy ({cnt.unitE})')
            ax.vlines(0,0, maxDos, linestyles ="dashed", colors ="grey")
    ax.legend()


def excitonDOS(cnt, ax=None, swapAxes=False):
    efactor, _, _ = physics.unitFactors(cnt)
    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    NN = len(cnt.excitonDOS)
    for i, name in enumerate(cnt.excitonDOS.keys()):
        en = cnt.excitonDOS[name][0]
        dos = cnt.excitonDOS[name][1]
        #integral = np.trapz(dos, en)
        #print(name, integral)
        if swapAxes:
            ax.plot(dos, efactor * en, color=mycolors(i,NN), label=name)
            ax.set_xlabel(f'DOS')
            ax.set_ylabel(f'Energy ({cnt.unitE})')
        else:
            ax.plot(efactor * en, dos, color=mycolors(i,NN), label=name)
            ax.set_ylabel(f'DOS')
            ax.set_xlabel(f'Energy ({cnt.unitE})')
    ax.legend()


def boundingRectangle(cells):
    vecs = np.array(cells)
    minx = np.min(vecs[:, :, 0])
    maxx = np.max(vecs[:, :, 0])
    miny = np.min(vecs[:, :, 1])
    maxy = np.max(vecs[:, :, 1])
    return minx, maxx, miny, maxy


def boundingSquare(args):
    minx, maxx, miny, maxy = boundingRectangle(args)
    spanx = maxx - minx
    spany = maxy - miny
    span = max(spanx, spany)
    meanx = (maxx + minx) / 2
    meany = (maxy + miny) / 2
    return meanx-span/2, meanx+span/2, meany-span/2, meany+span/2


def arrowPatches(vecs):
    patches = []
    width = np.max(vecs)/400
    for origin, delta in vecs:
        arrow = mpatches.FancyArrow(*origin, *delta, width=width, length_includes_head=True, color='k')
        patches.append(arrow)
    return PatchCollection(patches, match_original=True)


def cellPatches(cells, colors, alpha=0.5):
    patches = []
    for aux, c in zip(cells, colors):
        cell = mpatches.Polygon(aux, closed=True, color=c, alpha=alpha)
        patches.append(cell)
    return PatchCollection(patches, match_original=True)


def linePatches(xs, ys, dxs, dys, ec="k", fc="w"):
    patches = []
    for x, y, dx, dy in zip(xs, ys, dxs, dys):
        line = mpatches.FancyArrow(x, y, dx, dy, width=0.01, head_width=0)
        patches.append(line)
    lines = PatchCollection(patches, edgecolor=ec, facecolor=fc)
    return lines


def dirHexPatches(minx, maxx, miny, maxy, c):
    minNx = 2 * int(np.floor(minx / c / np.sqrt(3)))
    maxNx = int(np.ceil(2 * maxx / c / np.sqrt(3)))
    minNy = int(np.floor(miny / c)) - 1
    maxNy = int(np.ceil(maxy / c))
    rotation = np.pi / 6
    xs, ys = np.meshgrid(range(minNx, maxNx + 1), range(minNy, maxNy + 1))
    xs = xs * np.sqrt(3) / 2
    ys = ys.astype("float")
    ys[:, 1::2] += 0.5
    ys = ys.reshape(-1) * c
    xs = xs.reshape(-1) * c
    patches = []
    for x, y in zip(xs, ys):
        hex = mpatches.RegularPolygon((x, y), numVertices=6, radius=c / np.sqrt(3), orientation=rotation)
        patches.append(hex)
    return PatchCollection(patches, edgecolor="k", facecolor=(1,1,1,0))


def recHexPatches(minx, maxx, miny, maxy, c):
    minNx = int(np.floor(minx / c)) - 1
    maxNx = int(np.ceil(maxx / c))
    minNy = 2 * int(np.floor(miny / c / np.sqrt(3)))
    maxNy = int(np.ceil(2 * maxy / c / np.sqrt(3)))
    rotation = 0
    xs, ys = np.meshgrid(range(minNx, maxNx + 1), range(minNy, maxNy + 1))
    ys = ys * np.sqrt(3) / 2
    xs = xs.astype("float")
    xs[1::2, :] += 0.5
    ys = ys.reshape(-1) * c
    xs = xs.reshape(-1) * c
    patches = []
    for x, y in zip(xs, ys):
        hex = mpatches.RegularPolygon((x, y), numVertices=6, radius=c / np.sqrt(3), orientation=rotation)
        patches.append(hex)
    return PatchCollection(patches, edgecolor="k", facecolor=(1,1,1,0))
