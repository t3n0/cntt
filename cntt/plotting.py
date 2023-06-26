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
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
from matplotlib.markers import MarkerStyle
from matplotlib.cm import get_cmap
import cntt.physics as physics
from cntt.swcnt import Swcnt


def show():
    plt.show()


def mycolors(n, which='cividis'):
    x = np.linspace(0, 1, n)
    res = get_cmap(which)(x)
    return res


def dirLat(*cnts: Swcnt, ax=None, pad='on_top', shift=[0,0], cmap = 'Set1'):
    '''
    Plots the super cell, unit cell and lattice vectors of the given CNTs.
    If an <axis> is given, the plot is drawn on the given axis, otherwise a new figure is made.
    CNTs should be commensurate (i.e. same a0), otherwise plots might be inconsistent.

    Parameters:
    -----------
        cnts: Swcnt objects
            List or tuple of CNTs.

        ax: matplotlib axis (optional)
            Axis where to draw the plot.
            Default, make a new figure.
        
        pad: str or int (optional)
            Defines the padding on x where to draw the CNTs
            can be either 'on_top', 'by_side' or any integer number of unitcells.
            Default 'on_top'.

        shift: [int, int] (optional)
            Rigid shift of the entire plot in units of lattice vectors.
            Useful to plot other cnts on a provious <ax>.
            Default (0,0).
        
        cmap: str (optional)
            Color map that specify the sequence of colors to use for the plots.
            Cmap options are listed on the matplotlib documentation.
            Default 'Paired'.

    Returns:
    --------
        ax: matplotlib axis object
            Axis where the plot has been drawn.
    '''
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches((5, 5))
        
    # collect and build all CNTs cells
    cells = []
    for cnt in cnts:
        sc = cnt.getCell('sc')
        uc = cnt.getCell('uc')
        cells.append( sc )
        cells.append( uc )
    cells = np.array(cells)
    
    # shift
    shift = shift[0] * (cnt.a1 + cnt.a2) + shift[1] * (cnt.a1 - cnt.a2)
    cells[:] += shift

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

    colors = mycolors(len(cells), cmap)
    unitCells = cellPatches(cells, colors=colors)

    # build all CNTS lattice vectors
    latVecs = arrowPatches(vectors, cnt.ac/30)

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
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    return ax


def recLat(*cnts: Swcnt, ax=None, pad='on_top', shift=[0,0], cmap = 'Set1'):
    '''
    Plots the reciprocal cell and the lattice vectors of the given CNTs.
    If an <axis> is given, the plot is drawn on the given axis, otherwise a new figure is made.
    CNTs should be commensurate (i.e. same a0), otherwise plots might be inconsistent.

    Parameters:
    -----------
        cnts: Swcnt objects
            List or tuple of CNTs.

        ax: matplotlib axis (optional)
            Axis where to draw the plot.
            Default, make a new figure.
        
        pad: str or int (optional)
            Defines the padding on kx where to draw the CNTs
            can be either 'on_top', 'by_side' or any integer number of unitcells.
            Default 'on_top'.

        shift: [int, int] (optional)
            Rigid shift of the entire plot in units of lattice vectors.
            Useful to plot other cnts on a provious <ax>.
            Default (0,0).
        
        cmap: str (optional)
            Color map that specify the sequence of colors to use for the plots.
            Cmap options are listed on the matplotlib documentation.
            Default 'Paired'.

    Returns:
    --------
        ax: matplotlib axis object
            Axis where the plot has been drawn.
    '''
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches((5, 5))

    # collect and build all CNTs reciprocal cells
    cells = []
    for cnt in cnts:
        rc = cnt.getCell('rc')
        cells.append( rc )
    cells = np.array(cells)
    
    # shift
    shift = shift[0] * (cnt.b1 + cnt.b2) + shift[1] * (cnt.b1 - cnt.b2)
    cells[:] += shift

    # define padding
    if pad == 'by_side':
        padx = np.rint(np.max(cells[:,:,0])/3/cnt.bc) * 3 *cnt.bc
    elif pad == 'on_top':
        padx = 0.0
    else:
        padx = pad * 3 *cnt.bc

    # padding all cells and define vectors
    vectors = []
    for i, cell in enumerate(cells):
        cell[:,0] += padx * i
        origin = cell[0]
        delta1 = cell[1] - origin
        delta2 = cell[3] - origin
        vectors.append( [origin, delta1] )
        vectors.append( [origin, delta2] )
    vectors = np.array(vectors)

    colors = mycolors(len(cells), cmap)
    recCells = cellPatches(cells, colors=colors)

    # build all CNTS lattice vectors
    latVecs = arrowPatches(vectors, cnt.bc/30)
    # hexagons        
    if pad == 'on_top':
        minx, maxx, miny, maxy = boundingSquare(cells)
    else:
        minx, maxx, miny, maxy = boundingRectangle(cells)
    hexs = recHexPatches(minx, maxx, miny, maxy, cnt.b0)

    # plot cutting lines
    for i, cnt in enumerate(cnts):
        cuts = cnt.cuttingLines
        cuts[:,:] += shift
        cuts[:,:,0] += padx * i
        lines = cuts[:, 0, 0], cuts[:, 0, 1], cuts[:, -1, 0] - cuts[:, 0, 0], cuts[:, -1, 1] - cuts[:, 0, 1]
        cutsPatches = linePatches(*lines, color='r', width=cnt.bc/50)
        ax.add_collection(cutsPatches)
    
    # plot
    ax.add_collection(hexs)
    ax.add_collection(recCells)
    ax.add_collection(latVecs)
    ax.set_aspect(1)
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    return ax


def electronBands(cnt: Swcnt, ax=None, gammaShift=True, ylims=None, cmap = 'Set1'):

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches((8, 5))

    colors = mycolors(cnt.subN, cmap)

    # custom lines for the legend
    custom_lines = []
    for mu in range(cnt.subN):
        line = Line2D([0], [0], color=colors[mu], lw=2, label=f'mu = {mu}')
        custom_lines.append( line )
    ax.legend(handles=custom_lines, loc = 'upper right')
    
    # gamma shift
    bz = max(cnt.kGrid)
    if gammaShift:
        kGrid = cnt.kGrid - bz/2
        xlims = (-bz/2, bz/2)
    else:
        kGrid = cnt.kGrid
        xlims = (0, bz)

    for mu in range(cnt.subN):
        bands = cnt.electronBands[mu]
        for band in bands:
            if gammaShift:
                band = np.roll(band, cnt.kSteps//2)
            ax.plot(kGrid, band, color = colors[mu])
    
    ax.set_ylabel('Energy')
    ax.set_xlabel('k')
    ax.set_xlim(xlims)
    if ylims:
        ax.set_ylim(ylims)
    ax.axhline(0, ls = '--', c = 'grey')
    ax.grid(True)
    return ax


def excitonBands(cnt: Swcnt, ax=None, arrange='individual', gammaShift=True, ylims=0, cmap = 'Set1'):
    '''
    arrange = together, individual
    mode = static, interactive
    '''
    if ax == None:
        if arrange == 'together':
            fig, ax = plt.subplots()
            fig.set_size_inches((8, 5))
        elif arrange == 'individual':
            fig, ax = plt.subplots(cnt.subN, cnt.subN, sharex='all', sharey='all')
            fig.set_size_inches((8, 5))

    # colors
    colors = mycolors(cnt.subN*cnt.subN, cmap)

    ort = cnt.normOrt
    bz = cnt.bzHel
    if gammaShift:
        kGrid = cnt.kGrid - bz/2
        xlims = (-bz/2, bz/2)
    else:
        kGrid = cnt.kGrid
        xlims = (0, bz)

    custom_lines = []
    if arrange == 'together':
        for mu1 in range(cnt.subN):
            for mu2 in range(cnt.subN):
                color = colors[mu1*cnt.subN + mu2]
                line = Line2D([0], [0], color=color, lw=2, label=f'mu1, mu2 = ({mu1},{mu2})')
                custom_lines.append( line )
                if gammaShift:
                    cont1 = np.roll(cnt.excitonContinuum[mu1, mu2][0], cnt.kSteps//2)
                    cont2 = np.roll(cnt.excitonContinuum[mu1, mu2][1], cnt.kSteps//2)
                ax.fill_between(kGrid, cont1, cont2, color='grey', alpha=0.2)
                for band in cnt.excitonBands[mu1, mu2]:
                    if gammaShift:
                        band = np.roll(band, cnt.kSteps//2)
                    ax.plot(kGrid, band, lw=2, color=color)
        ax.vlines( ort, 0, 10, linestyles ="dashed", colors ="k")
        ax.vlines(-ort, 0, 10, linestyles ="dashed", colors ="k")
        ax.legend(handles=custom_lines, loc='upper right')
        ax.set_ylabel('Energy')
        ax.set_xlabel('k')
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.grid(True)
    elif arrange == 'individual':
        for mu1 in range(cnt.subN):
            for mu2 in range(cnt.subN):
                color = colors[mu1*cnt.subN + mu2]
                line = Line2D([0], [0], color=color, lw=2, label=f'mu1, mu2 = ({mu1},{mu2})')
                ax[mu1,mu2].legend(handles=[line], loc='upper right')
                if gammaShift:
                    cont1 = np.roll(cnt.excitonContinuum[mu1, mu2][0], cnt.kSteps//2)
                    cont2 = np.roll(cnt.excitonContinuum[mu1, mu2][1], cnt.kSteps//2)
                ax[mu1,mu2].fill_between(kGrid, cont1, cont2, color='grey', alpha=0.2)
                for band in cnt.excitonBands[mu1, mu2]:
                    if gammaShift:
                        band = np.roll(band, cnt.kSteps//2)
                    ax[mu1,mu2].plot(kGrid, band, lw=2, color=color)                
                if abs(mu1 - mu2) == 1 or abs(mu1 - mu2) == cnt.subN - 1:
                    ax[mu1,mu2].vlines( ort, 0, 10, ls ='--', color ="k")
                    ax[mu1,mu2].vlines(-ort, 0, 10, ls ='--', color ="k")
                
                ax[mu1,mu2].set_xlim(xlims)
                ax[mu1,mu2].set_ylim(ylims)
        
                ax[mu1,mu2].grid(True)
    return ax


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


def arrowPatches(vecs, width):
    patches = []
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


def linePatches(xs, ys, dxs, dys, color, width):
    patches = []
    for x, y, dx, dy in zip(xs, ys, dxs, dys):
        line = mpatches.FancyArrow(x, y, dx, dy, width=width, head_width=0, color=color)
        patches.append(line)
    lines = PatchCollection(patches, match_original=True)
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
