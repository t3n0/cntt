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


def mylabel(i, label):
    if i == 0:
        return label
    else:
        return '_'


def dirLat(*cnts: Swcnt, ax=None, pad='on_top', shift=[0,0], cmap='Paired'):
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


def recLat(*cnts: Swcnt, ax=None, pad='on_top', shift=[0,0], cmap='Paired'):
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
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes([0.08, 0.08, 0.87, 0.87])

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
    ax.add_collection(latVecs)
    ax.set_aspect(1)
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    return ax


def electronBands(cnt: Swcnt, ax=None, legend=True, gammaShift=True, ylims=None):

    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_axes([0.08, 0.08, 0.87, 0.87])

    colors = mycolors(cnt.subN, 'Set1')

    # custom lines for the legend
    if legend:
        custom_lines = []
        for mu in range(cnt.subN):
            line = Line2D([0], [0], color=colors[mu], lw=2, label=f'mu = {mu}')
            custom_lines.append( line )
        ax.legend(handles=custom_lines, loc='upper right')
    
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
            ax.plot(kGrid, band, color=colors[mu])
    
    ax.set_ylabel('Energy')
    ax.set_xlabel('k')
    ax.set_xlim(*xlims)
    if ylims:
        ax.set_ylim(*ylims)
    ax.axhline(0, ls='--', c='grey')
    

# def energyExtrema(cnt, ax, which):
#     efactor, _, invLfactor = physics.unitFactors(cnt)
#     energies = getattr(cnt, f'{which}EnergyZeros') #cnt.condEnergyZeros
#     kpoints = getattr(cnt, f'{which}KpointZeros') #cnt.condKpointZeros
#     NN = len(kpoints)
#     for i, key in enumerate(kpoints):                                         # name of the calculation
#         for kcuts, ecuts in zip(kpoints[key], energies[key]):   # mu index
#             count = 0
#             for ks, es in zip(kcuts, ecuts):                    # band index
#                 for k, e in zip(ks, es):                        # extrema index
#                     plt.text(k * invLfactor, e * efactor, f'{count}', ha="center", va="center")
#                     ax.scatter(k * invLfactor, e * efactor, s=250, color=mycolors(i,NN), alpha=0.3)
#                     count += 1


def excitonBands(cnt, ax=None):
    
    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    
    kgrid = cnt.kGrid
    ort = cnt.normOrt
    bz = cnt.bzHel

    dataFill = []
    dataPlot = []
    for mu1 in range(cnt.subN):
        for mu2 in range(cnt.subN):
            dataFill.append( cnt.excitonContinuum[mu1, mu2] )
            dataPlot.append( cnt.excitonBands[mu1, mu2] )
            #ax.fill_between(kgrid, *cnt.excitonContinuum[mu1, mu2], color='grey')
            # for band in cnt.excitonBands[mu1, mu2]:
            #     ax.plot(kgrid, band)
            #     print(type(ax))
    

    scrollPlot = ScrollPlots(ax, kgrid, dataFill, dataPlot)   
    fig.canvas.mpl_connect('scroll_event', scrollPlot.on_scroll)
    
    return fig, scrollPlot
    
    # ax.vlines(cnt.normOrt,0,efactor*maxEnergy,linestyles ="dashed", colors ="k")
    # ax.vlines(-cnt.normOrt,0,efactor*maxEnergy,linestyles ="dashed", colors ="k")


class ScrollPlots:
    def __init__(self, ax, kGrid, dataFill, dataPlot):
        self.index = 0
        self.dataFill = dataFill
        self.dataPlot = dataPlot
        self.kGrid = kGrid
        self.ax = ax
        # ax.fill_between(self.kGrid, *self.dataFill[self.index], color='grey')
        # ax.plot(self.kGrid, dataPlot[self.index][0])
        self.update()

    def on_scroll(self, event):
        #print(event.button, event.step)
        max_idx = len(self.dataFill)
        if event.button == 'up':
            self.index += 1
        else:
            self.index -= 1
        self.index = self.index % max_idx
        self.update()

    def update(self):
        self.ax.clear()
        self.ax.fill_between(self.kGrid, *self.dataFill[self.index], color='grey')
        for band in self.dataPlot[self.index]:
            self.ax.plot(self.kGrid, band)
        self.ax.set_title(f'Use scroll wheel to navigate\nindex {self.index}')
        plt.draw()




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
    # lines = PatchCollection(patches, edgecolor=ec, facecolor=ec)
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
