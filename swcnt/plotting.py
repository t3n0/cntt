import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

def dirLat(cnt, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    # hexagons
    boundVectors = cnt.C, cnt.T, cnt.C + cnt.T, cnt.t1, cnt.t1 + cnt.T, cnt.t2, cnt.t2 + cnt.C / cnt.D
    minx, maxx, miny, maxy = boundingRectangle(*boundVectors)
    hexs = dirHexPatches(minx, maxx, miny, maxy, cnt.a0)
    # cells
    unitCell_la = np.array([[0.0, 0.0], cnt.C, cnt.C + cnt.T, cnt.T])
    unitCell_lh = np.array([[0.0, 0.0], cnt.t1, cnt.t1 + cnt.T, cnt.T])
    unitCell_ha = np.array([[0.0, 0.0], cnt.C / cnt.D, cnt.C / cnt.D + cnt.t2, cnt.t2])
    unitCells = cellPatches([unitCell_la, unitCell_ha, unitCell_lh], ["g", "b", "r"])
    # lattice vectors
    latVecs = arrowPatches(cnt.C, cnt.T, cnt.t1, cnt.t2, color='grey')
    # plot
    ax.add_collection(hexs)
    ax.add_collection(unitCells)
    ax.add_collection(latVecs)
    ax.set_aspect("equal")
    ax.set_xlabel(f'x ({cnt.unitL})')
    ax.set_ylabel(f'y ({cnt.unitL})')
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    return ax

def recLat(cnt, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    # hexagons
    boundVectors = cnt.k1L - 0.5*cnt.KT, \
        cnt.k1L + cnt.k2L- cnt.KT, cnt.k2L- 0.5*cnt.KT, \
        cnt.k1H + 0.5*cnt.NU/cnt.D*cnt.KT, \
        cnt.k1H + cnt.k2H + cnt.NU/cnt.D*cnt.KT, \
        cnt.k2H + 0.5*cnt.NU/cnt.D*cnt.KT
    minx, maxx, miny, maxy = boundingRectangle(*boundVectors)
    hexs = recHexPatches(minx, maxx, miny, maxy, cnt.b0)
    # cells
    recCell_lh = np.array([[0.0, 0.0], cnt.k1L, cnt.k1L + cnt.k2L, cnt.k2L]) - 0.5*cnt.KT
    recCell_ha = np.array([[0.0, 0.0], cnt.k1H, cnt.k1H + cnt.k2H, cnt.k2H]) + 0.5*cnt.NU/cnt.D*cnt.KT
    recCells = cellPatches([recCell_lh, recCell_ha], ["r", "b"])
    # lattice vectors
    #latVecs = arrowPatches(cnt.k1L, cnt.k2L, cnt.k1H, cnt.k2H, color='grey', d= cnt.KT)
    for sym in ['lin', 'hel']:
        if f'bzCuts{sym}' in cnt.data.keys():
            # plot cutting lines
            cuts = cnt.data[f'bzCuts{sym}']
            cutsPatches = linePatches(cuts[:, 0, 0], cuts[:, 0, 1], cuts[:, -1, 0] - cuts[:, 0, 0], cuts[:, -1, 1] - cuts[:, 0, 1], ec="r")
            ax.add_collection(cutsPatches)
    if hasattr(cnt, 'bandMinXy'):
        for mu in range(0, len(cnt.bandMinXy)):
            ax.plot(*cnt.bandMinXy[mu].T, "r.")
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

def subBands(cnt, attr, color='k', ax=None):
    if hasattr(cnt, attr):
        bands = getattr(cnt, attr)
        for mu in range(0, len(bands)):
            ax.plot(bands[mu,0,:], bands[mu,1,:], color=color)
            ax.plot(bands[mu,0,:], -bands[mu,1,:], color=color)
            ax.set_ylabel(f'Energy ({cnt.unitE})')
            ax.set_xlabel(f'k ({cnt.unitInvL})')

def excitonBands(cnt):
    pass

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
    width = np.max(vec)/150
    for v in vec:
        arrow = mpatches.FancyArrow(0,0,v[0],v[1], width=width, length_includes_head=True, color=color)
        patches.append(arrow)
    return PatchCollection(patches, match_original=True)


def cellPatches(cells, colors):
    patches = []
    for aux, c in zip(cells, colors):
        cell = mpatches.Polygon(aux, closed=True, color=c, alpha=0.2)
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
