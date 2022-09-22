import matplotlib as plt

def lattice(cnt, ax=None):
    ax.set_aspect("equal")
    minx, maxx, miny, maxy = utils.boundingRectangle(self.C, self.T, self.C + self.T, self.t1, self.t1 + self.T, self.t2, self.t2 + self.C / self.D,)
    hexDir = utils.hexPatches(minx, maxx, miny, maxy, self.a0, lat="dir")
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)