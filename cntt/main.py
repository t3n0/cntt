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


def main():
    from matplotlib.pyplot import show
    import cntt.utils as utils
    from cntt.swcnt import Swcnt
    import os

    args = utils.getArgs()

    n, m = args.n, args.m

    cnt = Swcnt(n, m)
    cnt.calculateCuttingLines()

    cnt.calculateElectronBands('TB', 'TB', 'lin')
    cnt.calculateElectronBands('TB', 'TB', 'hel')

    cnt.calculateKpointValleys()

    cnt.calculateExcitonBands('EM','TB', deltaK=10, bindEnergy=0.2)

    cnt.calculateDOS('electron')
    cnt.calculateDOS('exciton')


    if args.outdir:
        WORK_DIR = os.getcwd()
        OUT_DIR = os.path.join(WORK_DIR, args.outdir)
        print(f'Current working directory: {WORK_DIR}')

        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
        print(f'Output directoriy: {OUT_DIR}')

        out_figure = os.path.join(OUT_DIR, f'cnt({n},{m}).png')
        cnt.plot(out_figure)
        #cnt.saveToDirectory(OUT_DIR)
    else:
        cnt.plot()
        show()
