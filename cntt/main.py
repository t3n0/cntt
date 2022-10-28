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

def main():
    from matplotlib.pyplot import show
    import cntt.utils as utils
    from cntt.swcnt import Swcnt
    import os

    args = utils.getArgs()

    n, m = args.n, args.m
    cnt = Swcnt(n, m)

    cnt.calculateCuttingLines(31)
    cnt.calculateElectronBands('TB', 'TB3', sym='hel', gamma=3.0)
    cnt.calculateElectronBands('TB', 'TB3', sym='lin', gamma=3.0)

    if args.outdir:
        WORK_DIR = os.getcwd()
        OUT_DIR = os.path.join(WORK_DIR, args.outdir)
        print(f'Current working directory:\n\t{WORK_DIR}')

        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
        print(f'Output directoriy:\n\t{OUT_DIR}')

        out_figure = os.path.join(OUT_DIR, f'cnt({n},{m}).png')
        cnt.plot(out_figure)
        cnt.saveToDirectory(OUT_DIR)
    else:
        cnt.plot()
        show()
