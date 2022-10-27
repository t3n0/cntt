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

# from swcnt.main import main

# main()

from swcnt.swcnt import Swcnt
import swcnt.plotting as plt
import swcnt.utils as utils

cnt1 = Swcnt(4,2)

cnt1.calculateCuttingLines(51)

cnt1.calculateElectronBands('TB', 'TB3', 'lin', gamma=3.0)

cnt1.calculateElectronBands('TB', 'TB3+1.5', 'hel', gamma=3.0, fermi=1.5)
cnt1.calculateElectronBands('TB', 'TB3', 'hel', gamma=3.0, fermi=0.0)

cnt1.calculateKpointValleys()

cnt1.calculateExcitonBands('EM','TB3', deltaK=10, bindEnergy=0.5)

cnt1.calculateDOS('electron', enSteps=100)

cnt1.plot()
