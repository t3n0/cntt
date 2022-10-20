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

cnt1.calculateCuttingLines(15)
cnt1.calculateElectronBands('TB', 'TB2+1', 'hel', gamma=2.0, fermi=1.0)
#cnt1.calculateElectronBands('TB', 'TB2', 'hel', gamma=2.0)

cnt1.calculateKpointValleys('TB2+1')

print(cnt1.valeEnergyZeros['TB2+1'][1])
#cnt1.calculateExcitonBands('EM','pollo','TB2+1')
#vale, cond = utils.condValeBands(cnt1.electronBandsHel['TB2+2'])
#print(cond)

#x, y, yy, mask = utils.findFunctionListExtrema(cond,'min')
#print(x, mask)

cnt1.plot()

#cnt1.setUnits('Ry', 'bohr')

#plt.recLat(cnt1)
#plt.electronBands(cnt1)
#plt.show()



# plt.recLat(cnt1)
# plt.electronBands(cnt1)
# plt.show()