# SWCNT tight binding band structure

[![GitHub Release Date](https://img.shields.io/github/release-date/t3n0/swcnt-bands)](https://github.com/t3n0/swcnt-bands/releases/latest)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/t3n0/swcnt-bands)](https://github.com/t3n0/swcnt-bands/releases/latest)
[![GitHub all releases](https://img.shields.io/github/downloads/t3n0/swcnt-bands/total)](https://github.com/t3n0/swcnt-bands/releases/latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

![swcnt42](./example42/cnt(4,2).png)

Utility to compute the electronic band structure of carbon nanotubes. It uses the well known tight binding approximation of CNTs and displays the band structure in linear and helical coordinates.

## Installation with `pip`

Simply download [.zip file](https://github.com/t3n0/swcnt-bands/releases/latest), extract it at your favourite location and run
```
pip install .
```
This will install the system-wide command `swcnt-bands` and the python package `swcnt`.

## Usage: command line

For a basic usage of the tool, simply type in the terminal
```
swcnt-bands 4 2
```
This will display the linear and helical band structure of a [(4,2) single-walled carbon nanotube](./example42/cnt(4,2).png), along with the most important physical parameters, unit cells and Brillouin zones.

Also, typing `swcnt-bands -h` provide a help dialoge for advanced usage.

## Usage: package

From a pyhton interpreter, import the `Swcnt` class. For a basic usage, just copy the following snippet:
```
import swcnt.swcnt as Swcnt

mycnt = Swcnt(4,2)
mycnt.calculateElectronBands()
mycnt.calculateExcitonBands()
mycnt.plot()
mycnt.plotExcitons()
```
## Support

For any problems, questions or suggestions, please contact me at tenobaldi@gmail.com.

## Roadmap

Currently the project only supports:
 - plotting the unit cells in three different configurations (the cnt supercell N-atom flake, the linear 2-atom cell and the helical 2-atom cell);
 - visualizing the corresponding graphene-like Brillouin zones;
 - computing and displaying the carbon nanotube band structure from the tight-binding zone-folding approximation;
 - computing and displaying the dispersion relation of bright and dark singlet excitons;

Future developments will include:
- optical matrix elements
- density of states


## Authors and acknowledgment

The development of SWCNT is proudly powered by [me](https://github.com/t3n0).
Also, please consider citing the relevant literature if you are going to use this tool:
 - [Carbon 186, 465-474 (2022)](https://doi.org/10.1016/j.carbon.2021.10.048)

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.
