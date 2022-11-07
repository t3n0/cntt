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


from cntt.utils import runProcess
from textwrap import dedent
import os


def pwInputFile(**kwargs):

    WORK_DIR   = os.getcwd()
    DFT_DIR    = os.path.join(WORK_DIR, kwargs['flag'])
    PSEUDO_DIR = os.path.join(WORK_DIR, kwargs['pseudo_dir'])
    OUT_DIR    = os.path.join(WORK_DIR, DFT_DIR, 'outdir')
    
    if not os.path.exists(DFT_DIR):
        os.makedirs(DFT_DIR)

    text = f'''\
    &control
        calculation = '{kwargs['calc']}'
        restart_mode = 'from_scratch',
        prefix = '{kwargs['flag']}',
        pseudo_dir = '{PSEUDO_DIR}',
        outdir = '{OUT_DIR}'
    /

    &system
        ibrav = 0,
        nat = 2,
        ntyp = 1,
        ecutwfc = {kwargs['ecutwfc']},
        ecutrho = {kwargs['ecutrho']},
        nbnd = {kwargs['nbnd']},
        occupations = 'smearing',
        smearing = 'mv',
        degauss = 0.001
    /

    &electrons
        diagonalization = 'david'
        mixing_mode = 'plain'
        mixing_beta = 0.7
        conv_thr = 1.0d-8
    /
    
    ATOMIC_SPECIES
        C  12.011  C.pbe-n-kjpaw_psl.1.0.0.UPF
    
    CELL_PARAMETERS (angstrom)
        {10*kwargs['a1'][0]} {10*kwargs['a1'][1]} 0.0
        {10*kwargs['a2'][0]} {10*kwargs['a2'][1]} 0.0
        0.0 0.0 {kwargs['clat']}

    ATOMIC_POSITIONS (angstrom)
        C   {10*kwargs['A'][0]}    {10*kwargs['A'][1]}   0.0
        C   {10*kwargs['B'][0]}    {10*kwargs['B'][1]}   0.0

    '''
    
    if kwargs['calc'] == 'scf':
        FILE_NAME = os.path.join(WORK_DIR, DFT_DIR, '1-scf.in')
        kpoints = f'''\
        K_POINTS automatic
                {kwargs['kpoints']} {kwargs['kpoints']} 1 0 0 0
        '''
        text += dedent(kpoints)
    elif kwargs['calc'] == 'bands':
        FILE_NAME = os.path.join(WORK_DIR, DFT_DIR, '2-bands.in')
        kpoints = f'''\
        K_POINTS tpiba_b
                2
                0.577350269 0.333333333 0.0 20
                0.0         0.0         0.0 0
        '''
        text += dedent(kpoints)

    with open(FILE_NAME, 'w') as f:
        f.write(dedent(text))
    
    return FILE_NAME


def scfCalculation(cnt, name, sym, pseudo_dir = 'pseudo_dir', ecutwfc = 20, ecutrho = 200, nbnd = 8, clat = 10, kpoints = 5):
    if sym == 'hel':
        a1 = cnt.C/cnt.D
        a2 = cnt.t2
        A = cnt.atomAhel
        B = cnt.atomBhel
    elif sym == 'lin':
        a1 = cnt.T
        a2 = cnt.t2
        A = cnt.atomAlin
        B = cnt.atomBlin
    else:
        print(f'Symmetry {sym} not recognised.')
    
    inputFile = pwInputFile(calc = 'scf',
                flag = name,
                pseudo_dir = pseudo_dir,
                ecutwfc = ecutwfc,
                ecutrho = ecutrho,
                nbnd = nbnd,
                a1 = a1,
                a2 = a2,
                clat = clat,
                A = A,
                B = B,
                kpoints = kpoints)

    process = runProcess('mpirun -n 4 pw.x', inputFile)
    print(process.stdout)


def dftElectronBands(cnt, name, sym, **kwargs):
    scfCalculation(cnt, name, sym, **kwargs)