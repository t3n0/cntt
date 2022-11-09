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
        ibrav = 12,
        a = {kwargs['alat']}
        b = {kwargs['blat']}
        c = {kwargs['clat']}
        cosAB = {kwargs['cosAB']}
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


    ATOMIC_POSITIONS crystal
        C   0.333333333333333    0.3333333333333333   0.0
        C   0.666666666666666    0.6666666666666666   0.0

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
        K_POINTS crystal_b
                2
                0.0000000000         0.000000000         0.000000000         {kwargs['kbands']}
                0.0000000000         1.000000000         0.000000000         0
        '''
        text += dedent(kpoints)

    with open(FILE_NAME, 'w') as f:
        f.write(dedent(text))
    
    return FILE_NAME


def scfCalculation(cnt, name, sym, pseudo_dir = 'pseudo_dir', ecutwfc = 20, ecutrho = 200, nbnd = 8, clat = 10, kpoints = 5):
    if sym == 'hel':
        alat = 10 * cnt.normCD
        blat = 10 * cnt.normt2
        cosAB = cnt.cosCt2
    elif sym == 'lin':
        alat = 10 * cnt.normT
        blat = 10 * cnt.normt1
        cosAB = cnt.cosTt1
    else:
        print(f'Symmetry {sym} not recognised.')
    
    inputFile = pwInputFile(calc = 'scf',
                flag = name,
                pseudo_dir = pseudo_dir,
                ecutwfc = ecutwfc,
                ecutrho = ecutrho,
                nbnd = nbnd,
                alat = alat,
                blat = blat,
                clat = clat,
                cosAB = cosAB,
                kpoints = kpoints)

    print('Start scf calculation: ... ', end='', flush=True)
    process = runProcess('mpirun -n 4 pw.x', inputFile)
    if process.returncode == 0:
        print('DONE.')
    else:
        print(f'Failed with return code {process.returncode}.')

def bandCalculation(cnt, name, sym, pseudo_dir = 'pseudo_dir', ecutwfc = 20, ecutrho = 200, nbnd = 8, clat = 10):
    if sym == 'hel':
        alat = 10 * cnt.normCD
        blat = 10 * cnt.normt2
        cosAB = cnt.cosCt2
        kbands = len(cnt.bzCutsHel[0])
    elif sym == 'lin':
        alat = 10 * cnt.normT
        blat = 10 * cnt.normt1
        cosAB = cnt.cosTt1
        kbands = len(cnt.bzCutsHel[0])
    else:
        print(f'Symmetry {sym} not recognised.')
    
    inputFile = pwInputFile(calc = 'bands',
                flag = name,
                pseudo_dir = pseudo_dir,
                ecutwfc = ecutwfc,
                ecutrho = ecutrho,
                nbnd = nbnd,
                alat = alat,
                blat = blat,
                clat = clat,
                cosAB = cosAB,
                kbands = kbands)
    print('Start bands calculation: ... ', end='', flush=True)
    process = runProcess('mpirun -n 4 pw.x', inputFile)
    if process.returncode == 0:
        print('DONE.')
    else:
        print(f'Failed with return code {process.returncode}.')

def dftElectronBands(cnt, name, sym, **kwargs):
    scfCalculation(cnt, name, sym, **kwargs)
    bandCalculation(cnt, name, sym, **kwargs)
