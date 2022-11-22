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
from cntt.physics import Bohr2nm
from textwrap import dedent
import numpy as np
import os


def kPointsPathIbrav4(cnt, mu):
    # maybe there is a more efficient/pythonic/elegant way to do this
    # but i cannot be bothered, go to hell stupid brillouin zone!
    n, m, D, u, v = cnt.n, cnt.m, cnt.D, cnt.u2, cnt.v2
    k2 = np.array([v/D, u/D])
    length = np.linalg.norm([n/D,m/D])
    ksteps = cnt.kStepsHel
    dk = length/ksteps
    
    x1 = [i*m/n for i in range(n//D+1)]
    x2 = [float(i) for i in range(m//D+1)]
    xs = np.concatenate((x1,x2))
    xs = np.sort(list(dict.fromkeys(xs)))
    
    y1 = [i*n/m for i in range(m//D+1)]
    y2 = [float(i) for i in range(n//D+1)]
    ys = np.concatenate((y1,y2))
    ys = np.sort(list(dict.fromkeys(ys)))
    
    points = np.zeros((len(xs),2))
    points[:,0] = xs
    points[:,1] = ys

    cuts = []
    for i in range(len(points)-1):
        cuts.append([points[i], points[i+1]])
        delta = np.floor(points[i+1])
        points = points - delta # Satan hides in here, DONT use -= operator!
    cuts = np.array(cuts).reshape(-1,2) + k2 * mu

    text = f'K_POINTS crystal_b\n    {len(cuts)}\n'
    for i in range(0,len(cuts)-1,2):
        segment = np.linalg.norm(cuts[i+1] - cuts[i])
        text += f'    {cuts[i,0]} {cuts[i,1]} {0.0} {int(segment/dk)}\n    {cuts[i+1,0]} {cuts[i+1,1]} {0.0} {0}\n'
    
    return text


def espressoBaseTxt(**kwargs):
    text = f'''\
    &control
        calculation = '{kwargs['calc']}'
        restart_mode = 'from_scratch',
        prefix = '{kwargs['flag']}',
        pseudo_dir = '{kwargs['pseudo_dir']}',
        outdir = '{kwargs['outdir']}'
    /

    &system
        ibrav = 4,
        celldm(1) = {kwargs['alat']},
        celldm(3) = {kwargs['clat']},
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
    return dedent(text)


def pwxInputFile(**kwargs):

    WORK_DIR  = os.getcwd()
    DFT_DIR   = os.path.join(WORK_DIR, kwargs['flag'])
    FILE_NAME = os.path.join(WORK_DIR, DFT_DIR, kwargs['filename'])
    if not os.path.exists(DFT_DIR):
        os.makedirs(DFT_DIR)
    kwargs['pseudo_dir'] = os.path.join(WORK_DIR, kwargs['pseudo_dir'])
    kwargs['outdir']     = os.path.join(WORK_DIR, DFT_DIR, 'outdir')    
    text = espressoBaseTxt(**kwargs) + kwargs['kpointCard']
    with open(FILE_NAME, 'w') as f:
            f.write(text)
    return FILE_NAME


def bandsxInputFile(prefix, outdir, filband):
    text = f'''\
        &BANDS
        prefix = '{prefix}',
        outdir = '{outdir}',
        filband = '{filband}'
        /'''


def scfCalculation(cnt, name, pseudo_dir = 'pseudo_dir', ecutwfc = 20, ecutrho = 200, nbnd = 8, clat = 10, kpoints = 5):

    kpointCard = f'K_POINTS automatic\n    {kpoints} {kpoints} 1 0 0 0\n'

    inputFile = pwxInputFile(
        calc = 'scf',
        flag = name,
        filename = '1-scf.in',
        pseudo_dir = pseudo_dir,
        ecutwfc = ecutwfc,
        ecutrho = ecutrho,
        nbnd = nbnd,
        alat = cnt.a0 / Bohr2nm,
        clat = clat,
        kpointCard = kpointCard)

    # print('Start scf calculation: ... ', end='', flush=True)
    # process = runProcess('mpirun -n 4 pw.x', inputFile)
    # if process.returncode == 0:
    #     print('DONE.')
    # else:
    #     print(f'Failed with return code {process.returncode}.')
    # todo
    # parse information (eg the fermi level) from the process.output
    # print(process.stdout)
    # print(process.stderr)


def bandCalculation(cnt, name, pseudo_dir = 'pseudo_dir', ecutwfc = 20, ecutrho = 200, nbnd = 8, clat = 10, mu = 0):
    
    kpointCard = kPointsPathIbrav4(cnt, mu)

    inputFile = pwxInputFile(
        calc = 'bands',
        flag = name,
        filename = f'2-bands-{mu:02d}.in',
        pseudo_dir = pseudo_dir,
        ecutwfc = ecutwfc,
        ecutrho = ecutrho,
        nbnd = nbnd,
        alat = cnt.a0 / Bohr2nm,
        clat = clat,
        kpointCard = kpointCard)

    # print('Start bands calculation: ... ', end='', flush=True)
    # process = runProcess('mpirun -n 4 pw.x', inputFile)
    # # process = runProcess('/home/tentacolo/quantum-espresso/qe-7.1-serial/bin/pw.x', inputFile)
    # if process.returncode == 0:
    #     print('DONE.')
    # else:
    #     print(f'Failed with return code {process.returncode}.')
    # print(process.stdout)
    # print(process.stderr)

def dftElectronBands(cnt, name, **kwargs):

    scfCalculation(cnt, name, **kwargs)
    for mu in range(cnt.D):
        bandCalculation(cnt, name, mu = mu, **kwargs)
