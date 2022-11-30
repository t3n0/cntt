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
from cntt.mathematics import fourierInterpolation
from textwrap import dedent
import numpy as np
import os
import matplotlib.pyplot as plt


def parseScf(text: str):
    for line in text.splitlines():
        if 'the Fermi energy is' in line:
            fermi = line.split()[-2]
    return float(fermi)


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
    C   0.6666666666666667    0.3333333333333333   0.0
    C   0.3333333333333333    0.6666666666666667   0.0

    '''
    return dedent(text)


def pwxInputFile(**kwargs):

    WORK_DIR  = os.getcwd()
    DFT_DIR   = os.path.join(WORK_DIR, 'dft', kwargs['flag'])
    FILE_IN = os.path.join(DFT_DIR, kwargs['filename']+'.in')
    FILE_OUT  = os.path.join(DFT_DIR, kwargs['filename']+'.out')
    FILE_ERR = os.path.join(DFT_DIR, kwargs['filename']+'.err')
    if not os.path.exists(DFT_DIR):
        os.makedirs(DFT_DIR)
    kwargs['pseudo_dir'] = os.path.join(WORK_DIR, kwargs['pseudo_dir'])
    kwargs['outdir']     = os.path.join(WORK_DIR, DFT_DIR, 'outdir')

    text = espressoBaseTxt(**kwargs) + kwargs['kpointCard']
    with open(FILE_IN, 'w') as f:
            f.write(text)
    return FILE_IN, FILE_OUT, FILE_ERR


def bandsxInputFile(flag, filename, filebands):
    WORK_DIR  = os.getcwd()
    DFT_DIR   = os.path.join(WORK_DIR, 'dft', flag)
    FILE_IN = os.path.join(DFT_DIR, filename+'.in')
    FILE_OUT = os.path.join(DFT_DIR, filename+'.out')
    FILE_ERR = os.path.join(DFT_DIR, filename+'.err')
    FILE_BANDS  = os.path.join(DFT_DIR, filebands)
    OUTDIR    = os.path.join(DFT_DIR, 'outdir')
    if not os.path.exists(DFT_DIR):
        os.makedirs(DFT_DIR)
    text = f'''\
        &BANDS
        prefix = '{flag}',
        outdir = '{OUTDIR}',
        filband = '{FILE_BANDS}'
        /
    '''
    
    with open(FILE_IN, 'w') as f:
            f.write(dedent(text))
    return FILE_IN, FILE_BANDS, FILE_OUT, FILE_ERR


def scfCalculation(cnt, name, pseudo_dir = 'pseudo_dir', ecutwfc = 20, ecutrho = 200, nbnd = 8, clat = 1, kpoints = 12, nprocs = 1):

    kpointCard = f'K_POINTS automatic\n    {kpoints} {kpoints} 1 0 0 0\n'

    inputFile, outputFile, errFile = pwxInputFile(
        calc = 'scf',
        flag = name,
        filename = '1-scf',
        pseudo_dir = pseudo_dir,
        ecutwfc = ecutwfc,
        ecutrho = ecutrho,
        nbnd = nbnd,
        alat = cnt.a0 / Bohr2nm,
        clat = clat / Bohr2nm,
        kpointCard = kpointCard)

    print('Start scf calculation: ... ', end='', flush=True)
    process = runProcess(f'mpirun -n {nprocs} pw.x', inputFile)
    if process.returncode == 0:
        print('DONE.')
        fermi = parseScf(process.stdout)
    else:
        print(f'Failed with return code {process.returncode}.')
        with open(errFile, 'w') as f:
            f.write(dedent(process.stderr))
    with open(outputFile, 'w') as f:
            f.write(dedent(process.stdout))
    
    return fermi


def bandCalculation(cnt, name, pseudo_dir = 'pseudo_dir', ecutwfc = 20, ecutrho = 200, nbnd = 8, clat = 1, mu = 0, nprocs = 1):
    
    kpointCard = kPointsPathIbrav4(cnt, mu)

    inputFile, outputFile, errFile = pwxInputFile(
        calc = 'bands',
        flag = name,
        filename = f'2-bands-{mu:02d}',
        pseudo_dir = pseudo_dir,
        ecutwfc = ecutwfc,
        ecutrho = ecutrho,
        nbnd = nbnd,
        alat = cnt.a0 / Bohr2nm,
        clat = clat / Bohr2nm,
        kpointCard = kpointCard)

    print(f'Start mu={mu} bands calculation: ... ', end='', flush=True)
    process = runProcess(f'mpirun -n {nprocs} pw.x', inputFile)
    if process.returncode == 0:
        print('DONE.')
    else:
        print(f'Failed with return code {process.returncode}.')
        with open(errFile, 'w') as f:
            f.write(dedent(process.stderr))
    with open(outputFile, 'w') as f:
            f.write(dedent(process.stdout))


def bandsXCalculation(name, mu = 0, nprocs = 1):

    inputFile, bandFile, outputFile, errFile = bandsxInputFile(
        name,
        filename = f'3-bandsx-{mu:02d}',
        filebands = f'bands-{mu:02d}.txt')

    print(f'Start mu={mu} k-path calculation: ... ', end='', flush=True)
    process = runProcess(f'mpirun -n {nprocs} bands.x', inputFile)
    if process.returncode == 0:
        print('DONE.')
    else:
        print(f'Failed with return code {process.returncode}.')
        with open(errFile, 'w') as f:
            f.write(dedent(process.stderr))
    with open(outputFile, 'w') as f:
            f.write(dedent(process.stdout))

    return np.loadtxt(bandFile + '.gnu').T


def dftElectronBands(cnt, name, from_file = False, fourier_interp = False, pseudo_dir = 'pseudo_dir', ecutwfc = 20, ecutrho = 200, nbnd = 8, clat = 1, kpoints = 12, deltan = 1, nprocs = 1):

    attrBands = 'electronBandsHel'
    _, ksteps, _ = cnt.bzCutsHel.shape
    bz = cnt.bzHel
    bands = np.zeros( (cnt.D, 2*deltan, 2, ksteps) ) # bands = E_n^mu(k), bands[mu index, n index, k/energy index, grid index]
    bands[:,:,0,:] = bz

    if from_file:
        WORK_DIR  = os.getcwd()
        outputFile  = os.path.join(WORK_DIR, 'dft', name, f'1-scf.out')
        with open(outputFile, 'r') as f:
            text = f.read()
        fermi = parseScf(text)
    else:
        fermi = scfCalculation(cnt, name, pseudo_dir = pseudo_dir, ecutwfc = ecutwfc, ecutrho = ecutrho, nbnd = nbnd, clat = clat, kpoints = kpoints, nprocs = nprocs)
    for mu in range(cnt.D):
        if from_file:
            WORK_DIR  = os.getcwd()
            bandFile  = os.path.join(WORK_DIR, 'dft', name, f'bands-{mu:02d}.txt.gnu')
            band = np.loadtxt(bandFile).T
        else:
            bandCalculation(cnt, name, pseudo_dir = pseudo_dir, ecutwfc = ecutwfc, ecutrho = ecutrho, nbnd = nbnd, clat = clat, mu = mu, nprocs = nprocs)
            band = bandsXCalculation(name, mu = mu, nprocs = nprocs)
        dftKgrid = band[0].reshape(nbnd,-1)[0]
        # print(len(dftKgrid))
        # dftKgrid = np.array(list(dict.fromkeys(band[0])))
        # print(len(dftKgrid))
        dftKgrid = (dftKgrid - np.max(dftKgrid)/2) * np.pi * 2 / cnt.a0
        if abs(dftKgrid[0]-bz[0]) + abs(dftKgrid[-1]-bz[-1]) > 1e-2:
            print('Warning: dft BZ mismatch.')
            print('    ', dftKgrid[0], bz[0], abs(dftKgrid[0]-bz[0]))
            print('    ', dftKgrid[-1], bz[-1], abs(dftKgrid[-1]-bz[-1]))
        dftKsteps = len(dftKgrid)
        dftBz = np.linspace(bz[0], bz[-1], dftKsteps)
        dftEgrids = band[1].reshape(nbnd,-1)
        dftEgrids = np.roll(dftEgrids, len(dftEgrids[0])//2, axis=1) - fermi
        for i in range(nbnd//2-deltan, nbnd//2+deltan):
            if len(bz) > len(dftBz) and fourier_interp:
                print(f'Performing Fourier interpolation: kpoints {len(dftBz)} -> {len(bz)}')
                egrid = fourierInterpolation(dftBz[:-1], dftEgrids[i][:-1], bz[:-1])
                egrid = np.append(egrid, dftEgrids[i][-1])
            else:
                #print(f'Performing linear interpolation: kpoints {len(dftBz)} -> {len(bz)}')
                egrid = np.interp(bz, dftBz, dftEgrids[i])
            bands[mu,i-nbnd//2+deltan,1,:] = egrid
    getattr(cnt, attrBands)[name] = bands

