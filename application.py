#!/usr/bin/env python
from ase import io, units, Atom
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.constraints import FixAtoms
from ase.calculators.neighborlist import NeighborList
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from amp import Amp
from amp.descriptor.gaussian import FingerprintCalculator
from espresso import espresso

from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from keras.models import model_from_json, Sequential, model_from_json
from keras.layers import Dense, Activation
from keras import optimizers, regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from scipy.interpolate import RegularGridInterpolator
import numpy as np
import copy
import os,time

def make_symmetry_functions( elements):
    G = {}
    for element0 in elements:
        etas = [0.003214*42.25, 0.035711*42.25, 0.071421*42.25, 0.124987*42.25, 0.214264*42.25, 0.357106*42.25, 0.714213*42.25, 1.428426*42.25]
        _G = [{'type': 'G2', 'element': element, 'eta': eta}
              for eta in etas
              for element in elements]

        etas = [0.000357*42.25, 0.028569*42.25, 0.089277*42.25]
        zetas = [1., 2., 4.]
        gammas = [+1., -1.]
        for eta in etas:
            for zeta in zetas:
                for gamma in gammas:
                    for i1, el1 in enumerate(elements):
                        for el2 in elements[i1:]:
                            els = sorted([el1, el2])
                            _G.append({'type': 'G4',
                                       'elements': els,
                                       'eta': eta,
                                       'gamma': gamma,
                                       'zeta': zeta})
        G[element0] = _G
    return G

def get_fingerprints(atoms):
    elements =['O','Ru']
    Gs = make_symmetry_functions(elements)
    fp = np.zeros([2, len(Gs['O'])])
    cutoff ={'name': 'Cosine', 'kwargs': {'Rc': 6.5}}
    cutoff_globle = 6.5
    _nl = NeighborList(cutoffs=([cutoff_globle / 2.] *
                                len(atoms)),
                       self_interaction=False,
                       bothways=True,
                       skin=0.)
    _nl.update(atoms)

    amp_obj = FingerprintCalculator(neighborlist=_nl, Gs=Gs, cutoff=cutoff,fortran=False)
    amp_obj.initialize(fortran=False,atoms=atoms)
    n=0
    fingerprint = []
    for index in range(12,14):
        symbol = atoms[index].symbol
        neighborindices, neighboroffsets = _nl.get_neighbors(index)
        neighborsymbols = [atoms[_].symbol for _ in neighborindices]
        neighborpositions = [atoms.positions[neighbor] + np.dot(offset, atoms.cell)
                             for (neighbor, offset) in zip(neighborindices,
                                                           neighboroffsets)]
        indexfp = amp_obj.get_fingerprint(
            index, symbol, neighborsymbols, neighborpositions)
        fp[n] = indexfp[1]
        n = n+1
    return fp

def get_frictions(atoms, O_elec_den):
    eta_ph = 0.35/1000/units.fs
    frict = np.array([eta_ph for i in range(len(atoms))])
    rs = (3./(4.*np.pi * O_elec_den))**(1./3.)
    eta = 5.07895* rs**(-3.73291)*np.exp(0.251941*rs)
    M_O = 16
    [[eta12], [eta13]] = eta * (units._me/units._amu/M_O)/(units._hbar/units.Hartree*units.J*10**15)/units.fs
    frict[12], frict[13]= eta12, eta13
    return frict

atoms = io.read('~/relaxed_surf/F_100/InitialGeom.traj')

calc_espresso=espresso(pw=500,
                       dw=5000,
                       kpts=(6,6,1),
                       dipole={'status':True},
                       outdir='./',
                       mode = 'scf',
                       xc='BEEF-vdW',
                       nbands=-30,
                       spinpol=False,
                       occupations = 'smearing', # 'smearing', 'fixed', 'tetrahedra'           
                       smearing = 'fd',
                       sigma = 0.1,
                       output = {'avoidio':False,
                                 'removewf':True,
                                 'wf_collect':False},
                       convergence = {'energy':1e-6*13.6, # in eV    
                                      'mixing':0.2,
                                      'maxsteps':500,
                                      'diag':'david'},
                       parflags='-npool 1',
                       tprnfor = True,
                       calcstress=True,
                       tstress = True,
                       nosym=True)

calc_bare=espresso(pw=500,
                   dw=5000,
                   kpts=(6,6,1),
                   dipole={'status':True},
                   outdir='./',
                   mode = 'scf',
                   xc='BEEF-vdW',
                   nbands=-30,
                   spinpol=False,
                   occupations = 'smearing', # 'smearing', 'fixed', 'tetrahedra'      
                   smearing = 'fd',
                   sigma = 0.1,
                   output = {'avoidio':False,
                             'removewf':True,
                             'wf_collect':False},
                   convergence = {'energy':1e-6*13.6, # in eV                                 
                                  'mixing':0.2,
                                  'maxsteps':500,
                                  'diag':'david'},
                   parflags='-npool 1',
                   tprnfor = True,
                   calcstress=True,
                   tstress = True,
                   nosym=True)


c = FixAtoms(indices=[atom.index for atom in atoms if atom.position[2]<8])
atoms.set_constraint(c)

identified_images = Trajectory('identified_images.traj','a', properties = ['energy','forces'])
traj_md = Trajectory('md.traj','a', properties=['energy','forces'])

path = ['~/non_adiabatic/Langevin/Training_3nd/sym70_2L70/newriver/amp-checkpoint.amp',
        '~/non_adiabatic/Langevin/Training_3nd/sym70_2L65/newriver/amp-checkpoint.amp',
        '~/non_adiabatic/Langevin/Training_3nd/sym70_2L60/newriver/amp-checkpoint.amp',
        ]

fingerprints = np.loadtxt('~/non_adiabatic/Langevin/Training_3nd/trajs/fingerprints/total_fingerp.txt')
scaler_fp = MinMaxScaler(feature_range=(-1,1), copy=True)
scaler_fp.fit(fingerprints)
scaled_fp = scaler_fp.transform(fingerprints)

atoms_chg = io.read('~/non_adiabatic/Langevin/Training_3nd/trajs/fingerprints/identified.traj',index=':')
chg = np.zeros(len(atoms_chg)*2)
i = 0
for atom in atoms_chg:
    for index in range(12,14):
        chg[i]=atom.get_charges()[index]
        i += 1
scaler_chg = MinMaxScaler(feature_range=(-1,1), copy=True)
scaler_chg.fit(chg.reshape(-1,1))
scaled_chg = scaler_chg.transform(chg.reshape(-1,1))

X_train, X_test, Y_train, Y_test = train_test_split(scaled_fp[:], scaled_chg[:], test_size = 0.2, random_state = None)
x_train, x_validation, y_train, y_validation = train_test_split(X_train, Y_train, test_size = 0.2, random_state = None)
model = Sequential()
model.add(Dense(100, input_dim= len(scaled_fp[0]), init='glorot_normal', activation='tanh'))
model.add(Dense(100, init='glorot_normal', activation='tanh'))
model.add(Dense(1, init='glorot_normal'))
model.compile(loss='mean_squared_error', optimizer='adam')
callbacks = [EarlyStopping(monitor='val_loss', min_delta=1e-06, patience=20, verbose=0, mode='auto'), ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True, verbose=0), ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001, mode='auto')]
model.fit(x_train, y_train, epochs = 1000, batch_size=10, validation_data=(x_validation, y_validation), callbacks=callbacks)

RMSE_tr = np.sqrt(np.mean((y_train.ravel() - model.predict(x_train))**2))
RMSE_te = np.sqrt(np.mean((Y_test.ravel() - model.predict(X_test))**2))
np.savetxt('para_rawdata.txt', np.column_stack((RMSE_tr, RMSE_te)))

Temp_profile = np.loadtxt('~/non_adiabatic/Langevin/2TempModel/F_100/tempfinal.txt')
MaxwellBoltzmannDistribution(atoms, 300 * units.kB) # initialize         

for i in range(1000):
    log = open('log.txt', 'a')    
    energy = []
    T_el = Temp_profile[i][2]

    start = time.time()
    for j in range(len(path)):
        calc_NN = Amp.load(path[j],cores=1)
        energy.append(calc_NN.get_potential_energy(atoms))    
    rmse = np.average(np.std(energy, axis=0))

    if rmse <0.03:
        atoms.set_calculator(Amp.load(path[0],cores=1))
        fp_Oxy = scaler_fp.transform(get_fingerprints(atoms))
        O_elec_den = scaler_chg.inverse_transform(model.predict(fp_Oxy))
        frict = get_frictions(atoms, O_elec_den)
        log.write("step %d, time %11.4f fs, rmse %11.4f eV, T_el %f, friction %f and %f, using NNs. " % (i, Temp_profile[i][0], rmse, T_el, frict[-2], frict[-1]))

    else:
        atoms.set_calculator(calc_espresso)
        atoms.get_potential_energy()

        bare_surf = atoms.copy()
        del(bare_surf[-1], bare_surf[-1])
        calc_bare_surf=copy.deepcopy(calc_bare)
        bare_start = time.time()
        bare_surf.set_calculator(calc_bare_surf)
        bare_surf.get_potential_energy()
        bare_end = time.time()
        n = calc_bare_surf.extract_charge_density(spin='both')[2]
        nx, ny, nz = n.shape
        x = np.linspace(0,1,nx+1)
        y = np.linspace(0,1,ny+1)
        z = np.linspace(0,1,nz+1)
        interp_func = RegularGridInterpolator((x,y,z),np.tile(n,(2,2,2))[0:nx+1,0:ny+1,0:nz+1])
        pos = atoms.get_scaled_positions()
        identified_images.write(atoms, charges = interp_func(pos))

        O_elec_den = np.array([interp_func(pos[-2]), interp_func(pos[-1])])
        frict = get_frictions(atoms, O_elec_den)
        log.write("step %d, time %11.4f fs, rmse %11.4f eV, T_el  %f,friction %f and %f using DFT, bare_surf %f " % (i, Temp_profile[i][0], rmse, T_el, frict[-2], frict[-1], (bare_end - bare_start)))

    dyn = Langevin(atoms, 2 * units.fs, T_el * units.kB, frict)
    dyn.run(steps=1)
    end = time.time()
    log.write("using time %f .\n" %(end-start))
    log.close()
    traj_md.write(atoms)
try:
    os.system('rm -r amp-fingerprint-primes.ampdb')
    os.system('rm -r amp-fingerprints.ampdb')
    os.system('rm -r amp-neighborlists.ampdb')
except:
    print 'finished'
