import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyqmc.api as pyq
eV = 27.2114

natom = 1

spin = 0
r = 1.4
nk = 40

kpt_unit = 2*np.pi/r


# get the homo and lumo
cell, mf = pyq.recover_pyscf(f'./r{r}/h1_r{r}_k{nk}_pbc_scf_vdz.chk',cancel_outputs=False)
homo = np.sort(np.asarray(mf.mo_energy_kpts).reshape(-1))[natom*nk-1]
lumo = np.sort(np.asarray(mf.mo_energy_kpts).reshape(-1))[natom*nk]
print('homo and lumo:', homo, lumo)

# plot the bands
fig, ax = plt.subplots(1,1,figsize=(5,4))

with h5py.File(f'./r{r}/h1_r{r}_k{nk}_pbc_bands_vdz.chk','r') as f:
  for iband in np.arange(f['energy'].shape[2]):
    plt.plot(f['kpts'][:,-1]/kpt_unit,f['energy'][spin, :, iband]*eV)

# plot the homo
plt.hlines(xmin=-0.5,xmax=0.5,y = homo*eV, color = 'grey', linestyle='dashed')

plt.xlim([-0.5,0.5])
plt.ylim([-20,30])

plt.xlabel(r'k/(2$\pi$/r)')
plt.ylabel(r'$E$/eV')

plt.savefig(f'h1_r{r}_band.pdf',bbox_inches='tight')
