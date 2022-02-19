# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import pyqmc.api as pyq
eV = 27.2114

# %%
cell, mf = pyq.recover_pyscf('../scf/r1.4/h1_r1.4_k40_pbc_scf_vdz.chk')
nkpts = len(mf.kpts)
natom = 1
homo = np.sort(np.asarray(mf.mo_energy_kpts).reshape(-1))[nkpts*natom-1]
lumo = np.sort(np.asarray(mf.mo_energy_kpts).reshape(-1))[nkpts*natom]
ef = (homo+lumo)/2*eV

# %%
r = 1.4
natom = 1
k_unit = 2*np.pi/(natom*r)

projections = ['s','px_py']
fig, ax = plt.subplots(1,len(projections),figsize=(5*len(projections),4))

for ip, proj in enumerate(projections):
  path = f"../{proj}"
  with h5py.File(f"{path}/wannier_bands.chk",'r') as f_w,\
    h5py.File(f"../scf/r1.4/h1_r1.4_k40_pbc_bands_vdz.chk",'r') as f:
    ax[ip].plot(f_w['kpts'][:,-1], f_w['eigvals'][...]*eV, label = f'wannier interpolated band')
    ax[ip].plot(f['kpts'][:,-1]/k_unit, f['energy'][0,:]*eV, color = 'k')
    ax[ip].set_ylim([-20,80])
    # plot the Fermi level
  ax[ip].hlines(xmin = -0.5, xmax = 0.5, y = ef, color='r', linestyle='dashed')
  #ax[ip].legend(loc='lower left')
  ax[ip].set_xlim([-0.5,0.5])
  ax[ip].set_xlabel(r'$k/(\frac{2\pi}{r})$')
  ax[ip].set_title(f"trial projection = {proj}")

ax[0].set_ylabel(r'$E$/eV')
plt.savefig(f'wannier_bands.pdf',bbox_inches='tight')


