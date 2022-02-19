# %%
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
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
spin = 0
natom = 1
r = 1.4
k_unit = 2*np.pi/(natom*r)

# %%
# Fat bands plot
fig, ax = plt.subplots(1,3,figsize=(12,3))
with h5py.File('../scf/r1.4/h1_r1.4_k40_pbc_bands_vdz.chk','r') as f:
  for iband in np.arange(5):
    norm = np.sum(abs(f['eigvecs'][spin, :, :, iband])**2, axis=1)
    proj_s = np.sum(abs(f['eigvecs'][spin, :, :2, iband])**2, axis = 1)/norm
    proj_pxy = np.sum(abs(f['eigvecs'][spin, :, 2:4, iband])**2, axis = 1)/norm
    proj_pz = abs(f['eigvecs'][spin, :, -1, iband])**2/norm

    for (i_proj,proj),proj_orb in\
        zip(enumerate([proj_s, proj_pxy, proj_pz]), ['s','px,py','pz']):
      im = ax[i_proj].scatter(f['kpts'][:,-1]/k_unit, f['energy'][spin, :,iband]*eV,
      c = proj, cmap = 'Reds', vmax = 1, vmin = 0)
      ax[i_proj].plot(f['kpts'][:,-1]/k_unit, f['energy'][spin, :,iband].real*eV,'grey')
      ax[i_proj].hlines(xmin = -0.5, xmax = 0.5, y = ef, color='r', linestyle='dashed')
      ax[i_proj].set_xlim([-0.5,0.5])
      ax[i_proj].set_ylim([-20,200])
      ax[i_proj].set_title(f'g = {proj_orb}')
      

ax[0].set_ylabel(r'$E$/eV')
for i in [0,1,2]: ax[i].set_xlabel(r'$k/(\frac{2\pi}{r})$')
plt.tight_layout()
cbar = plt.colorbar(im, ax = ax)
cbar.ax.set_title(r'$|\langle g|\psi_{nk}\rangle|^2$')

plt.savefig('./h1_fatbands.pdf',bbox_inches='tight')


