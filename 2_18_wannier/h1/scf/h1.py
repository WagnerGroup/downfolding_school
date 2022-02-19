import numpy as np
import h5py
from pyscf.pbc import df as pdf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbchf
import pyqmc.api as pyq


# spin = 0
def run_hchain(r, chkfile, nk = 40, basis = 'ccpvdz', exp_to_discard=0.1, natom=2, smearing=0.01, rcut = 20, shifted = True):
  vacuum = 10.
  c = r*natom

  latvec = np.array([[vacuum,0.,0.],[0.,vacuum,0.],[0.,0.,c]])
  xyz = ";".join(f"H 0. 0. {i*r+r/2}" for i in range(natom))

  cell = pbcgto.Cell(unit = 'B',
                a = latvec,
                atom = xyz,
                verbose = 4,
                precision = 1e-6,
                basis = basis,
            )

  cell.exp_to_discard = exp_to_discard
  # cutoff radius (in Bohr) of the lattice summation in the integral evaluation
  cell.rcut = rcut 
  cell.build()

  kpts=cell.make_kpts([1,1,nk],wrap_around=shifted)
  mf = pbchf.KUKS(cell, kpts).density_fit()
  mf.xc = 'PBE'
  mf.chkfile = chkfile
  mf = pbchf.addons.smearing_(mf, sigma=smearing)

  dm0 = mf.get_init_guess()
  mf.kernel(dm0)

  return cell, mf

if __name__ == "__main__":
  r = 1.4
  natom = 1
  k = 40
  a = r
  with h5py.File(f'./r{r}/h{natom}_r{r}_k{k}_pbc_bands_vdz.chk','w') as f:
    cell, mf = run_hchain(r, f'./r{r}/h{natom}_r{r}_k{k}_pbc_scf_vdz.chk', nk=k, exp_to_discard=0.1, natom = natom, rcut=25, shifted=True)
  
    band_kpts = [[0.,0.,ik] for ik in np.linspace(-np.pi/a, np.pi/a, 100)]
    eigvals, eigvecs = mf.get_bands(band_kpts)
    eigvals, eigvecs = np.asarray(eigvals), np.asarray(eigvecs)

    f.create_dataset(f'kpts',data=band_kpts)
    f.create_dataset(f'energy',data=eigvals)
    f.create_dataset(f'eigvecs',data=eigvecs)


