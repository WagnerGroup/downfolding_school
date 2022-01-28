import numpy as np
import h5py
from pathlib import Path

from pyscf.pbc import df as pdf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbchf
from ase.lattice import bulk
from ase.dft.kpoints import sc_special_points as special_points, get_bandpath

def run_hchain(r, chkfile, nk = 10, basis = 'ccpvdz', exp_to_discard=0.1, natom=2, smearing=0.01, rcut = 20, shifted = True):
    vacuum = 10.
    c = r*natom

    latvec = np.array([[vacuum,0.,0.],[0.,vacuum,0.],[0.,0.,c]])
    xyz = ";".join(f"H 0. 0. {i*r}" for i in range(natom))

    cell = pbcgto.Cell(unit = 'B',
                a = latvec,
                atom = xyz,
                verbose = 4,
                precision = 1e-6,
                basis = basis,
            )

    cell.exp_to_discard = exp_to_discard
    cell.rcut = rcut
    cell.build()

    kpts=cell.make_kpts([1,1,nk],wrap_around=shifted)
    mf = pbchf.KUKS(cell, kpts).density_fit()
    mf.xc = 'PBE'
    mf.chkfile = chkfile
    mf = pbchf.addons.smearing_(mf, sigma=smearing)

    # break spin up and down symmetry
    H = cell.search_ao_label("H 1s")
    dm0 = mf.get_init_guess()
    for i in np.arange(0,natom,2):
      H_A = H[[i]]
      H_B = H[[i+1]]
      dm0[0, :, H_A, H_A] = 1.0
      dm0[0, :, H_B, H_B]  = 0.0
      dm0[1, :, H_A, H_A]  = 0.0
      dm0[1, :, H_B, H_B] = 1.0
    mf.kernel(dm0)

    return cell, mf


if __name__ == "__main__":
  r = 1.4
  basis = 'vdz'
  Path(f'./r{r}/').mkdir(parents=True, exist_ok=True)
  with h5py.File(f'./r{r}/h2_pbc_bands_{basis}.chk','w') as f:
    for nk in [1,2,3,4,5,6,7,8,9,10,11,21]:
      cell, mf = run_hchain(r, f'./r{r}/h2_k{nk}_pbc_scf_{basis}.chk', nk=nk, basis=f'ccp{basis}', exp_to_discard=0.1, natom = 2, rcut=25, shifted=True)

      points = special_points['orthorhombic']
      G = points['G']
      Z = points['Z']
      Z_prime = [0., 0., -0.5]
      band_kpts, kpath, sp_points = get_bandpath([Z_prime, G, Z], cell.a, npoints = 50)
      band_kpts = cell.get_abs_kpts(band_kpts)

      eigvals, eigvecs = mf.get_bands(band_kpts)
      eigvals, eigvecs = np.asarray(eigvals), np.asarray(eigvecs)

      f.create_dataset(f'k{nk}/kpts',data=band_kpts)
      f.create_dataset(f'k{nk}/energy',data=eigvals)
      f.create_dataset(f'k{nk}/eigvecs',data=eigvecs)
  
