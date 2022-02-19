import numpy as np
import h5py
import pyscf
from pyscf.pbc import df as pdf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbchf
from pyscf.pbc.tools import pywannier90
import pyqmc.api as pyq

def recover_mf(r, chkfile):
  cell, mf = pyq.recover_pyscf(chkfile,cancel_outputs=False)
  mf.xc = 'PBE'
  return cell, mf

if __name__ == "__main__":
  r = 1.4
  natom = 1
  nk = 40
  scf_chkfile = f'../scf/r{r}/h{natom}_r{r}_k{nk}_pbc_scf_vdz.chk'
  cell, mf = recover_mf(r,scf_chkfile)

  num_wann = 1
  keywords = \
  """
  begin projections
  H:s
  end projections
  write_xyz = True
  write_hr = True
  """

  kmesh = [1,1,nk]
  w90 = pywannier90.W90(mf, cell, kmesh, num_wann, spin='up', other_keywords=keywords)

  w90.make_win()
  w90.setup()
  #w90.export_unk()
  w90.kernel()
  w90.plot_wf(outfile=f"MLWF", supercell=kmesh, grid=[100,100,50])
  w90.plot_guess_orbs()

  # export A, M matrices and use them for getting the wannier functions
  w90.export_AME()
  w90.kernel(external_AME='wannier90')

  # get the interpolated bands
  a = natom*r
  band_kpts = [[0.,0.,ik] for ik in np.linspace(-np.pi/a, np.pi/a, 100)] 
  frac_kpts = cell.get_scaled_kpts(band_kpts)
  interpolated_fock = w90.interpolate_ham_kpts(frac_kpts)   
  eigvals, eigvecs = w90.interpolate_band(frac_kpts)                
  with h5py.File(f'wannier_bands.chk','w') as f:
    f.create_dataset('kpts', data=np.asarray(frac_kpts))
    f.create_dataset('eigvals', data=np.asarray(eigvals))
    f.create_dataset('eigvecs', data=np.asarray(eigvecs))


