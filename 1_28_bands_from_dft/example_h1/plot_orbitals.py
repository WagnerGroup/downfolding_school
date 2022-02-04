import numpy as np
from pyscf.tools import cubegen
import pyqmc.api as pyq

cell, mf = pyq.recover_pyscf('./r1.4/h1_r1.4_k40_pbc_scf_vdz.chk',cancel_outputs=False)

ikpt = 0
for iorb in [0,1,2]:
  cubegen.orbital(cell, f"./mo_plots/k{ikpt}_mo{iorb}.cube", mf.mo_coeff_kpts[0][ikpt][:,iorb])
