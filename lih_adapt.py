import pyscf
from pyscf.tools.fcidump import from_scf
import openfermion as of
from adaptvqe.utils import hamiltonian_from_fcidump
from adaptvqe.hamiltonians import FermionicHamiltonian
from adaptvqe.pools import DVG_CEO
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt

mol = pyscf.gto.Mole()
geom = of.chem.geometry_from_pubchem("LiH")
basis = "sto-3g"
symmetry = "C1"
multiplicity = 1
charge = 0
mol.build(
    atom=geom,
    basis=basis,
    symmetry=symmetry,
)
mf = pyscf.scf.RHF(mol)
mf.kernel()
lih_fname = "data/lih.fcidump"
from_scf(mf, lih_fname)

h_fcidump, norb, nelec = hamiltonian_from_fcidump(lih_fname)
h = FermionicHamiltonian(h_fcidump, "atp", nelec)
print(f"Hamiltonian has {h.n} qubits.")

pool = DVG_CEO(n=h.n)
print(f"Finished building pool. Has {len(pool.operators)} operators.")
max_mpo_bond = 200
adapt_mps_bond = 10
my_adapt = TensorNetAdapt(
    pool=pool,
    custom_hamiltonian=h,
    max_adapt_iter=1,
    recycle_hessian=True,
    tetris=True,
    verbose=True,
    threshold=0.1,
    max_mpo_bond=max_mpo_bond,
    max_mps_bond=adapt_mps_bond
)
my_adapt.initialize()