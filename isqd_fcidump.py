from adaptvqe.utils import hamiltonian_from_fcidump
from adaptvqe.hamiltonians import FermionicHamiltonian
from adaptvqe.pools import DVG_CEO
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt

h_fcidump, norb, nelec = hamiltonian_from_fcidump("data/atp_0_fcidump/atp_0_be2_f338")
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