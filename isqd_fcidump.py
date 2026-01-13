from adaptvqe.utils import hamiltonian_from_fcidump
from adaptvqe.hamiltonians import FermionicHamiltonian

h_fcidump, norb, nelec = hamiltonian_from_fcidump("data/atp_0_fcidump/atp_0_be2_f338")
h = FermionicHamiltonian(h_fcidump, "atp", nelec)