from copy import deepcopy
from functools import partial
import time

import h5py
import numpy as np

import pyscf
from pyscf import ao2mo
from pyscf.tools.fcidump import from_scf, read

import openfermion as of

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.qasm2 import dumps
from qiskit_aer import AerSimulator
import qiskit_aer.noise as noise
import qiskit_ibm_runtime
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch, SCIResult

import ffsim

from adaptvqe.utils import hamiltonian_from_fcidump
from adaptvqe.hamiltonians import FermionicHamiltonian
from adaptvqe.pools import DVG_CEO
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt

ibm_computer: str = "ibm_fez"

service = qiskit_ibm_runtime.QiskitRuntimeService(channel="local")
computer = service.backend()
sampler = Sampler(computer)

# prob_1 = 0.001
# prob_2 = 0.01

# error_1 = noise.depolarizing_error(prob_1, 1)
# error_2 = noise.depolarizing_error(prob_2, 2)

# noise_model = noise.NoiseModel()
# noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
# noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

# computer = AerSimulator(noise_model=noise_model)
# sampler = Sampler(computer)

lih_fname = "data/atp_0_fcidump/atp_0_be1_f0"

h_fcidump, norb, nelec = hamiltonian_from_fcidump(lih_fname)
h = FermionicHamiltonian(h_fcidump, "atp", nelec, diag_mode="quimb", max_mpo_bond=10, max_mps_bond=20)
nq = h.n
print(f"Hamiltonian has {nq} qubits.")

# Since we have alpha and beta e-'s, an n-qubit Hamiltonian as n/2 spatial orbitals.
assert h.n % 2 == 0
num_orbitals = h.n // 2

# TODO Get the exact energy from PySCF.
exact_energy = h.ground_energy
print(f"Got exact energy {exact_energy}.")

# Get one- and two-body integrals.
fci_read = read(lih_fname)
h1 = fci_read["H1"]
h2_packed = fci_read["H2"]
h2 = ao2mo.restore(1, h2_packed, num_orbitals)  # (norb,norb,norb,norb)
n_electrons = fci_read["NELEC"]
ecore = fci_read["ECORE"]
spin = 0 # TODO How would I know from the FCIDUMP alone?
num_elec_a = (n_electrons + spin) // 2
num_elec_b = (n_electrons - spin) // 2
nelec = (num_elec_a, num_elec_b)

start = time.monotonic()
pool = DVG_CEO(n=h.n)
stop = time.monotonic()
print(f"Finished building pool in {stop - start} seconds. Has {len(pool.operators)} operators.")

max_mpo_bond = 8
adapt_mps_bond = 16
my_adapt = TensorNetAdapt(
    pool=pool,
    custom_hamiltonian=h,
    max_adapt_iter=1,
    recycle_hessian=True,
    tetris=True,
    verbose=True,
    threshold=0.1,
    max_mpo_bond=max_mpo_bond,
    max_mps_bond=adapt_mps_bond,
)
my_adapt.initialize()

circuits = []
adapt_energies = []
for i in range(32):
    print(f"On iteration {i}.")
    my_adapt.run_iteration()
    data = my_adapt.data
    print(my_adapt.coefficients)
    ansatz_circuit = pool.get_circuit(my_adapt.indices, my_adapt.coefficients)
    print("coefficients:", my_adapt.coefficients)
    print("indices:", my_adapt.indices)
    # Prepare the HF reference state, then add the Ansatz circuit.
    q = QuantumRegister(2 * num_orbitals)
    circuit = QuantumCircuit(q)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(num_orbitals, nelec), q)
    circuit = circuit.compose(ansatz_circuit)
    circuit.measure_all()
    circuits.append(circuit)
    adapt_energies.append(my_adapt.energy)

pass_manager = generate_preset_pass_manager(
    optimization_level=3, backend=computer
)
pass_manager.pre_init = ffsim.qiskit.PRE_INIT

circuit_qasm_strs = []
for circuit in circuits:
    isa_circuit = pass_manager.run(circuit)
    qasm_str = dumps(isa_circuit)
    circuit_qasm_strs.append(qasm_str)

f = h5py.File(f"data/atp_0_be1_f0_circuits.hdf5", "w")
f.create_dataset("adapt_energies", data=np.array(adapt_energies))
f.close()