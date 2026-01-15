from copy import deepcopy
from functools import partial
import h5py
import numpy as np

import pyscf
from pyscf import ao2mo
from pyscf.tools.fcidump import from_scf, read

import openfermion as of

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.qasm2 import dumps
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

circuits = []
adapt_energies = []
for i in range(3):
    print(f"On iteration {i}.")
    my_adapt.run_iteration()
    data = my_adapt.data
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

# Run SQD.
spin_a_layout = list(range(0, 12))
spin_b_layout = [12, 13, 14, 15, 19, 35, 34, 33, 32, 31, 30, 29]
initial_layout = spin_a_layout + spin_b_layout

pass_manager = generate_preset_pass_manager(
    optimization_level=3, backend=computer, initial_layout=initial_layout[:nq]
)

bit_arrays = []
counts_list = []
for circuit in circuits:
    pass_manager.pre_init = ffsim.qiskit.PRE_INIT
    to_run = pass_manager.run(circuit)
    print(f"Gate counts (w/ pre-init passes): {to_run.count_ops()}")
    job = sampler.run([to_run], shots=30_000)
    bit_array = job.result()[0].data.meas
    counts1 = bit_array.get_counts()
    counts_list.append(counts1)
    bit_arrays.append(deepcopy(bit_array))

energies = []
errors = []

for bit_array in bit_arrays:
    # SQD options
    energy_tol = 1e-5
    occupancies_tol = 1e-6
    max_iterations = 20
    rng = np.random.default_rng(1)

    # Eigenstate solver options
    num_batches = 2
    samples_per_batch = 1000
    symmetrize_spin = True
    carryover_threshold = 1e-4
    max_cycle = 200

    # Pass options to the built-in eigensolver. If you just want to use the defaults,
    # you can omit this step, in which case you would not specify the sci_solver argument
    # in the call to diagonalize_fermionic_hamiltonian below.
    sci_solver = partial(solve_sci_batch, spin_sq=0.0, max_cycle=max_cycle)

    # List to capture intermediate results
    result_history = []


    def callback(results: list[SCIResult]):
        result_history.append(results)
        iteration = len(result_history)
        print(f"Iteration {iteration}")
        for i, result in enumerate(results):
            print(f"\tSubsample {i}")
            print(f"\t\tEnergy: {result.energy + ecore}") # + nuclear_repulsion_energy}")
            print(f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}")


    result = diagonalize_fermionic_hamiltonian(
        h1,
        h2,
        bit_array,
        samples_per_batch=samples_per_batch,
        norb=num_orbitals,
        nelec=nelec,
        num_batches=num_batches,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        max_iterations=max_iterations,
        sci_solver=sci_solver,
        symmetrize_spin=symmetrize_spin,
        carryover_threshold=carryover_threshold,
        callback=callback,
        seed=rng,
    )

    min_e = [
        min(result, key=lambda res: res.energy).energy + ecore # + nuclear_repulsion_energy
        for result in result_history
    ]
    approx_energy = min(min_e)
    err = abs(min(min_e) - exact_energy)
    print(approx_energy, err)
    energies.append(approx_energy)
    errors.append(err)

qasm_strings = []
for circuit in circuits:
    isa_circuit = pass_manager.run(circuit)
    qasm_str = dumps(isa_circuit)
    qasm_strings.append(qasm_str)

molec_name = "LiH"
f = h5py.File(f"data/{molec_name}.hdf5", "w")
f.create_dataset("energies", data=np.array(energies))
f.create_dataset("circuit_qasm", data=qasm_strings)