import argparse
import pickle as pkl
import h5py

import numpy as np

import pyscf
from pyscf import mcscf

import openfermion as of
from openfermion import MolecularData
from openfermionpyscf import run_pyscf

from qiskit.qasm2 import dump
import qiskit_ibm_runtime

from adaptvqe.convert import cirq_pauli_sum_to_qiskit_pauli_op
from adaptvqe.pools import DVG_CEO
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt, TensorNetAdapt

parser = argparse.ArgumentParser()
parser.add_argument("num-iters", type=int, help="Number of ADAPT iterations.")
parser.add_argument("--mpo-bond", type=int, default=200, help="Max bond dim of the MPO.")
parser.add_argument("--mps-bond", type=int, default=20, help="Bond dimension of the MPS.")
args = parser.parse_args()

mol = pyscf.gto.Mole()
geom = of.chem.geometry_from_pubchem("water")
basis = "sto-3g"
symmetry = "C1"
multiplicity = 1
charge = 0
mol.build(
    atom=geom,
    basis=basis,
    symmetry=symmetry,
)

n_frozen = 0
active_space = range(n_frozen, mol.nao_nr())

# Get molecular integrals
scf = pyscf.scf.RHF(mol).run()
num_orbitals = len(active_space)
print(f"Molecule has {num_orbitals} orbitals.")
n_electrons = int(sum(scf.mo_occ[active_space]))
num_elec_a = (n_electrons + mol.spin) // 2
num_elec_b = (n_electrons - mol.spin) // 2
cas = mcscf.CASCI(scf, num_orbitals, (num_elec_a, num_elec_b))
mo = cas.sort_mo(active_space, base=0)
hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), num_orbitals)

# Compute exact energy
exact_energy = cas.run().e_tot

mol_of = MolecularData(geom, basis, multiplicity, charge, description='H2')
mol_of = run_pyscf(mol_of, run_fci=True, run_ccsd=True)
nelec = (num_elec_a, num_elec_b)
print(mol_of.n_orbitals)
print(mol_of.hf_energy)

f = h5py.File("data/h2o_molec_data.hdf5", "w")
f.create_dataset("hcore", data=hcore)
f.create_dataset("eri", data=eri)
f.create_dataset("norb", data=num_orbitals)
f.create_dataset("exact_energy", data=exact_energy)
f.create_dataset("nuclear_repulsion_energy", data=nuclear_repulsion_energy)
f.create_dataset("nelec", data=nelec)
f.close()

h_of_fermi = mol_of.get_molecular_hamiltonian()
h_of = of.transforms.jordan_wigner(h_of_fermi)
h_cirq = of.transforms.qubit_operator_to_pauli_sum(h_of)
h_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(h_cirq)

with open("data/h_qiskit.pkl", "wb") as f:
    pkl.dump(h_qiskit, f)

pool = DVG_CEO(mol_of)

max_mpo_bond = args.mpo_bond
adapt_mps_bond = args.mps_bond
my_adapt = TensorNetAdapt(
    pool=pool,
    molecule=mol_of,
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
for i in range(args.num_iters):
    print(f"On iteration {i}.")
    my_adapt.run_iteration()
    data = my_adapt.data
    circuit = data.get_circuit(
        pool, indices=my_adapt.indices, coefficients=my_adapt.coefficients,
        include_ref=True
    )
    circuit.measure_all()
    circuits.append(circuit)
    adapt_energies.append(my_adapt.energy)
    dump(circuit, f"data/h2o_circuit_{i}.qasm")

adapt_energies = np.array(adapt_energies)
np.savetxt("data/h2o_adapt_energies.txt", adapt_energies)