import argparse
import collections
from functools import partial
import json

import h5py
import numpy as np

from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian, solve_sci_batch

import ffsim

parser = argparse.ArgumentParser()
parser.add_argument("--samples-per-batch", type=int, default=1_000, help="Number of samples per batch.")
parser.add_argument("--max-iterations", type=int, default=100, help="Maximum number of iterations.")
parser.add_argument("--num-batches", type=int, default=2, help="Number of batches.")
parser.add_argument("bit_array_files", type=str, nargs='*', help="HDF5 files with BitArray data.")
args = parser.parse_args()

f = h5py.File("data/h2o_molec_data.hdf5", "r")
hcore = f["hcore"][:, :]
eri = f["eri"][:, :, :,:]
nelec = tuple(f["nelec"][:])
num_orbitals = f["norb"][()]
nuclear_repulsion_energy = f["nuclear_repulsion_energy"][()]
exact_energy = f["exact_energy"][()]
f.close()

p_depols = []
adapt_iters = []
bool_arrays = []
for fname in args.bit_array_files:
    f = h5py.File(fname, "r")
    p_depol = f["p_depol"][()]
    adapt_iter = f["adapt_iter"][()]
    bool_array = f["bool_array"][()]
    f.close()
    print(fname, adapt_iter)
    p_depols.append(p_depol)
    adapt_iters.append(adapt_iter)
    bool_arrays.append(bool_array)

assert np.all(np.abs(p_depols - p_depols[0]) <= 1e-4), \
    f"Depolarizing probabilities must be all the same, but got\n{p_depols}."
p_depol = p_depols[0]

bit_arrays = [BitArray.from_bool_array(bool_array) for bool_array in bool_arrays]
counts_list = [bit_array.get_counts() for bit_array in bit_arrays]
all_counts = collections.Counter()
tuple_of_counts = tuple(counts_list)
for counts in tuple_of_counts:
    for bitstring, count in counts.items():
        all_counts[bitstring] += count

bit_array = BitArray.from_counts(all_counts)

# SQD options
energy_tol = 1e-5
occupancies_tol = 1e-6
max_iterations = args.max_iterations
rng = np.random.default_rng(1)

# Eigenstate solver options
num_batches = args.num_batches
samples_per_batch = args.samples_per_batch
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
        print(f"\t\tEnergy: {result.energy + nuclear_repulsion_energy}")
        print(f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}")


result = diagonalize_fermionic_hamiltonian(
    hcore,
    eri,
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
    min(result, key=lambda res: res.energy).energy + nuclear_repulsion_energy
    for result in result_history
]
approx_energy = min(min_e)
err = abs(min(min_e) - exact_energy)
print(f"Got energy {approx_energy:4.5e}, err {err:4.5e}")

adapt_iters = [int(it) for it in adapt_iters] # For output.

output_dict = {
    "p_depol": p_depol,
    "adapt_iters": adapt_iters,
    "energy": approx_energy,
    "error": err
}
iters_str = '_'.join([str(iter) for iter in adapt_iters])
output_fname = f"data/h2o_isqd_p{p_depol:5.4e}_{iters_str}.json"
with open(output_fname, "w") as f:
    json.dump(output_dict, f)