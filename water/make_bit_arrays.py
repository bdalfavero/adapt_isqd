"""Simulate a circuit with a depolarizing noise model and save the resulting BitArray for later use."""

import h5py

import argparse

import ffsim

import qiskit
from qiskit.qasm2 import load
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.primitives import BitArray
from qiskit_aer import AerSimulator  # For MPS Simulator.
from qiskit.primitives import StatevectorEstimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import EstimatorV2 as Estimator
import qiskit_ibm_runtime
from qiskit_ibm_runtime import SamplerV2 as Sampler

parser = argparse.ArgumentParser()
parser.add_argument("adapt_iter", type=int, help="Number of the ADAPT iteration to run.")
parser.add_argument("p_depol", type=float, help="Probability of depolarizing error.")
parser.add_argument("--nshots", type=int, default=30_000, help="Number of shots. Defaults to 30,000.")
args = parser.parse_args()

circuit_fname = f"data/h2o_circuit_{args.adapt_iter}.qasm"
circuit = load(circuit_fname)

nq = circuit.num_qubits

ibm_computer: str = "ibm_fez"

service = qiskit_ibm_runtime.QiskitRuntimeService(channel="local")
computer = service.backend()
pass_manager = generate_preset_pass_manager(
    optimization_level=3, backend=computer, initial_layout=list(range(nq))
)
simualtor = AerSimulator(method="matrix_product_state")
sampler = Sampler(simualtor)
pass_manager.pre_init = ffsim.qiskit.PRE_INIT
to_run = pass_manager.run(circuit)
print(f"Gate counts (w/ pre-init passes): {to_run.count_ops()}")
job = sampler.run([to_run], shots=30_000)
bit_array = job.result()[0].data.meas

bool_array = bit_array.to_bool_array()

f = h5py.File(f"data/bit_array_iter{args.adapt_iter}_p{args.p_depol:5.4e}.hdf5", "w")
f.create_dataset("p_depol", data=args.p_depol)
f.create_dataset("adapt_iter", data=args.adapt_iter)
f.create_dataset("bool_array", data=bool_array)
f.close()