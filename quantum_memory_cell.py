import torch
import torch.nn as nn
import numpy as np
import time

# Attempt to import Qiskit first, as it's used for circuit construction
try:
    from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
    from qiskit_aer import AerSimulator # For cpu-qiskit mode
    HAS_QISKIT = True
    print("Qiskit and AerSimulator found.")
except ImportError:
    HAS_QISKIT = False
    print("Qiskit or AerSimulator not found. CPU-qiskit mode will be unavailable. Other operations might be limited.")
    # Define dummy Qiskit classes if Qiskit is not available
    class QuantumCircuit:
        def __init__(self, *regs, name=None):
            self.num_qubits = 0
            self.qregs = []
            self.cregs = []
            for reg in regs:
                if isinstance(reg, int): self.num_qubits += reg
                elif hasattr(reg, 'size') and hasattr(reg, 'name'):
                    if all(hasattr(reg, qubit_attr) for qubit_attr in ['register', 'index']): pass
                    elif 'Classical' in str(type(reg)): self.cregs.append(reg)
                    else: self.qregs.append(reg); self.num_qubits += reg.size
            self._data = []
        def ry(self, angle, qubit): self._data.append(('ry', angle, qubit))
        def rz(self, angle, qubit): self._data.append(('rz', angle, qubit))
        def measure(self, qubit, cbit): self._data.append(('measure', qubit, cbit))
        def copy(self, name=None):
            new_qc = QuantumCircuit(name=name); new_qc.num_qubits = self.num_qubits
            new_qc._data = list(self._data); new_qc.qregs = list(self.qregs); new_qc.cregs = list(self.cregs)
            return new_qc
        def draw(self, output='text'): return f"Dummy Qiskit Circuit ({self.num_qubits}q, {len(self._data)}ops)"
        @property
        def data(self): return self._data
        def __len__(self): return len(self._data)
        def to_gate(self, label=None): return self

    class QuantumRegister:
        def __init__(self, size, name): self.size = size; self.name = name
        def __getitem__(self, key): return (self, key)
    
    class ClassicalRegister:
        def __init__(self, size, name): self.size = size; self.name = name
        def __getitem__(self, key): return (self, key)

    def transpile(circuit, backend=None, **kwargs): return circuit
    class AerSimulator:
        def __init__(self): self.name = "dummy_aer_simulator"
        def run(self, circuit, shots=1): return DummyJob(circuit, shots)
        def __str__(self): return self.name


# Attempt to import QuantumRingsLib
try:
    from QuantumRingsLib import QuantumRingsProvider, JobStatus, BackendV2
    HAS_QUANTUM_RINGS = True
    print("QuantumRingsLib and BackendV2 found.")
except ImportError:
    HAS_QUANTUM_RINGS = False
    print("QuantumRingsLib or BackendV2 not found. 'qr-cloud' mode will use dummy backend.")
    class QuantumRingsProvider:
        def __init__(self, token, name): pass
        def active_account(self): pass
        def get_backend(self, backend_name): return None
    class BackendV2: # Dummy BackendV2
        def __init__(self, provider, num_qubits): # Provider can be None for dummy
            self.name = "dummy_qr_cloud_backend"
            self._num_qubits = num_qubits
            print(f"Initialized Dummy QuantumRings BackendV2 for {num_qubits} qubits.")
        def run(self, circuit, shots=1):
            print(f"Dummy QR BackendV2: Running circuit with {shots} shots.")
            return DummyJob(circuit, shots)
        def __str__(self): return self.name
        @property
        def num_qubits(self): return self._num_qubits


# Common Dummy Job/Result for QR dummy and AerSimulator dummy
class JobStatus: DONE = 'DONE'; ERROR = 'ERROR'; CANCELLED = 'CANCELLED'; RUNNING = 'RUNNING'; QUEUED = 'QUEUED'
class DummyJob:
    def __init__(self, circuit, shots): self.circuit = circuit; self.shots = shots; self._status = JobStatus.QUEUED; self._result = None
    def status(self):
        if self._status == JobStatus.QUEUED: self._status = JobStatus.RUNNING
        elif self._status == JobStatus.RUNNING: self._status = JobStatus.DONE
        return self._status
    def result(self):
        if self._status != JobStatus.DONE: raise RuntimeError("Job not done.")
        if self._result is None:
            num_q = self.circuit.num_qubits
            counts = {'0'*num_q: self.shots}
            self._result = DummyResult(counts)
        return self._result
class DummyResult:
    def __init__(self, counts): self._counts = counts
    def get_counts(self): return self._counts

from .cuquantum_sim import CuQuantumSimulator, HAS_CUQUANTUM

class QuantumMemoryCell(nn.Module):
    def __init__(self, num_qubits, memory_depth, classical_hidden_size,
                 backend_mode: str, # "qr-cloud", "gpu", "cpu-qiskit"
                 qr_token: str = None, qr_user_name: str = None,
                 qr_backend_name='scarlet_quantum_rings', 
                 job_poll_interval=5, job_timeout=300):
        super(QuantumMemoryCell, self).__init__()
        
        if not HAS_QISKIT and backend_mode != "gpu": # Qiskit needed for circuit construction unless pure GPU
            raise ImportError("Qiskit is required for this backend_mode.")

        self.num_qubits = num_qubits
        self.memory_depth = memory_depth
        self.classical_hidden_size = classical_hidden_size
        self.job_poll_interval = job_poll_interval
        self.job_timeout = job_timeout
        self.backend_mode = backend_mode
        
        self.gpu_simulators = None
        self.backend = None # For Qiskit-based backends (QR cloud, Aer)

        print(f"[QuantumMemoryCell] Initializing for backend_mode: {self.backend_mode}")

        if self.backend_mode == "gpu":
            if HAS_CUQUANTUM:
                print("[QuantumMemoryCell] Using cuQuantum GPU simulator.")
                # Initialize one GPU simulator per memory slot
                self.gpu_simulators = [CuQuantumSimulator(self.num_qubits) for _ in range(self.memory_depth)]
            else:
                raise RuntimeError("cuQuantum not found, but backend_mode='gpu' was requested.")
        elif self.backend_mode == "qr-cloud":
            if HAS_QUANTUM_RINGS and qr_token and qr_user_name:
                try:
                    provider = QuantumRingsProvider(token=qr_token, name=qr_user_name)
                    provider.active_account()
                    self.backend = provider.get_backend(qr_backend_name)
                    print(f"[QuantumMemoryCell] QuantumRingsLib: Initialized backend '{self.backend}'.")
                    if hasattr(self.backend, 'num_qubits'):
                         print(f"                 Available qubits on backend: {self.backend.num_qubits}")
                except Exception as e:
                    print(f"[QuantumMemoryCell] Error initializing QuantumRingsLib backend: {e}. Falling back to dummy QR backend.")
                    self.backend = BackendV2(None, num_qubits=self.num_qubits) # Dummy QR Backend
            else:
                print("[QuantumMemoryCell] QuantumRingsLib not found or credentials missing. Using dummy QR backend for 'qr-cloud' mode.")
                self.backend = BackendV2(None, num_qubits=self.num_qubits) # Dummy QR Backend
        elif self.backend_mode == "cpu-qiskit":
            if HAS_QISKIT and AerSimulator: # Check if AerSimulator class is truly available
                print("[QuantumMemoryCell] Using Qiskit AerSimulator (CPU).")
                self.backend = AerSimulator()
            else:
                print("[QuantumMemoryCell] Qiskit AerSimulator not available. Cannot use 'cpu-qiskit' mode.")
                raise ImportError("AerSimulator not found for 'cpu-qiskit' mode.")
        else:
            raise ValueError(f"Unsupported backend_mode: {self.backend_mode}")

        self.encoding_linear = nn.Linear(classical_hidden_size, num_qubits * 2)
        self.decoding_linear = nn.Linear(num_qubits, classical_hidden_size)
        # Stores Qiskit QuantumCircuit objects if not in pure GPU mode, or None
        self.quantum_memory_states = [None] * memory_depth 
        print(f"[QuantumMemoryCell] Initialized for {num_qubits} qubits/slot, depth {memory_depth}.")

    def _get_angles_single(self, classical_hidden_state_single_item):
        if classical_hidden_state_single_item.dim() != 2 or classical_hidden_state_single_item.size(0) != 1:
            raise ValueError("_get_angles_single expects input of shape (1, classical_hidden_size)")
        angles = self.encoding_linear(classical_hidden_state_single_item)
        return angles.view(self.num_qubits, 2) * np.pi

    def write(self, classical_hidden_state_batch, memory_slot_indices_batch):
        batch_size = classical_hidden_state_batch.size(0)
        
        if isinstance(memory_slot_indices_batch, int):
            memory_slot_indices_batch = torch.tensor([memory_slot_indices_batch] * batch_size, device=classical_hidden_state_batch.device)
        elif not isinstance(memory_slot_indices_batch, torch.Tensor):
             memory_slot_indices_batch = torch.tensor(memory_slot_indices_batch, device=classical_hidden_state_batch.device)


        if memory_slot_indices_batch.numel() != batch_size:
             raise ValueError("memory_slot_indices_batch must be an int or a list/tensor of length batch_size.")

        for i in range(batch_size):
            slot_idx = memory_slot_indices_batch[i].item()
            if not (0 <= slot_idx < self.memory_depth):
                raise ValueError(f"Slot index {slot_idx} for item {i} out of bounds.")

            hidden_item = classical_hidden_state_batch[i].unsqueeze(0)
            angles = self._get_angles_single(hidden_item)
            
            if self.backend_mode == "gpu":
                sim = self.gpu_simulators[slot_idx]
                # Reset state to |0...0> before applying new gates
                sim.state.fill(0) 
                sim.state[0] = 1.0 + 0.0j
                for q_idx in range(self.num_qubits):
                    sim.ry(angles[q_idx, 0].item(), q_idx)
                    sim.rz(angles[q_idx, 1].item(), q_idx)
                self.quantum_memory_states[slot_idx] = None # Mark as GPU handled
            else: # "qr-cloud" or "cpu-qiskit"
                qr = QuantumRegister(self.num_qubits, f'q_b{i}_s{slot_idx}')
                qc = QuantumCircuit(qr, name=f"qmem_b{i}_s{slot_idx}")
                for q_idx in range(self.num_qubits):
                    qc.ry(angles[q_idx, 0].item(), qr[q_idx])
                    qc.rz(angles[q_idx, 1].item(), qr[q_idx])
                self.quantum_memory_states[slot_idx] = qc

    def read(self, memory_slot_indices_batch, shots=1024):
        if isinstance(memory_slot_indices_batch, int):
            memory_slot_indices_batch = torch.tensor([memory_slot_indices_batch] * 1) # Assume batch of 1
        elif not isinstance(memory_slot_indices_batch, torch.Tensor):
             memory_slot_indices_batch = torch.tensor(memory_slot_indices_batch)
        
        batch_size = memory_slot_indices_batch.numel()
        batch_outputs = []

        if (self.backend_mode != "gpu" and self.backend is None):
            print(f"ERROR: Qiskit-based backend not available for mode '{self.backend_mode}'.")
            return torch.randn(batch_size, self.classical_hidden_size)
        if (self.backend_mode == "gpu" and self.gpu_simulators is None):
            print("ERROR: GPU simulators not available for mode 'gpu'.")
            return torch.randn(batch_size, self.classical_hidden_size)


        for i in range(batch_size):
            slot_idx = memory_slot_indices_batch[i].item()
            if not (0 <= slot_idx < self.memory_depth):
                raise ValueError(f"Slot index {slot_idx} for item {i} out of bounds.")

            counts = None
            if self.backend_mode == "gpu":
                sim = self.gpu_simulators[slot_idx]
                # Note: measure_shots in CuQuantumSim takes a list of qubit indices
                counts = sim.measure_shots(list(range(self.num_qubits)), shots=shots)
            else: # "qr-cloud" or "cpu-qiskit"
                qc_blueprint = self.quantum_memory_states[slot_idx]
                if qc_blueprint is None: # Should not happen if write was called for this slot
                    print(f"Warning: No circuit blueprint for slot {slot_idx} in mode {self.backend_mode}. Using empty circuit.")
                    qr_empty = QuantumRegister(self.num_qubits, f'q_empty_b{i}_s{slot_idx}')
                    qc_blueprint = QuantumCircuit(qr_empty, name=f"qmem_empty_b{i}_s{slot_idx}")

                qr_meas = QuantumRegister(self.num_qubits, f'q_meas_b{i}_s{slot_idx}')
                cr_meas = ClassicalRegister(self.num_qubits, f'c_meas_b{i}_s{slot_idx}')
                measurement_qc = QuantumCircuit(qr_meas, cr_meas, name=qc_blueprint.name + "_meas")
                
                # Append blueprint operations
                if HAS_QISKIT and isinstance(qc_blueprint, QuantumCircuit):
                    # Create a gate from the blueprint and append it
                    # This assumes qc_blueprint only contains operations on a single register matching qr_meas
                    try:
                        temp_gate = qc_blueprint.to_gate(label=f"bp_b{i}_s{slot_idx}")
                        measurement_qc.append(temp_gate, qr_meas)
                    except Exception as e_gate:
                        print(f"Warning: Could not convert blueprint to gate directly ({e_gate}). Rebuilding ops.")
                        # Fallback: Rebuild (less efficient, more error prone if blueprint complex)
                        for op in qc_blueprint.data:
                            instruction, qargs, cargs = op
                            # Remap qargs to qr_meas, this is simplified
                            remapped_qargs = [qr_meas[q.index] for q in qargs]
                            measurement_qc.append(instruction, qargs=remapped_qargs, cargs=cargs)

                measurement_qc.measure(qr_meas, cr_meas)

                try:
                    tqc = measurement_qc # Default if no specific transpilation needed
                    if self.backend_mode == "cpu-qiskit": # AerSimulator might benefit from transpile
                         tqc = transpile(measurement_qc, backend=self.backend)
                    # For Quantum Rings, we run 'measurement_qc' directly as 'tqc' (no transpile needed if backend has no target)
                    # elif self.backend_mode == "qr-cloud" and (not hasattr(self.backend, 'target') or self.backend.target is None):
                    #    tqc = measurement_qc
                    # else: # Other qiskit backends might require this
                    #    if hasattr(self.backend, 'target'): # Qiskit >=0.45
                    #         tqc = transpile(measurement_qc, backend=self.backend)
                    #    else: # Older Qiskit or unknown backend, try basic transpile
                    #         tqc = transpile(measurement_qc)


                    job = self.backend.run(tqc, shots=shots)
                    
                    if self.backend_mode == "qr-cloud": # Only QR cloud needs polling
                        start_time = time.time()
                        while True:
                            status = job.status()
                            if status in [JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED]: break
                            if time.time() - start_time > self.job_timeout:
                                raise TimeoutError(f"Job (item {i}, slot {slot_idx}) timed out. Status: {status}")
                            time.sleep(self.job_poll_interval)
                        if status != JobStatus.DONE:
                            raise RuntimeError(f"Job (item {i}, slot {slot_idx}) failed. Status: {status}")
                    
                    result = job.result()
                    counts = result.get_counts()
                except Exception as e:
                    print(f"Error for item {i}, slot {slot_idx} (mode: {self.backend_mode}): {e}")
                    batch_outputs.append(torch.randn(1, self.classical_hidden_size))
                    continue
            
            # Process counts
            probs = torch.zeros(self.num_qubits)
            if counts:
                for qubit_idx in range(self.num_qubits):
                    prob1 = 0
                    for bitstring, count_val in counts.items():
                        if len(bitstring) == self.num_qubits and bitstring[self.num_qubits - 1 - qubit_idx] == '1':
                            prob1 += count_val
                    probs[qubit_idx] = prob1 / shots if shots > 0 else 0.0
            
            classical_features_item = probs.unsqueeze(0)
            decoded_output_item = self.decoding_linear(classical_features_item)
            batch_outputs.append(decoded_output_item)
        
        if not batch_outputs: return torch.empty(0, self.classical_hidden_size, device=self.encoding_linear.weight.device)
        return torch.cat(batch_outputs, dim=0)

    def forward(self, classical_hidden_state_batch, memory_slot_indices_batch, shots_for_read=1024):
        self.write(classical_hidden_state_batch, memory_slot_indices_batch)
        classical_output_batch = self.read(memory_slot_indices_batch, shots=shots_for_read)
        return classical_output_batch

    def __del__(self):
        if self.backend_mode == "gpu" and self.gpu_simulators:
            for sim in self.gpu_simulators:
                if hasattr(sim, 'release') and callable(sim.release):
                    sim.release()
            print("[QuantumMemoryCell] Released cuQuantum simulators.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="QuantumMemoryCell Example Runner")
    parser.add_argument("--backend_mode", type=str, default="cpu-qiskit", choices=["qr-cloud", "gpu", "cpu-qiskit"], help="Backend to use for quantum operations.")
    parser.add_argument("--num_qubits", type=int, default=2, help="Number of qubits per memory slot.")
    parser.add_argument("--shots", type=int, default=100, help="Number of shots for read operations.")
    args = parser.parse_args()

    print(f"\nRunning QuantumMemoryCell example with backend: {args.backend_mode}")
    
    if not HAS_QISKIT and args.backend_mode != "gpu":
        print("Qiskit is not installed. This example requires Qiskit for non-GPU modes.")
    else:
        num_q = args.num_qubits
        mem_depth = 1 
        hidden_size = 4
        
        YOUR_QR_TOKEN = "rings-200.awzkwVoeeFuJXmwhgtcXYw8thSABPd3k" 
        YOUR_QR_USER_NAME = "gyanateet@gmail.com"

        q_memory_params = {
            "num_qubits": num_q,
            "memory_depth": mem_depth,
            "classical_hidden_size": hidden_size,
            "backend_mode": args.backend_mode
        }
        if args.backend_mode == "qr-cloud":
            q_memory_params["qr_token"] = YOUR_QR_TOKEN
            q_memory_params["qr_user_name"] = YOUR_QR_USER_NAME
            q_memory_params["job_poll_interval"] = 2 # Faster polling for example
            q_memory_params["job_timeout"] = 60    # Shorter timeout

        try:
            q_memory = QuantumMemoryCell(**q_memory_params)
        except Exception as e:
            print(f"Failed to initialize QuantumMemoryCell: {e}")
            exit()
            
        print(f"Using backend: {q_memory.backend if q_memory.backend_mode != 'gpu' else 'cuQuantum GPU Simulators'}")
        
        # Prepare batch inputs
        batch_s = 2 # Test with a batch of 2
        batch_hidden_state = torch.randn(batch_s, hidden_size)
        # Each item in batch targets a different slot if depth allows, else 0
        batch_slot_indices = torch.arange(batch_s) % mem_depth 

        print(f"\nWriting to slots {batch_slot_indices.tolist()} for a batch of {batch_s}...")
        q_memory.write(batch_hidden_state, batch_slot_indices)
        
        # Illustrate circuit drawing for the first slot if not GPU mode
        if args.backend_mode != "gpu" and mem_depth > 0:
            print("Circuit for slot 0 after write:")
            qc_slot0 = q_memory.quantum_memory_states[0]
            if qc_slot0 and hasattr(qc_slot0, 'draw') and callable(qc_slot0.draw):
                try: print(qc_slot0.draw(output='text'))
                except Exception as e: print(f"Could not draw circuit: {e}")
            elif qc_slot0: print("Circuit object does not have a callable draw method.")
            else: print("No Qiskit circuit stored for slot 0 (possibly GPU mode or unwritten).")


        print(f"\nReading from slots {batch_slot_indices.tolist()} for batch of {batch_s} with {args.shots} shots...")
        output = q_memory.read(batch_slot_indices, shots=args.shots)
        print(f"Output from quantum memory (batch_size {output.shape[0]}):\n{output}")
        print(f"Output shape: {output.shape}")

        print("\nTesting forward pass...")
        forward_output = q_memory.forward(torch.rand(batch_s, hidden_size), batch_slot_indices, shots_for_read=args.shots)
        print(f"Output from forward pass (batch_size {forward_output.shape[0]}):\n{forward_output}")
        print(f"Forward output shape: {forward_output.shape}")
        
        # Explicitly delete to test __del__ for GPU mode
        del q_memory 
        print("\nQuantumMemoryCell example finished.") 