import torch
import torch.nn as nn
import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator # Use AerSimulator
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("Qiskit or Qiskit Aer not found. Quantum operations will be non-functional placeholders.")
    # Define dummy classes if Qiskit is not available, so the rest of the code can still run
    class QuantumCircuit:
        def __init__(self, *args): pass
        def ry(self, angle, qubit): pass
        def rz(self, angle, qubit): pass
        def measure(self, qubit, cbit): pass
        def copy(self): return self
    
    class AerSimulator:
        def __init__(self): pass
        def run(self, qc, shots=1): 
            # Return dummy probabilities if Qiskit is not installed
            # Number of qubits is qc.num_qubits if that were available
            # For now, assume self.num_qubits is set in QuantumMemoryCell
            num_q = self.num_qubits if hasattr(self, 'num_qubits') else 1
            return DummyRunResult(num_q)

    class DummyRunResult:
        def __init__(self, num_qubits):
            self.num_qubits = num_qubits
        def result(self): return self
        def get_counts(self): 
            # Return a dummy count for a single outcome (all zeros)
            return {'0'*self.num_qubits: 1}


class QuantumMemoryCell(nn.Module):
    def __init__(self, num_qubits, memory_depth, classical_hidden_size):
        super(QuantumMemoryCell, self).__init__()
        self.num_qubits = num_qubits
        self.memory_depth = memory_depth
        self.classical_hidden_size = classical_hidden_size

        if not HAS_QISKIT:
            print("Warning: Qiskit not found. QuantumMemoryCell will operate in a dummy mode.")
            # Pass num_qubits to AerSimulator for dummy result generation
            AerSimulator.num_qubits = self.num_qubits


        # Each memory slot will have its own quantum state (circuit)
        # For simulation, we can re-initialize circuits as needed or maintain state vectors.
        # Here, we just store the blueprint; circuits are created on-the-fly for write/read.
        
        # Parameters for encoding classical hidden state to quantum rotation angles
        # We need 2 angles (e.g., for RY, RZ) per qubit.
        self.encoding_linear = nn.Linear(classical_hidden_size, num_qubits * 2)

        # Parameters for decoding quantum measurement outcomes (probabilities) to classical values
        # Input to this layer will be 2^num_qubits (for probabilities) or num_qubits (for expectation values)
        # For simplicity, let's assume we derive `num_qubits` features from measurements.
        self.decoding_linear = nn.Linear(num_qubits, classical_hidden_size) # Simplified: num_qubits features from measurement

        # Qiskit simulator
        self.simulator = AerSimulator() if HAS_QISKIT else AerSimulator() # AerSimulator will be dummy if no Qiskit

        # Store "memory" as a list of quantum circuits or statevectors.
        # For simplicity in this version, we'll re-create and run circuits.
        # A more advanced version might store and evolve statevectors directly if using statevector_simulator.
        self.quantum_memory_states = [None] * memory_depth # Placeholder, actual state is in the simulator run

        print(f"QuantumMemoryCell initialized with {num_qubits} qubits per memory slot, depth {memory_depth}.")
        if HAS_QISKIT:
            print("Qiskit backend: AerSimulator")

    def _get_angles(self, classical_hidden_state):
        """Maps classical hidden state to rotation angles for qubits."""
        # classical_hidden_state: (batch_size, classical_hidden_size)
        # For now, assume batch_size = 1 for quantum ops
        if classical_hidden_state.size(0) != 1:
            # This basic version processes one item at a time for quantum ops
            # A batch-compatible version would require more complex Qiskit handling or looping
            raise NotImplementedError("Batch size > 1 not supported for quantum ops in this basic version.")

        angles = self.encoding_linear(classical_hidden_state) # (1, num_qubits * 2)
        # Reshape to (num_qubits, 2) where each row is [angle_ry, angle_rz]
        return angles.view(self.num_qubits, 2) * np.pi # Scale to be in range [0, pi] or [0, 2pi] for angles

    def write(self, classical_hidden_state, memory_slot_index):
        """
        Encodes classical hidden state information into the specified quantum memory slot
        by creating and running a quantum circuit.
        The "state" is effectively re-written by running a new circuit.
        """
        if not HAS_QISKIT:
            print(f"Dummy Write to slot {memory_slot_index}")
            return

        if not (0 <= memory_slot_index < self.memory_depth):
            raise ValueError(f"Memory slot index {memory_slot_index} out of bounds.")

        angles = self._get_angles(classical_hidden_state) # (num_qubits, 2)

        qc = QuantumCircuit(self.num_qubits, self.num_qubits) # num_qubits, num_classical_bits_for_measurement
        for i in range(self.num_qubits):
            qc.ry(angles[i, 0].item(), i) # .item() to get Python float from tensor
            qc.rz(angles[i, 1].item(), i)
        
        # For a "write" operation, we prepare the state.
        # If using statevector simulator, we could save the statevector.
        # For now, this circuit represents the "written" state.
        # We don't strictly "run" it for write, but store its description (or could run and save statevector)
        self.quantum_memory_states[memory_slot_index] = qc 
        # print(f"Write operation: Prepared quantum circuit for memory slot {memory_slot_index}")
        return qc # Return the circuit representing the state

    def read(self, memory_slot_index, shots=1024):
        """
        Reads information from the specified quantum memory slot by measuring the prepared circuit.
        Returns a classical tensor derived from measurement probabilities.
        """
        if not (0 <= memory_slot_index < self.memory_depth):
            raise ValueError(f"Memory slot index {memory_slot_index} out of bounds.")

        qc = self.quantum_memory_states[memory_slot_index]
        if qc is None:
            # If never written, initialize to |0...0> state and measure
            # print(f"Warning: Reading from unwritten slot {memory_slot_index}. Initializing to |0...0>.")
            qc = QuantumCircuit(self.num_qubits, self.num_qubits)
            # Fallthrough to measurement

        if not HAS_QISKIT:
            print(f"Dummy Read from slot {memory_slot_index}")
            # Return a dummy tensor of the expected classical_hidden_size
            dummy_output_tensor = torch.randn(1, self.classical_hidden_size)
            return dummy_output_tensor


        # Add measurements
        measurement_qc = qc.copy() # Don't alter the stored state circuit blueprint
        measurement_qc.measure(range(self.num_qubits), range(self.num_qubits))

        # Execute the circuit
        compiled_circuit = transpile(measurement_qc, self.simulator)
        job = self.simulator.run(compiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(measurement_qc)

        # Convert counts to a probability distribution / feature vector
        # For simplicity, let's derive one feature per qubit: probability of being in state |1>
        # This is a very basic way to get a classical vector. More sophisticated methods exist.
        # For example, expectation values of Pauli operators, or full probability vector.
        
        probs = torch.zeros(self.num_qubits)
        for i in range(self.num_qubits):
            prob1 = 0
            for bitstring, count in counts.items():
                if bitstring[self.num_qubits - 1 - i] == '1': # Qiskit bitstrings are often reversed
                    prob1 += count
            probs[i] = prob1 / shots
        
        # Unsqueeze to make it (1, num_qubits) for the linear layer
        classical_features = probs.unsqueeze(0) 
        
        # Decode these features into a classical hidden state representation
        decoded_output = self.decoding_linear(classical_features) # (1, classical_hidden_size)
        return decoded_output

    def forward(self, classical_hidden_state, memory_slot_to_access):
        """
        Performs a write then a read operation on the specified memory slot.
        Assumes classical_hidden_state is for a single batch item.
        """
        self.write(classical_hidden_state, memory_slot_to_access)
        classical_output = self.read(memory_slot_to_access)
        return classical_output

# Example Usage (conceptual)
if __name__ == '__main__':
    if not HAS_QISKIT:
        print("\nSkipping Qiskit-based example as Qiskit is not installed.")
    else:
        print("\nRunning Qiskit-based example:")
        num_qubits_per_slot = 2
        memory_slots = 1 # Simplified for this example
        hidden_dim = 4  # Reduced for simpler display
        
        q_memory = QuantumMemoryCell(num_qubits_per_slot, memory_slots, hidden_dim)
        
        # Dummy classical hidden state (batch size 1)
        dummy_hidden = torch.randn(1, hidden_dim) 
        
        slot_idx = 0
        
        print(f"\nSimulating access to slot {slot_idx}:")
        
        # Perform a write
        print(f"Writing to slot {slot_idx} using hidden state: {dummy_hidden}")
        q_memory.write(dummy_hidden, slot_idx)
        print(f"Circuit for slot {slot_idx} after write: {q_memory.quantum_memory_states[slot_idx].draw(output='text')}")

        # Perform a read
        print(f"Reading from slot {slot_idx}:")
        output_from_qmem = q_memory.read(slot_idx)
        print(f"Output from quantum memory (slot {slot_idx}, shape {output_from_qmem.shape}): {output_from_qmem}")

        # Test forward pass
        print(f"\nTesting forward pass for slot {slot_idx} with new hidden state:")
        new_dummy_hidden = torch.rand(1, hidden_dim)
        output_forward = q_memory.forward(new_dummy_hidden, slot_idx)
        print(f"Circuit for slot {slot_idx} after forward: {q_memory.quantum_memory_states[slot_idx].draw(output='text')}")
        print(f"Output from forward pass (slot {slot_idx}, shape {output_forward.shape}): {output_forward}") 