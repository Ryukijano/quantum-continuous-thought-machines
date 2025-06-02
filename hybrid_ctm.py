import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse # Added for command-line arguments

# Ensure QuantumMemoryCell can be imported when running as script or module
try:
    from .quantum_memory_cell import QuantumMemoryCell
except ImportError:
    from quantum_memory_cell import QuantumMemoryCell # Fallback for direct script execution

class HybridCTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_memory_slots,
                 num_qubits_per_slot, memory_depth_per_slot, 
                 backend_mode: str, # "qr-cloud", "gpu", "cpu-qiskit"
                 qr_token: str = None, qr_user_name: str = None, 
                 qr_backend_name='scarlet_quantum_rings',
                 qr_job_poll_interval=5, qr_job_timeout=300):
        super(HybridCTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_memory_slots = num_memory_slots

        # Classical components (simplified from original CTM)
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.query_generator = nn.Linear(hidden_size, hidden_size) # Generates query for memory
        
        # Quantum Memory Cell - one instance that manages multiple slots internally
        self.quantum_memory = QuantumMemoryCell(
            num_qubits=num_qubits_per_slot, 
            memory_depth=num_memory_slots, # The cell sees num_memory_slots as its internal depth
            classical_hidden_size=hidden_size, # The size of vectors it processes
            backend_mode=backend_mode,
            qr_token=qr_token, 
            qr_user_name=qr_user_name,
            qr_backend_name=qr_backend_name,
            job_poll_interval=qr_job_poll_interval, 
            job_timeout=qr_job_timeout
        )

        # Memory slot selection (simplified: learnable query, dot product with slot embeddings)
        # For simplicity, we don't explicitly model slot embeddings here, but rather pass
        # the slot index directly based on some logic (e.g., round-robin or learned).
        # Here, we assume the slot index will be provided to the forward pass.
        
        # Output layer / transformation (after memory read)
        self.output_gate_and_transform = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), # hidden_state + memory_output
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.current_hidden_state = None # To store the hidden state across steps if needed

    def forward(self, x_batch, memory_slot_indices_batch):
        """
        x_batch: (batch_size, input_size)
        memory_slot_indices_batch: (batch_size) tensor or list of integers indicating which memory slot to use for each item in the batch.
                                 Each index should be < self.num_memory_slots.
        """
        batch_size = x_batch.size(0)

        # 1. Process input and update hidden state
        processed_input = self.input_layer(x_batch)
        # For a recurrent model, you'd combine processed_input with self.current_hidden_state
        # For this example, let's assume hidden_state is directly derived from current input for simplicity.
        current_hidden_state_batch = torch.relu(processed_input) # (batch_size, hidden_size)
        self.current_hidden_state = current_hidden_state_batch # Store for potential recurrent use

        # 2. Generate query from hidden state (if needed for content-based addressing)
        # memory_query_batch = self.query_generator(current_hidden_state_batch) # (batch_size, hidden_size)
        # For this example, we are using explicit slot_indices, so query is for the content to write.
        # The `classical_hidden_state` passed to quantum_memory.write is effectively the content.

        # 3. Write to and Read from Quantum Memory
        # The QuantumMemoryCell now handles batching internally for write and read based on slot_indices_batch
        # The `current_hidden_state_batch` is the information we want to encode and then retrieve.
        memory_output_batch = self.quantum_memory.forward(
            classical_hidden_state_batch=current_hidden_state_batch, 
            memory_slot_indices_batch=memory_slot_indices_batch,
            shots_for_read=1024 # Default shots, can be made configurable
        )
        # memory_output_batch is (batch_size, hidden_size)

        # 4. Combine hidden state with memory output
        combined_representation = torch.cat((current_hidden_state_batch, memory_output_batch), dim=1)

        # 5. Generate final output
        final_output_batch = self.output_gate_and_transform(combined_representation)
        
        return final_output_batch, current_hidden_state_batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HybridCTM Example Runner")
    parser.add_argument("--backend_mode", type=str, default="cpu-qiskit", 
                        choices=["qr-cloud", "gpu", "cpu-qiskit"], 
                        help="Backend to use for quantum operations.")
    parser.add_argument("--num_qubits", type=int, default=2, help="Number of qubits per memory slot.")
    parser.add_argument("--num_slots", type=int, default=3, help="Number of memory slots.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for the example run.")
    parser.add_argument("--shots", type=int, default=100, help="Number of shots for quantum memory read.")
    # Add QR specific args only if needed, QuantumMemoryCell has defaults or can take them
    args = parser.parse_args()

    print(f"\nRunning HybridCTM example with backend: {args.backend_mode}")

    # Define model parameters
    input_dim = 10
    hidden_dim = 20 
    output_dim = 5
    num_mem_slots = args.num_slots
    qubits_per_slot = args.num_qubits
    # memory_depth_per_slot is implicitly 1 for each slot within QuantumMemoryCell, 
    # as QuantumMemoryCell's memory_depth parameter corresponds to num_mem_slots here.
    
    # Quantum Rings credentials (ensure these are set if using qr-cloud)
    YOUR_QR_TOKEN = "rings-200.awzkwVoeeFuJXmwhgtcXYw8thSABPd3k"
    YOUR_QR_USER_NAME = "gyanateet@gmail.com"

    # Instantiate HybridCTM
    model_params = {
        "input_size": input_dim,
        "hidden_size": hidden_dim,
        "output_size": output_dim,
        "num_memory_slots": num_mem_slots,
        "num_qubits_per_slot": qubits_per_slot,
        "memory_depth_per_slot": 1, # See note above
        "backend_mode": args.backend_mode
    }
    if args.backend_mode == "qr-cloud":
        model_params["qr_token"] = YOUR_QR_TOKEN
        model_params["qr_user_name"] = YOUR_QR_USER_NAME
        # Can add poll_interval, timeout here if desired to override defaults

    try:
        hybrid_ctm_model = HybridCTM(**model_params)
        print("HybridCTM model initialized successfully.")
    except Exception as e:
        print(f"Error initializing HybridCTM model: {e}")
        exit()

    # Create dummy input data
    batch_s = args.batch_size
    dummy_x_batch = torch.randn(batch_s, input_dim)
    
    # For this example, let's assign slots in a round-robin fashion for the batch
    # Ensure slot indices are within [0, num_mem_slots - 1]
    dummy_slot_indices = torch.arange(batch_s) % num_mem_slots

    print(f"Input batch shape: {dummy_x_batch.shape}")
    print(f"Target memory slots for batch: {dummy_slot_indices.tolist()}")

    # Perform a forward pass
    try:
        print("\nPerforming forward pass...")
        output, hidden_state = hybrid_ctm_model(dummy_x_batch, dummy_slot_indices)
        print("Forward pass successful.")
        print(f"Output shape: {output.shape}")
        print(f"Hidden state shape: {hidden_state.shape}")
        print(f"Output example (first item):\n{output[0]}")
    except Exception as e:
        print(f"Error during forward pass: {e}")

    print("\nHybridCTM example finished.") 