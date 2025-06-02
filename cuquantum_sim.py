import math

try:
    import cupy as cp
    from cuquantum import custatevec as cusv
    HAS_CUQUANTUM = True
except ImportError:
    HAS_CUQUANTUM = False


class CuQuantumSimulator:
    """Minimal wrapper around cuStateVec for small-to-medium circuits on a single GPU."""

    def __init__(self, num_qubits: int):
        if not HAS_CUQUANTUM:
            raise RuntimeError("cuQuantum Python bindings not found; cannot use GPU simulator.")
        if num_qubits > 26:
            # 26 qubits ‑-> 1 GB statevector (complex128). Fits in 4090 24 GB with headroom.
            # Users can override if they wish.
            print("[CuQuantumSimulator] Warning: >26 qubits will consume a lot of GPU memory!")
        self.num_qubits = num_qubits
        self.state = cp.zeros(2 ** num_qubits, dtype=cp.complex128)
        self.state[0] = 1.0 + 0.0j  # |0…0> initial state
        self.handle = cusv.create()  # cuStateVec handle

    # --------------------------------------------------
    # Gate helpers
    # --------------------------------------------------
    def _apply_single_qubit_gate(self, matrix, qubit):
        """Apply a 2×2 unitary ``matrix`` to ``qubit`` (0-based, little-endian)."""
        # cuStateVec expects column-major, complex64/128
        gate_mat = cp.asarray(matrix, dtype=cp.complex128).reshape((2, 2), order="F")
        cusv.apply_matrix(self.handle, self.state.data.ptr, gate_mat.data.ptr,
                          self.num_qubits, 1 << qubit, 0, cp.complex128)

    def ry(self, angle, qubit):
        c = math.cos(angle / 2)
        s = math.sin(angle / 2)
        mat = [[c, -s], [s, c]]
        self._apply_single_qubit_gate(mat, qubit)

    def rz(self, angle, qubit):
        e_minus = math.e ** (-1j * angle / 2)
        e_plus = math.e ** (1j * angle / 2)
        mat = [[e_minus, 0], [0, e_plus]]
        self._apply_single_qubit_gate(mat, qubit)

    # --------------------------------------------------
    # Measurement
    # --------------------------------------------------
    def measure_shots(self, qubits, shots: int = 1024):
        """Return a dict of bitstring -> counts for the specified ``qubits`` list."""
        # Create a copy to avoid modifying the original list if it's reused elsewhere
        bit_order = list(qubits) # Ensure it's a list and a copy
        bit_order.reverse()  # cuStateVec expects high-to-low ordering
        results = cusv.measure(self.handle, self.state.data.ptr, self.num_qubits,
                               (cp.asarray(bit_order, dtype=cp.int32).data).ptr,
                               len(bit_order), shots, 0)
        # results is a CuPy array of size shots with integer bitstrings (packed)
        counts = {}
        for idx in results.get():
            bitstring = format(idx, f"0{len(qubits)}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    def release(self):
        if self.handle:
            cusv.destroy(self.handle)
        self.handle = None
        self.state = None 