# Hybrid Quantum-Classical Continuous Thought Machine (Quantum CTM)

This project aims to implement a hybrid quantum-classical version of the Continuous Thought Machine (CTM).

## Approach

Our initial approach focuses on integrating **Quantum Memory Cells** into the CTM architecture. The core idea is to replace or augment the classical memory cells of the CTM with quantum circuits (qubits).

Key components will include:
- A classical controller (similar to the original CTM).
- Quantum memory cells that can be written to, read from, and evolved using quantum operations.
- An interface between the classical controller and the quantum memory.

## Potential Quantum Components Explored:

1.  **Quantum Memory Cells (Current Focus):**
    *   Memory states are represented by qubits.
    *   Operations on memory involve quantum gate applications and measurements.
2.  **Quantum Transition Functions:**
    *   State update mechanisms within the CTM could be implemented using quantum circuits.
3.  **Quantum Input/Output Encoding:**
    *   Input data could be encoded into quantum states, and CTM outputs could be derived from quantum measurements.

## Technology Stack (Tentative)

*   **Classical Deep Learning:** PyTorch (to align with the original CTM implementation)
*   **Quantum Computing:** Qiskit or Cirq (with potential cuQuantum acceleration)

## Next Steps

1.  Define the `QuantumMemoryCell` class.
2.  Define the `HybridCTM` class that incorporates `QuantumMemoryCell`.
3.  Develop methods for encoding classical data into quantum states and decoding quantum states back to classical data.
4.  Implement the forward pass of the `HybridCTM`.
5.  Design training procedures for the hybrid model. 