import socket
from qiskit import QuantumCircuit, Aer, execute
from random import choice
import pickle  # For sending complex data structures

# BB84 Measurement (Bob's Side)
def bb84_bob(qubits, num_bits=16):
    bob_bases = [choice(['X', 'Z']) for _ in range(num_bits)]
    bob_results = []

    for qubit, basis in zip(qubits, bob_bases):
        qc = qubit
        if basis == 'X':
            qc.h(0)  # Change to X-basis for measurement
        qc.measure(0, 0)

        # Simulate measurement
        backend = Aer.get_backend('qasm_simulator')
        result = execute(qc, backend, shots=1).result()
        counts = result.get_counts()
        measured_bit = int(max(counts, key=counts.get))  # Most likely result
        bob_results.append(measured_bit)

    return bob_bases, bob_results

def main():
    host = '127.0.0.1'
    port = 65432

    # Connect to the server
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host, port))

    # Receive qubits from server (Alice)
    alice_qubits = pickle.loads(client.recv(4096))

    # Bob's side of BB84
    num_bits = 16
    bob_bases, bob_results = bb84_bob(alice_qubits, num_bits)

    # Send Bob's bases and results to Alice
    bob_data = {'bases': bob_bases, 'results': bob_results}
    client.sendall(pickle.dumps(bob_data))

    # Receive Alice's bases
    alice_bases = pickle.loads(client.recv(4096))
    sifted_key = [bob_results[i] for i in range(num_bits) if alice_bases[i] == bob_bases[i]]
    print(f"Sifted Key: {sifted_key}")

    client.close()

if __name__ == '__main__':
    main()
