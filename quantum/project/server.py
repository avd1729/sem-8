import socket
from qiskit import QuantumCircuit, Aer, execute
from random import choice, randint
import pickle  # For sending complex data structures

# BB84 Key Generation (Alice's Side)
def bb84_alice(num_bits=16):
    alice_bits = [randint(0, 1) for _ in range(num_bits)]
    alice_bases = [choice(['X', 'Z']) for _ in range(num_bits)]
    alice_qubits = []

    for bit, basis in zip(alice_bits, alice_bases):
        qc = QuantumCircuit(1, 1)
        if bit == 1:
            qc.x(0)  # Prepare |1>
        if basis == 'X':
            qc.h(0)  # Change to X-basis
        alice_qubits.append(qc)

    return alice_bits, alice_bases, alice_qubits

# Sift Key Based on Matching Bases
def sift_key(alice_bits, alice_bases, bob_bases, bob_results):
    sifted_key = []
    for i in range(len(alice_bits)):
        if alice_bases[i] == bob_bases[i]:
            sifted_key.append(alice_bits[i])
    return sifted_key

def main():
    host = '127.0.0.1'
    port = 65432

    # Set up server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    print("Server is waiting for a connection...")

    conn, addr = server.accept()
    print(f"Connected by {addr}")

    # Alice's side of BB84
    num_bits = 16
    alice_bits, alice_bases, alice_qubits = bb84_alice(num_bits)
    conn.sendall(pickle.dumps(alice_qubits))  # Send qubits to client

    # Receive Bob's bases and measurement results
    bob_data = pickle.loads(conn.recv(4096))
    bob_bases, bob_results = bob_data['bases'], bob_data['results']

    # Sift the key
    sifted_key = sift_key(alice_bits, alice_bases, bob_bases, bob_results)
    print(f"Sifted Key: {sifted_key}")

    # Send Alice's bases for basis comparison
    conn.sendall(pickle.dumps(alice_bases))

    conn.close()

if __name__ == '__main__':
    main()
