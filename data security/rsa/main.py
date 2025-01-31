from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256


def generate_rsa_keys():
    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key = key.publickey().export_key()
    return private_key, public_key

def encrypt_message(message, receiver_public_key):
    rsa_key = RSA.import_key(receiver_public_key)
    cipher = PKCS1_OAEP.new(rsa_key)
    encrypted_message = cipher.encrypt(message.encode())
    return encrypted_message


def decrypt_message(encrypted_message, receiver_private_key):
    rsa_key = RSA.import_key(receiver_private_key)
    cipher = PKCS1_OAEP.new(rsa_key)
    decrypted_message = cipher.decrypt(encrypted_message)
    return decrypted_message.decode()


def sign_message(message, sender_private_key):
    rsa_key = RSA.import_key(sender_private_key)
    message_hash = SHA256.new(message.encode())
    signature = pkcs1_15.new(rsa_key).sign(message_hash)
    return signature


def verify_signature(message, signature, sender_public_key):
    rsa_key = RSA.import_key(sender_public_key)
    message_hash = SHA256.new(message.encode())
    try:
        pkcs1_15.new(rsa_key).verify(message_hash, signature)
        return True 
    except (ValueError, TypeError):
        return False  

# Main execution
if __name__ == "__main__":
    
    sender_private, sender_public = generate_rsa_keys()
    receiver_private, receiver_public = generate_rsa_keys()

    
    message = "This is a secure message."

    signature = sign_message(message, sender_private)

    encrypted_message = encrypt_message(message, receiver_public)

    decrypted_message = decrypt_message(encrypted_message, receiver_private)

    is_valid = verify_signature(decrypted_message, signature, sender_public)

    print("Original Message: ", message)
    print("Decrypted Message: ", decrypted_message)
    print("Is the Signature Valid?: ", is_valid)
