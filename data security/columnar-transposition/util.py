def columnar_transposition_encrypt(text, key):
    columns = ['' for _ in key]
    for i, char in enumerate(text):
        columns[i % len(key)] += char

    key_order = sorted(range(len(key)), key=lambda x: key[x])
    encrypted_text = ''.join(columns[i] for i in key_order)
    return encrypted_text

def columnar_transposition_decrypt(cipher, key):
    n_cols = len(key)
    n_rows = len(cipher) // n_cols
    extra_chars = len(cipher) % n_cols

    columns = ['' for _ in key]
    key_order = sorted(range(len(key)), key=lambda x: key[x])

    idx = 0
    for i in key_order:
        col_length = n_rows + (1 if i < extra_chars else 0)
        columns[i] = cipher[idx:idx + col_length]
        idx += col_length

    text = ''
    for i in range(n_rows + (1 if extra_chars else 0)):
        for col in columns:
            if i < len(col):
                text += col[i]
    return text
