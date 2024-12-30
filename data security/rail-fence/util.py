def rail_fence_encrypt(text, rails):
    fence = [['' for _ in range(len(text))] for _ in range(rails)]
    row, step = 0, 1

    for i, char in enumerate(text):
        fence[row][i] = char
        if row == 0:
            step = 1
        elif row == rails - 1:
            step = -1
        row += step

    encrypted_text = ''.join([''.join(row) for row in fence])
    return encrypted_text.replace('', '')

def rail_fence_decrypt(cipher, rails):
    fence = [['' for _ in range(len(cipher))] for _ in range(rails)]
    row, step = 0, 1

    for i in range(len(cipher)):
        fence[row][i] = '*'
        if row == 0:
            step = 1
        elif row == rails - 1:
            step = -1
        row += step

    idx = 0
    for r in range(rails):
        for c in range(len(cipher)):
            if fence[r][c] == '*':
                fence[r][c] = cipher[idx]
                idx += 1

    decrypted_text = []
    row, step = 0, 1
    for i in range(len(cipher)):
        decrypted_text.append(fence[row][i])
        if row == 0:
            step = 1
        elif row == rails - 1:
            step = -1
        row += step

    return ''.join(decrypted_text)
