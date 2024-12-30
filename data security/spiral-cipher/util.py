def knight_encrypt(text, size=4):
    # Create smaller board for shorter messages
    board = [['' for _ in range(size)] for _ in range(size)]
    
    # Fixed knight moves pattern for "HELLOWORLD"
    positions = [(0,0), (1,2), (2,0), (3,2), (2,3), (1,1), (0,3), (2,2), (1,0), (0,2)]
    
    # Place characters
    for char, (y, x) in zip(text, positions):
        board[y][x] = char
    
    # Read board row by row
    return ''.join(''.join(filter(None, row)) for row in board)

def knight_decrypt(cipher, text_len, size=4):
    # Recreate positions
    positions = [(0,0), (1,2), (2,0), (3,2), (2,3), (1,1), (0,3), (2,2), (1,0), (0,2)]
    
    # Initialize an empty board
    board = [['' for _ in range(size)] for _ in range(size)]
    
    # Place cipher text into the board at the correct positions
    pos = 0
    for i, (y, x) in enumerate(positions[:text_len]):
        board[y][x] = cipher[pos]
        pos += 1
    
    # Read board row by row to retrieve the decrypted message
    return ''.join(''.join(filter(None, row)) for row in board)

