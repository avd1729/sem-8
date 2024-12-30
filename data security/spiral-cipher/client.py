import asyncio
import websockets
from util import knight_encrypt, knight_decrypt

message = "HELLOWORLD"

async def connect_to_server():
    uri = "ws://localhost:8080"  # WebSocket server URI

    async with websockets.connect(uri) as websocket:
        # Encrypt the message before sending
        encrypted_message = knight_encrypt(message)
        await websocket.send(encrypted_message)

        print("Message sent to server")

        # Receive and decrypt the message from the server
        response = await websocket.recv()
        decrypted_message = knight_decrypt(response, len(response))
        print(f"Received from server: {decrypted_message}")

# Run the client
asyncio.get_event_loop().run_until_complete(connect_to_server())
