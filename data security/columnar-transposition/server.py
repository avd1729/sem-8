import asyncio
import websockets
from util import columnar_transposition_encrypt, columnar_transposition_decrypt

key = "world"
async def handle_client(websocket, path):
    print("Client connected")

    try:
        async for message in websocket:

            message = columnar_transposition_decrypt(message, key)
            print(f"Received: {message}")

            message = columnar_transposition_encrypt(message, key)
            await websocket.send(message)

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Client disconnected: {e}")

start_server = websockets.serve(handle_client, "localhost", 8080)

print("WebSocket server is running on ws://localhost:8080")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
