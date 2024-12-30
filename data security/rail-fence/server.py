import asyncio
import websockets
from util import rail_fence_encrypt, rail_fence_decrypt

rails = 3

async def handle_client(websocket, path):
    print("Client connected")

    try:
        async for message in websocket:

            message = rail_fence_decrypt(message, rails)
            print(f"Received: {message}")

            message = rail_fence_encrypt(message, rails)
            await websocket.send(message)

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Client disconnected: {e}")

start_server = websockets.serve(handle_client, "localhost", 8080)

print("WebSocket server is running on ws://localhost:8080")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
