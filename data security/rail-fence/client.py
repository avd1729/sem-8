import asyncio
import websockets
from util import rail_fence_encrypt, rail_fence_decrypt

message = "HELLOWORLD"
rails = 3

async def connect_to_server():
    uri = "ws://localhost:8080"

    async with websockets.connect(uri) as websocket:

        text = rail_fence_encrypt(message, rails)
        await websocket.send(text)

        print("Message sent to server")

        response = rail_fence_decrypt(await websocket.recv(), rails)
        print(f"Received from server: {response}")

asyncio.get_event_loop().run_until_complete(connect_to_server())
