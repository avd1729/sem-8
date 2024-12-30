import asyncio
import websockets
from util import columnar_transposition_encrypt, columnar_transposition_decrypt

key = "world"
async def connect_to_server():
    uri = "ws://localhost:8080"

    async with websockets.connect(uri) as websocket:

        text = columnar_transposition_encrypt("hello", key)
        await websocket.send(text)

        print("Message sent to server")

        response = columnar_transposition_decrypt(await websocket.recv(), key)
        print(f"Received from server: {response}")

asyncio.get_event_loop().run_until_complete(connect_to_server())
