import asyncio
import websockets
from util import encrypt, decrypt

async def connect_to_server():
    uri = "ws://localhost:8080"
    
    async with websockets.connect(uri) as websocket:

        text = encrypt("HelloServer", 5)
        await websocket.send(text)

        print("Message sent to server")

        response = decrypt(await websocket.recv(), 5)
        print(f"Received from server: {response}")

asyncio.get_event_loop().run_until_complete(connect_to_server())
