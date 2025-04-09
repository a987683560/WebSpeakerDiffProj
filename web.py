import asyncio
import json
import websockets
from queue import Queue
from aioconsole import ainput
from manager import *

# 存储所有连接的 WebSocket 客户端
connected_clients = set()
audio_recorder = AudioRecorder()
audio_recorder.start_audio_handle_loop()


# 处理 WebSocket 连接
async def handle_connection(websocket, path):
    # 将新连接的客户端添加到集合中
    connected_clients.add(websocket)
    try:
        while True:
            audio_data = await websocket.recv()
            # print(audio_data)
            audio_recorder.audio_frames.put(audio_data)
    except websockets.exceptions.ConnectionClosedOK:
        print("Connection closed normally")
    except websockets.exceptions.ConnectionClosedError:
        print("Connection closed with error")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # 连接关闭时从集合中移除客户端
        connected_clients.remove(websocket)


# 启动 WebSocket 服务器
async def start_websocket_server():
    server = await websockets.serve(handle_connection, "0.0.0.0", 8765)
    print("WebSocket server started on ws://0.0.0.0:8765")
    await server.wait_closed()


# 后端接收消息并发送给客户端
async def send_messages():
    while True:
        try:
            if not text_speaker_id.empty():
                text_id_data = text_speaker_id.get()
                json_data = json.dumps({
                    'id': text_id_data[0],
                    'name': text_id_data[1],
                    'text': text_id_data[2],
                })
                # 遍历所有连接的客户端并发送文本数据
                for client in connected_clients.copy():
                    try:
                        for i in range(1):
                            await client.send(json_data)
                        print('send msg', json_data)
                    except Exception as e:
                        print(f"Error sending message to client: {e}")
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error getting text: {e}")


# 模拟向 text_speaker_id 队列添加数据


# 运行异步事件循环
async def main():
    # 启动 WebSocket 服务器
    server_task = asyncio.create_task(start_websocket_server())
    # 启动发送消息任务
    send_task = asyncio.create_task(send_messages())
    # 启动模拟数据输入任务
    await asyncio.gather(server_task, send_task)


if __name__ == "__main__":
    # 安装 aioconsole 库
    # pip install aioconsole
    asyncio.run(main())
