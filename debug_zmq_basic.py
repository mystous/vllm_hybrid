
import zmq
import multiprocessing
import os
import time

def server_func(url):
    print(f"Server: process {os.getpid()} starting")
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    try:
        socket.bind(url)
        print(f"Server: bound to {url}")
        
        message = socket.recv()
        print(f"Server: received request: {message}")
        socket.send(b"World")
        print("Server: sent reply")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        socket.close()
        context.term()

def client_func(url):
    print(f"Client: process {os.getpid()} starting")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    try:
        socket.connect(url)
        print(f"Client: connected to {url}")
        
        print("Client: sending Hello")
        socket.send(b"Hello")
        reply = socket.recv()
        print(f"Client: received reply: {reply}")
    except Exception as e:
        print(f"Client error: {e}")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    ipc_path = f"ipc:///tmp/debug_zmq_{os.getpid()}"
    
    server = multiprocessing.Process(target=server_func, args=(ipc_path,))
    client = multiprocessing.Process(target=client_func, args=(ipc_path,))
    
    server.start()
    time.sleep(1) # Give server time to bind
    client.start()
    
    server.join(timeout=5)
    client.join(timeout=5)
    
    if server.exitcode is None:
        print("Server timed out")
        server.terminate()
    if client.exitcode is None:
        print("Client timed out")
        client.terminate()
        
    print("Test complete")
