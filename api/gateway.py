import subprocess
import time
import sys

def start_server():
    """Start the Flask API server."""
    print("Starting api server")
    server_process = subprocess.Popen([sys.executable, '-m', 'api.server'])
    time.sleep(5)  # Wait for the server to start
    return server_process

def start_device_clients(num_clients):
    """Start multiple device clients."""
    clients = []
    for i in range(num_clients):
        print(f"Starting device client {i + 1}...")
        client_process = subprocess.Popen([sys.executable, '-m', 'nn.device_client'])
        clients.append(client_process)
        time.sleep(5)
    return clients

if __name__ == "__main__":
    server_process = start_server()
    try:
        num_clients = int(input("Enter the number of device clients to start: "))
        clients = start_device_clients(num_clients)
        print("started")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for client in clients:
            client.terminate()
        server_process.terminate()
        print("stopped")