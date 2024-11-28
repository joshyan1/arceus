import os
import sys
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import grpc
from concurrent import futures
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from protos import device_service_pb2 as pb2
from protos import device_service_pb2_grpc as pb2_grpc

class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize weights using PyTorch
        self.W = torch.randn(input_size, output_size, device=self.device) * np.sqrt(2.0 / input_size)
        self.b = torch.zeros(1, output_size, device=self.device)
        self.W.requires_grad_(True)
        self.b.requires_grad_(True)
        
        self.activation = activation
        
    def forward(self, A_prev):
        """Forward pass for a single layer"""
        self.A_prev = A_prev
        self.Z = torch.mm(A_prev, self.W) + self.b
        
        if self.activation == 'relu':
            self.A = F.relu(self.Z)
        else:  # softmax
            self.A = F.softmax(self.Z, dim=1)
        return self.A
    
    def backward(self, dA):
        """Backward pass for a single layer"""
        m = self.A_prev.size(0)
        
        if self.activation == 'relu':
            dZ = dA * (self.Z > 0)
        else:  # softmax
            dZ = dA
            
        self.dW = torch.mm(self.A_prev.t(), dZ) / m
        self.db = torch.sum(dZ, dim=0, keepdim=True) / m
        dA_prev = torch.mm(dZ, self.W.t())
        
        return dA_prev
    
    def update(self, learning_rate):
        """Update parameters for a single layer"""
        with torch.no_grad():
            self.W -= learning_rate * self.dW
            self.b -= learning_rate * self.db

class Device:
    def __init__(self, layer_configs, device_id=0):
        """
        Initialize device with multiple layers
        layer_configs: list of dicts, each containing:
            - input_size
            - output_size
            - activation
        """
        self.device_id = device_id
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize layers
        self.layers = []
        for config in layer_configs:
            layer = Layer(
                input_size=config['input_size'],
                output_size=config['output_size'],
                activation=config['activation']
            )
            self.layers.append(layer)
        
    def quantize(self, x, bits=8):
        """Quantize gradients to reduce communication overhead"""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            
        max_val = np.max(np.abs(x))
        if max_val == 0:
            return x
        
        scale = (2 ** (bits - 1) - 1) / max_val
        quantized = np.round(x * scale)
        return quantized / scale
    
    def forward(self, A_prev):
        """Forward pass through all layers in this device"""
        # Convert numpy array to torch tensor if needed
        if isinstance(A_prev, np.ndarray):
            # Ensure input is 2D
            if len(A_prev.shape) == 1:
                A_prev = A_prev.reshape(1, -1)
            elif len(A_prev.shape) == 3:  # For MNIST images
                A_prev = A_prev.reshape(A_prev.shape[0], -1)
            elif len(A_prev.shape) == 4:  # For MNIST images with channel
                A_prev = A_prev.reshape(A_prev.shape[0], -1)
            
            A_prev = torch.from_numpy(A_prev).float().to(self.device)
        
        # Store all activations for backward pass
        self.activations = [A_prev]
        A = A_prev
        
        # Forward through each layer
        for layer in self.layers:
            A = layer.forward(A)
            self.activations.append(A)
            
        return A.detach().cpu().numpy()
    
    def backward(self, dA):
        """Backward pass through all layers in this device"""
        if isinstance(dA, np.ndarray):
            dA = torch.from_numpy(dA).float().to(self.device)
        
        # Backward through each layer in reverse
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            dA = layer.backward(dA)
            
        return dA.detach().cpu().numpy()
    
    def update(self, learning_rate):
        """Update parameters of all layers"""
        for layer in self.layers:
            layer.update(learning_rate)

class DeviceServicer(pb2_grpc.DeviceServiceServicer):
    def __init__(self):
        self.device = None
    
    def Initialize(self, request, context):
        """Initialize device with layer configurations"""
        layer_configs = [
            {
                'input_size': config.input_size,
                'output_size': config.output_size,
                'activation': config.activation
            }
            for config in request.layer_configs
        ]
        
        self.device = Device(layer_configs, device_id=request.device_id)
        
        return pb2.InitResponse(
            status='initialized',
            device_id=self.device.device_id
        )
    
    def Forward(self, request, context):
        """Forward pass through device layers"""
        input_array = np.array(request.input).reshape(request.input_shape)
        output = self.device.forward(input_array)
        
        return pb2.ForwardResponse(
            output=output.flatten().tolist(),
            output_shape=list(output.shape)
        )
    
    def Backward(self, request, context):
        """Backward pass through device layers"""
        grad_input = np.array(request.grad_input).reshape(request.grad_shape)
        grad_output = self.device.backward(grad_input)
        
        return pb2.BackwardResponse(
            grad_output=grad_output.flatten().tolist(),
            grad_shape=list(grad_output.shape)
        )
    
    def Update(self, request, context):
        """Update device parameters"""
        self.device.update(request.learning_rate)
        return pb2.UpdateResponse(status='updated')
    
    def Ping(self, request, context):
        """Health check"""
        return pb2.PingResponse(status='connection successful')

def serve(port):
    """Start gRPC server"""
    try:
        print(f"Creating gRPC server...")
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        
        print(f"Adding servicer...")
        pb2_grpc.add_DeviceServiceServicer_to_server(
            DeviceServicer(), server
        )
        
        # Try to bind to the port
        address = f'0.0.0.0:{port}'
        print(f"Binding to address: {address}")
        port = server.add_insecure_port(address)
        
        # Start the server
        print("Starting server...")
        server.start()
        print(f"Device server successfully started and listening on {address}")
        
        # Keep the server running
        server.wait_for_termination()
        
    except Exception as e:
        print(f"Failed to start server: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python device_server.py <port>", file=sys.stderr)
        sys.exit(1)
    
    try:
        port = int(sys.argv[1])
        print(f"Attempting to start server on port {port}...")
        serve(port)
    except ValueError:
        print(f"Invalid port number: {sys.argv[1]}", file=sys.stderr)
        sys.exit(1)