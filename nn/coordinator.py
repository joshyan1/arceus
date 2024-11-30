import grpc
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import time
from protos import device_service_pb2 as pb2
from protos import device_service_pb2_grpc as pb2_grpc
from threading import Lock


# Global variable to store previous training sessions' teraflops data
previous_training_sessions = []
lock = Lock()

class DistributedNeuralNetwork:
    def __init__(self, layer_sizes, quantization_bits=8):
        self.layer_sizes = layer_sizes
        self.max_devices = len(layer_sizes) - 1
        self.device_connections = {}
        self.device_layer_map = {}
        self.quantization_bits = quantization_bits
        self.job_id = 'training_room'  # Use a default room or pass it as a parameter
        self.message_queue = None

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.timing_stats = {
            'forward_time': [],
            'backward_time': [],
            'update_time': [],
            'communication_time': [],
            'data_prep_time': []
        }
        print(f"Using device: {self.device}")
        print(f"Maximum number of devices: {self.max_devices}")
    
    def connect_to_device(self, ip, port):
        """Connect to a device using gRPC"""
        if len(self.device_connections) >= self.max_devices:
            print(f"Maximum number of devices ({self.max_devices}) reached")
            return False
        
        try:
            channel = grpc.insecure_channel(f'{ip}:{port}')
            stub = pb2_grpc.DeviceServiceStub(channel)
            
            # Test connection
            response = stub.Ping(pb2.PingRequest())
            if response.status != 'connection successful':
                return False
            
            device_id = len(self.device_connections) + 1
            self.device_connections[device_id] = stub
            print(f"Connected to device {device_id} at {ip}:{port}")
            return True
            
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def initialize_devices(self):
        """Initialize devices with balanced layer distribution"""
        num_devices = len(self.device_connections)
        num_layers = len(self.layer_sizes) - 1
        
        layers_per_device = num_layers // num_devices
        extra_layers = num_layers % num_devices
        
        print(f"\nDistributing {num_layers} layers across {num_devices} devices")
        
        current_layer = 0
        device_layer_map = {}
        
        for device_id, stub in self.device_connections.items():
            n_layers = layers_per_device + (1 if device_id <= extra_layers else 0)
            layer_configs = []
            layer_indices = []
            
            for _ in range(n_layers):
                if current_layer < len(self.layer_sizes) - 1:
                    config = pb2.LayerConfig(
                        input_size=self.layer_sizes[current_layer],
                        output_size=self.layer_sizes[current_layer + 1],
                        activation='relu' if current_layer < len(self.layer_sizes) - 2 else 'softmax'
                    )
                    layer_configs.append(config)
                    layer_indices.append(current_layer)
                    current_layer += 1
            
            device_layer_map[device_id] = layer_indices
            
            response = stub.Initialize(pb2.InitRequest(
                layer_configs=layer_configs,
                device_id=device_id
            ))
            print(f"Initialized device {device_id} with layers {layer_indices}")
        
        self.device_layer_map = device_layer_map
        print("\nLayer distribution complete")
    
    def forward(self, X):
        """Distributed forward pass with timing"""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(self.device)
        
        start_prep = time.time()
        A = X
        activations = [X]
        prep_time = time.time() - start_prep
        self.timing_stats['data_prep_time'].append(prep_time)
        
        total_forward_time = 0
        total_comm_time = 0
        
        for device_id in sorted(self.device_connections.keys()):
            stub = self.device_connections[device_id]
            
            # Convert to numpy for gRPC transfer
            start_comm = time.time()
            A_numpy = A.detach().cpu().numpy()
            forward_request = pb2.ForwardRequest(
                input_data=A_numpy.tobytes(),
                input_shape=list(A_numpy.shape)
            )
            comm_time = time.time() - start_comm
            
            # Time forward computation
            start_forward = time.time()
            response = stub.Forward(forward_request)
            forward_time = time.time() - start_forward
            
            # Convert back to PyTorch tensor
            start_comm = time.time()
            A = torch.from_numpy(
                np.frombuffer(response.output_data, dtype=np.float32).reshape(response.output_shape)
            ).to(self.device)
            activations.append(A)
            comm_time += time.time() - start_comm
            
            total_forward_time += forward_time
            total_comm_time += comm_time
        
        self.timing_stats['forward_time'].append(total_forward_time)
        self.timing_stats['communication_time'].append(total_comm_time)
        return activations
    
    def backward(self, activations, y_true):
        """Distributed backward pass with timing"""
        start_prep = time.time()
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true).to(self.device)
        
        # Create one-hot encoding using PyTorch
        batch_size = y_true.shape[0]
        y_onehot = torch.zeros(batch_size, activations[-1].shape[1], device=self.device)
        y_onehot.scatter_(1, y_true.unsqueeze(1), 1)
        dA = activations[-1] - y_onehot
        prep_time = time.time() - start_prep
        self.timing_stats['data_prep_time'][-1] += prep_time
        
        total_backward_time = 0
        total_comm_time = 0
        
        for device_id in sorted(self.device_connections.keys(), reverse=True):
            stub = self.device_connections[device_id]
            
            # Convert to numpy for gRPC transfer
            start_comm = time.time()
            dA_numpy = dA.detach().cpu().numpy()
            backward_request = pb2.BackwardRequest(
                grad_data=dA_numpy.tobytes(),
                grad_shape=list(dA_numpy.shape)
            )
            comm_time = time.time() - start_comm
            
            start_backward = time.time()
            response = stub.Backward(backward_request)
            backward_time = time.time() - start_backward
            
            # Convert back to PyTorch tensor
            start_comm = time.time()
            dA = torch.from_numpy(
                np.frombuffer(response.grad_output_data, dtype=np.float32).reshape(response.grad_shape)
            ).to(self.device)
            comm_time += time.time() - start_comm
            
            total_backward_time += backward_time
            total_comm_time += comm_time
        
        self.timing_stats['backward_time'].append(total_backward_time)
        self.timing_stats['communication_time'][-1] += total_comm_time
    
    def update_parameters(self, learning_rate):
        """Update parameters with timing"""
        start_update = time.time()
        for device_id in sorted(self.device_connections.keys()):
            stub = self.device_connections[device_id]
            stub.Update(pb2.UpdateRequest(learning_rate=learning_rate))
        self.timing_stats['update_time'].append(time.time() - start_update)
        
    
    def print_timing_stats(self, batch_idx):
        """Print timing statistics"""
        if batch_idx % 10 == 0:  # Print every 10 batches
            avg_forward = np.mean(self.timing_stats['forward_time'][-10:]) if self.timing_stats['forward_time'] else 0
            avg_backward = np.mean(self.timing_stats['backward_time'][-10:]) if self.timing_stats['backward_time'] else 0
            avg_update = np.mean(self.timing_stats['update_time'][-10:]) if self.timing_stats['update_time'] else 0
            avg_comm = np.mean(self.timing_stats['communication_time'][-10:]) if self.timing_stats['communication_time'] else 0
            avg_prep = np.mean(self.timing_stats['data_prep_time'][-10:]) if self.timing_stats['data_prep_time'] else 0
            
            print(f"\nTiming Statistics (last 10 batches):")
            print(f"Forward pass: {avg_forward:.4f}s")
            print(f"Backward pass: {avg_backward:.4f}s")
            print(f"Parameter update: {avg_update:.4f}s")
            print(f"Communication overhead: {avg_comm:.4f}s")
            print(f"Data preparation: {avg_prep:.4f}s")
            print(f"Total computation: {avg_forward + avg_backward + avg_update:.4f}s")
            print(f"Total overhead: {avg_comm + avg_prep:.4f}s")
            print("-" * 50)

            print("Sending timing stats to frontend")

            # Assuming this is part of your existing method in coordinator.py
            self.teraflops_data = {}
            forward_totals = []
            backward_totals = []

            for device_id, stub in self.device_connections.items():
                try:
                    request = pb2.TeraflopsRequest(device_id=device_id)
                    response = stub.GetTeraflops(request)
                    
                    # Store the individual device data
                    self.teraflops_data[device_id] = {
                        'forward_tflops': response.forward_tflops,
                        'backward_tflops': response.backward_tflops,
                        'total_tflops': response.forward_tflops + response.backward_tflops
                    }
                    
                    # Print individual device data
                    print(f"Device {device_id} - Forward TFLOPs: {response.forward_tflops}, Backward TFLOPs: {response.backward_tflops}")
                    
                    # Accumulate totals for averaging
                    forward_totals.append(response.forward_tflops)
                    backward_totals.append(response.backward_tflops)

                except Exception as e:
                    print(f"Error retrieving teraflops for device {device_id}: {e}")

            # UPDATE THIS LATER
            device_specific_data = []
            for device_id in self.device_connections.keys():
                device_specific_data.append({'device_id': device_id, 'total_teraflops': self.teraflops_data[device_id]['total_tflops'], 'chip': 'M2', 'device_layers': self.device_layer_map})

            # Instead of emitting directly, put the message in the queue
            if self.message_queue:
                self.message_queue.put({
                    'event': 'timing_stats',
                    'data': {
                        'avg_forward': avg_forward,
                        'avg_backward': avg_backward,
                        'avg_update': avg_update,
                        'avg_comm': avg_comm,
                        'avg_prep': avg_prep,
                        'device_data': device_specific_data,
                        'batch_idx': batch_idx
                    },
                    'room': self.job_id
                })

    def train(self, train_loader, message_queue, val_loader, epochs=10, learning_rate=0.1):
        self.message_queue = message_queue
        print("\nStarting distributed training across devices...")
        print(f"Learning rate: {learning_rate}")
        print(f"Quantization bits: {self.quantization_bits}")
        print(f"Device: {self.device}")
        print("-" * 100)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_loss = 0
            epoch_acc = 0
            n_batches = 0
            
            # Training loop
            for batch_idx, (data, target) in enumerate(train_loader):
                batch_start = time.time()
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                activations = self.forward(data)
                y_pred = activations[-1]
                
                batch_loss = self.compute_loss(target, y_pred)
                batch_acc = self.compute_accuracy(target, y_pred)
                
                epoch_loss += batch_loss
                epoch_acc += batch_acc
                n_batches += 1
                
                # Backward pass and update
                self.backward(activations, target)
                self.update_parameters(learning_rate)
                
                batch_time = time.time() - batch_start
                
                if (batch_idx + 1) % 20 == 0:
                    print(f"\rEpoch {epoch+1}/{epochs} "
                          f"[Batch {batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {batch_loss:.4f} "
                          f"Acc: {batch_acc:.4f} "
                          f"Time: {batch_time:.2f}s")
                    if self.message_queue:
                        self.message_queue.put({
                            'event': 'training_data',
                            'data': {
                                'epoch': epoch + 1,
                                'epochs': epochs,
                                'train_loss': float(batch_loss),
                                'train_acc': float(batch_acc), 
                                'batch_idx': batch_idx + 1,
                                'batch_time': batch_time
                            },
                            'room': self.job_id
                        })
                    self.print_timing_stats(batch_idx + 1)

            # Accidentally deleted this validation loop
            epoch_loss /= n_batches
            epoch_acc /= n_batches
            
            # Validation loop
            val_loss = 0
            val_acc = 0
            n_val_batches = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data = data.to(self.device)
                    target = target.to(self.device)
                    
                    val_activations = self.forward(data)
                    val_pred = val_activations[-1]
                    val_loss += self.compute_loss(target, val_pred)
                    val_acc += self.compute_accuracy(target, val_pred)
                    n_val_batches += 1

            val_loss /= n_val_batches
            val_acc /= n_val_batches
            epoch_time = time.time() - epoch_start

            if self.message_queue:
                self.message_queue.put({
                    'event': 'epoch_stats',
                    'data': {
                        'epoch': epoch + 1,
                        'epochs': epochs,
                        'train_loss': float(epoch_loss),
                        'train_acc': float(epoch_acc), 
                        'val_loss': float(val_loss),
                        'val_acc': float(val_acc),
                        'epoch_time': epoch_time
                    },
                    'room': self.job_id
                })
            
            print(f"\nEpoch {epoch+1:2d}/{epochs} - "
                  f"Loss: {epoch_loss:.4f} - "
                  f"Acc: {epoch_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"Val Acc: {val_acc:.4f}")
            print("-" * 100)
            
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1} completed in {epoch_time:.2f}s")
        
        # Aggregate teraflops data after training
        self.aggregate_teraflops()

    def aggregate_teraflops(self):
        """Aggregate teraflops data from all devices."""
        self.teraflops_data = {}
        for device_id, stub in self.device_connections.items():
            try:
                request = pb2.TeraflopsRequest(device_id=device_id)
                response = stub.GetTeraflops(request)
                self.teraflops_data[device_id] = {
                    'forward_tflops': response.forward_tflops,
                    'backward_tflops': response.backward_tflops,
                    'total_tflops': response.forward_tflops + response.backward_tflops
                }
                print(f"Device {device_id} - Forward TFLOPs: {response.forward_tflops}, Backward TFLOPs: {response.backward_tflops}")
            except Exception as e:
                print(f"Error retrieving teraflops for device {device_id}: {e}")

        # Log the aggregated teraflops data
        # print(f"Aggregated teraflops data: {self.teraflops_data}")

        # Store the aggregated data in the global previous training sessions list
        with lock:
            previous_training_sessions.append(self.teraflops_data)

    def compute_loss(self, y_true, y_pred):
        """Compute cross entropy loss using PyTorch"""
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true).to(self.device)
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred).to(self.device)
            
        batch_size = y_true.shape[0]
        y_onehot = torch.zeros(batch_size, y_pred.shape[1], device=self.device)
        y_onehot.scatter_(1, y_true.unsqueeze(1), 1)
        
        # Compute cross entropy loss manually for numerical stability
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        loss = -torch.sum(y_onehot * torch.log(y_pred)) / batch_size
        return loss.item()

    def compute_accuracy(self, y_true, y_pred):
        """Compute accuracy using PyTorch"""
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true).to(self.device)
        if isinstance(y_pred, np.ndarray):
            y_pred = torch.from_numpy(y_pred).to(self.device)
            
        predictions = torch.argmax(y_pred, dim=1)
        return (predictions == y_true).float().mean().item()

def main():
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset using PyTorch
    print("\nLoading MNIST dataset...")
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000)
    
    layer_sizes = [784, 128, 64, 10]
    nn = DistributedNeuralNetwork(layer_sizes)
    
    # Connect to all available devices up to max_devices
    ports = [5001, 5002]  # Can be extended or loaded from config
    connected_ports = []
    
    print("\nAttempting to connect to available devices...")
    for port in ports:
        if len(connected_ports) >= nn.max_devices:
            print(f"\nReached maximum number of devices ({nn.max_devices}). Stopping connection attempts.")
            break
            
        if nn.connect_to_device(port):
            connected_ports.append(port)
    
    if not connected_ports:
        print("No devices connected. Exiting.")
        return
        
    print(f"\nSuccessfully connected to {len(connected_ports)} devices")
    
    if len(connected_ports) < len(layer_sizes) - 1:
        print(f"Warning: Running with fewer devices ({len(connected_ports)}) than layers ({len(layer_sizes) - 1})")
        print("Some devices will handle multiple layers")
    
    response = input("\nStart training? (y/n): ").lower()
    if response != 'y':
        print("Training cancelled.")
        return
    
    nn.initialize_devices()
    nn.train(train_loader, val_loader)

if __name__ == "__main__":
    main() 