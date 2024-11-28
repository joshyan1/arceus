from flask import Flask, request, jsonify, Blueprint
from nn.coordinator import DistributedNeuralNetwork
import threading
import os

# Create a blueprint for the API
api_bp = Blueprint('api', __name__)

# Global state
coordinator = None
lock = threading.Lock()
registered_devices = {}
rooms = {}  # In-memory room storage

@api_bp.route('/api/devices/register', methods=['POST'])
def register_device():
    """Register a new device with its IP and port number"""
    data = request.get_json()
    
    if not data or 'port' not in data or 'ip' not in data:
        return jsonify({'error': 'IP and port number are required'}), 400
        
    port = data['port']
    ip = data['ip']
    device_address = f"{ip}:{port}"
    
    with lock:
        if device_address in registered_devices:
            return jsonify({'error': 'Device already registered'}), 409
            
        # Initialize coordinator if this is the first device
        global coordinator
        if coordinator is None:
            coordinator = DistributedNeuralNetwork(layer_sizes=[784, 128, 64, 10])
            
        # Try to connect to the device
        if coordinator.connect_to_device(ip, port):
            device_id = len(registered_devices) + 1
            registered_devices[device_address] = device_id
            return jsonify({
                'message': 'Device registered successfully',
                'device_id': device_id
            }), 201
        else:
            return jsonify({'error': 'Failed to connect to device'}), 500

@api_bp.route('/api/devices', methods=['GET'])
def get_devices():
    """Get list of registered devices"""
    return jsonify({
        'devices': [
            {'port': port, 'device_id': device_id} 
            for port, device_id in registered_devices.items()
        ],
        'max_devices': coordinator.max_devices if coordinator else 0,
        'connected_devices': len(registered_devices)
    })

@api_bp.route('/api/network/initialize', methods=['POST'])
def initialize_network():
    """Initialize the neural network across registered devices"""
    if not coordinator:
        return jsonify({'error': 'No devices registered'}), 400
        
    if len(registered_devices) == 0:
        return jsonify({'error': 'No devices available'}), 400
        
    try:
        coordinator.initialize_devices()
        return jsonify({'message': 'Network initialized successfully'})
    except Exception as e:
        return jsonify({'error': f'Failed to initialize network: {str(e)}'}), 500

@api_bp.route('/api/network/train', methods=['POST'])
def start_training():
    """Start the training process"""
    if not coordinator:
        return jsonify({'error': 'Network not initialized'}), 400
        
    data = request.get_json() or {}
    epochs = data.get('epochs', 10)
    learning_rate = data.get('learning_rate', 0.1)
    
    try:
        # Start training in a separate thread
        def train_thread():
            from torch.utils.data import DataLoader
            from torchvision import datasets, transforms
            
            # Define transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            # Load MNIST dataset
            train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
            val_dataset = datasets.MNIST('data', train=False, transform=transform)
            
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1000)
            
            coordinator.train(train_loader, val_loader, epochs=epochs, learning_rate=learning_rate)
        
        thread = threading.Thread(target=train_thread)
        thread.start()
        
        return jsonify({
            'message': 'Training started',
            'epochs': epochs,
            'learning_rate': learning_rate
        })
    except Exception as e:
        return jsonify({'error': f'Failed to start training: {str(e)}'}), 500

@api_bp.route('/api/devices/<int:port>', methods=['DELETE'])
def unregister_device(port):
    """Unregister a device"""
    device_address = next((addr for addr, id in registered_devices.items() if id == port), None)
    if device_address is None:
        return jsonify({'error': 'Device not found'}), 404
        
    with lock:
        del registered_devices[device_address]
        if len(registered_devices) == 0:
            global coordinator
            coordinator = None
            
        return jsonify({'message': 'Device unregistered successfully'})

# Room Management Endpoints

@api_bp.route('/api/rooms', methods=['GET'])
def get_rooms():
    """Get list of available training rooms"""
    return jsonify({'rooms': list(rooms.keys())})

@api_bp.route('/api/rooms/create', methods=['POST'])
def create_room():
    """Create a new training room"""
    data = request.get_json()
    room_name = data.get('room_name')
    
    if not room_name:
        return jsonify({'error': 'Room name is required'}), 400
    
    if room_name in rooms:
        return jsonify({'error': 'Room already exists'}), 409
    
    rooms[room_name] = []  # Initialize with an empty list of devices
    return jsonify({'message': 'Room created successfully', 'room_name': room_name}), 201

@api_bp.route('/api/rooms/join', methods=['POST'])
def join_room():
    """Join an existing training room"""
    data = request.get_json()
    room_name = data.get('room_name')
    device_id = data.get('device_id')
    
    if room_name not in rooms:
        return jsonify({'error': 'Room not found'}), 404
    
    rooms[room_name].append(device_id)  # Add device to the room
    return jsonify({'message': 'Joined room successfully', 'room_name': room_name}), 200

def create_app():
    """Create and configure the Flask app"""
    app = Flask(__name__)
    # Register the blueprint
    app.register_blueprint(api_bp)
    
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=4000, debug=True)