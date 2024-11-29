from flask import Flask, request, jsonify
from nn.coordinator import DistributedNeuralNetwork, previous_training_sessions
import threading
import os
from typing import Dict, Optional
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
import numpy as np
from api.socket import socketio

# Create Flask app
app = Flask(__name__)
# Initialize SocketIO with the Flask app
socketio.init_app(app)

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# Global state
coordinator = None
lock = threading.Lock()
registered_devices = {}
active_jobs: Dict[str, dict] = {}  # job_id -> job_config
aggregated_teraflops_data = {}

@app.route('/api/jobs', methods=['POST'])
def create_job():
    """Create a new distributed training job"""
    data = request.get_json()
    
    if not data or 'model_config' not in data or 'dataset_config' not in data:
        return jsonify({'error': 'Model and dataset configurations are required'}), 400
        
    try:
        job_id = str(len(active_jobs) + 1)  # Simple job ID generation
        job_config = {
            'model_config': data['model_config'],
            'dataset_config': data['dataset_config'],
            'devices': {},
            'coordinator': None,
            'status': 'created'  # created -> initialized -> training -> completed
        }
        active_jobs[job_id] = job_config
        
        return jsonify({
            'message': 'Job created successfully',
            'job_id': job_id
        }), 201
        
    except Exception as e:
        return jsonify({'error': f'Failed to create job: {str(e)}'}), 500

@app.route('/api/jobs', methods=['GET'])
def get_jobs():
    """Get list of all jobs"""
    return jsonify({
        'jobs': [{
            'job_id': job_id,
            'status': config['status'],
            'devices': len(config['devices']),
            'model_config': config['model_config'],
            'dataset_config': config['dataset_config']
        } for job_id, config in active_jobs.items()]
    })

@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    """Get details of a specific job"""
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
        
    config = active_jobs[job_id]
    return jsonify({
        'job_id': job_id,
        'status': config['status'],
        'devices': len(config['devices']),
        'model_config': config['model_config'],
        'dataset_config': config['dataset_config']
    })

@app.route('/api/devices/register', methods=['POST'])
def register_device():
    """Register a new device with its IP, port number, and job ID"""
    data = request.get_json()
    
    if not data or 'port' not in data or 'ip' not in data or 'job_id' not in data:
        return jsonify({'error': 'IP, port number, and job ID are required'}), 400
        
    job_id = data['job_id']
    if job_id not in active_jobs:
        return jsonify({'error': 'Invalid job ID'}), 404
        
    port = data['port']
    ip = data['ip']
    device_address = f"{ip}:{port}"
    
    with lock:
        job_config = active_jobs[job_id]
        
        if device_address in job_config['devices']:
            return jsonify({'error': 'Device already registered to this job'}), 409
            
        # Initialize coordinator if this is the first device
        if job_config['coordinator'] is None:
            layer_sizes = job_config['model_config']['layer_sizes']
            job_config['coordinator'] = DistributedNeuralNetwork(layer_sizes=layer_sizes)
            
        coordinator = job_config['coordinator']
        
        # Try to connect to the device
        if coordinator.connect_to_device(ip, port):
            device_id = len(job_config['devices']) + 1
            job_config['devices'][device_address] = device_id
            return jsonify({
                'message': 'Device registered successfully',
                'device_id': device_id
            }), 201
        else:
            return jsonify({'error': 'Failed to connect to device'}), 500

@app.route('/api/network/initialize/<job_id>', methods=['POST'])
def initialize_network(job_id):
    """Initialize the neural network for a specific job"""
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
        
    job_config = active_jobs[job_id]
    
    if not job_config['coordinator']:
        return jsonify({'error': 'No devices registered'}), 400
        
    if len(job_config['devices']) == 0:
        return jsonify({'error': 'No devices available'}), 400
        
    try:
        job_config['coordinator'].initialize_devices()
        job_config['status'] = 'initialized'
        return jsonify({'message': 'Network initialized successfully'})
    except Exception as e:
        return jsonify({'error': f'Failed to initialize network: {str(e)}'}), 500

@app.route('/api/network/train/<job_id>', methods=['POST'])
def start_training(job_id):
    """Start the training process for a specific job"""
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
        
    job_config = active_jobs[job_id]
    
    if not job_config['coordinator']:
        return jsonify({'error': 'Network not initialized'}), 400
        
    if job_config['status'] != 'initialized':
        return jsonify({'error': 'Network must be initialized before training'}), 400
        
    data = request.get_json() or {}
    epochs = data.get('epochs', 10)
    learning_rate = data.get('learning_rate', 0.1)
    
    try:
        # Start training in a separate thread
        def train_thread():
            dataset_config = job_config['dataset_config']
            
            # Initialize dataset based on configuration
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(**dataset_config.get('normalize', {'mean': (0.1307,), 'std': (0.3081,)}))
            ])
            
            # Load dataset based on configuration
            dataset_class = getattr(datasets, dataset_config['name'])
            train_dataset = dataset_class(
                'data', 
                train=True, 
                download=True, 
                transform=transform,
                **dataset_config.get('args', {})
            )
            val_dataset = dataset_class(
                'data', 
                train=False, 
                transform=transform,
                **dataset_config.get('args', {})
            )
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=dataset_config.get('batch_size', 256), 
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=dataset_config.get('val_batch_size', 1000)
            )
            
            job_config['status'] = 'training'
            job_config['coordinator'].train(
                train_loader, 
                val_loader, 
                epochs=epochs, 
                learning_rate=learning_rate
            )
            print("Training completed. Aggregating teraflops data...")
            job_config['status'] = 'completed'
        
        thread = threading.Thread(target=train_thread)
        thread.start()
        
        return jsonify({
            'message': 'Training started',
            'epochs': epochs,
            'learning_rate': learning_rate
        })
    except Exception as e:
        return jsonify({'error': f'Failed to start training: {str(e)}'}), 500

@app.route('/api/devices/<int:port>', methods=['DELETE'])
def unregister_device(port):
    """Unregister a device"""
    if port not in registered_devices:
        return jsonify({'error': 'Device not found'}), 404
        
    with lock:
        del registered_devices[port]
        if len(registered_devices) == 0:
            global coordinator
            coordinator = None
            
        return jsonify({'message': 'Device unregistered successfully'})

@app.route('/api/devices/teraflops', methods=['GET'])
def get_device_teraflops():
    """Endpoint to retrieve teraflops data from all devices."""
    if coordinator is None:
        return jsonify({'error': 'No active jobs or devices found'}), 404

    teraflops_data = coordinator.get_device_teraflops()
    
    # Print the teraflops data for each device
    for device_id, tflops in teraflops_data.items():
        print(f"Device {device_id} - Forward TFLOPs: {tflops['forward_tflops']:.4f}, Backward TFLOPs: {tflops['backward_tflops']:.4f}")

    return jsonify(teraflops_data), 200

@app.route('/api/teraflops', methods=['GET'])
def get_aggregated_teraflops():
    """Endpoint to retrieve aggregated teraflops data."""
    if coordinator is None or not hasattr(coordinator, 'teraflops_data') or not coordinator.teraflops_data:
        return jsonify({'error': 'No teraflops data available'}), 404

    return jsonify(coordinator.teraflops_data), 200

@app.route('/api/previous_teraflops', methods=['GET'])
def get_previous_teraflops():
    """Endpoint to retrieve teraflops data from previous training sessions."""
    with lock:
        if not previous_training_sessions:
            return jsonify({'error': 'No previous training sessions found'}), 404

        print(f"Previous training sessions: {previous_training_sessions}")  # Log the previous sessions
        return jsonify(previous_training_sessions), 200

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=4000, allow_unsafe_werkzeug=True)