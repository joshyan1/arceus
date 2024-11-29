from flask_socketio import SocketIO, emit, Namespace
from threading import Lock
import json
import numpy as np
from flask import request

# Initialize SocketIO
socketio = SocketIO(cors_allowed_origins="*")

# Thread synchronization
thread = None
thread_lock = Lock()

class TrainingNamespace(Namespace):
    """Namespace for training-related events"""
    
    def __init__(self):
        super().__init__('/training')
        self.clients = set()
        
    def on_connect(self):
        """Handle client connection"""
        client_id = request.sid
        self.clients.add(client_id)
        print(f"Client {client_id} connected")
        
    def on_disconnect(self):
        """Handle client disconnection"""
        client_id = request.sid
        self.clients.remove(client_id)
        print(f"Client {client_id} disconnected")
        
    def emit_training_update(self, data):
        """Emit training update to all connected clients"""
        self.emit('training_update', data)
        
    def emit_batch_complete(self, data):
        """Emit batch completion update"""
        self.emit('batch_complete', data)
        
    def emit_epoch_complete(self, data):
        """Emit epoch completion update"""
        self.emit('epoch_complete', data)
        
    def emit_teraflops_update(self, data):
        """Emit teraflops update"""
        self.emit('teraflops_update', data)
        
    def emit_activation_update(self, data):
        """Emit activation update"""
        self.emit('activation_update', data)

# Create namespace instance
training_namespace = TrainingNamespace()

# Register namespace
socketio.on_namespace(training_namespace)
