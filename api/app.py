# app.py
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

# Initialize Flask app
app = Flask(__name__)

# Allow CORS from both ports
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000",
     "http://localhost:8000", "http://127.0.0.1:3000", "http://127.0.0.1:8000"]}})

# Initialize Socket.IO with multiple allowed origins
socketio = SocketIO(
    app,
    cors_allowed_origins=["http://localhost:3000", "http://localhost:8000",
                          "http://127.0.0.1:3000", "http://127.0.0.1:8000"],
    async_mode='threading'
)

print('SocketIO async_mode is', socketio.async_mode)
