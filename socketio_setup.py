from flask_socketio import SocketIO, send, emit

socketio = SocketIO()

# Event for client connection
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    send('Welcome! You are connected.')

# Event for receiving a message from the client
@socketio.on('message')
def handle_message(message):
    print(f'Received message: {message}')
    send(f'Server received: {message}')

def init_socketio(app):
    socketio.init_app(app,cors_allowed_origins=["http://localhost:3000"])