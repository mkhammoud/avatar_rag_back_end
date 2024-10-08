import json
import time
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, send, emit

from app.core.Pipeline import Pipeline
from app.main import init_retrieval_pipe, init_llm
from socketio_setup import socketio, init_socketio

app = Flask("Avatar RAG Backend")
app_url = "http://localhost:5000/"
CORS(app, origins=['*'])

init_socketio(app)


def init_pipline():
    pipeline = Pipeline()
    pipeline.queue(init_retrieval_pipe())
    pipeline.queue(init_llm())
    pipeline.init()
    return pipeline


pipeline = init_pipline()


# GET IDLE AVATAR FOR LOCAL VIDEO

@app.route('/getIdleAvatar', methods=["POST", "OPTIONS"])
@cross_origin(supports_credentials=True)
def get_idle_avatar_route():
    try:
        if "avatarId" in request.form:
            avatarId = request.form["avatarId"]
            avatarProvider = request.form["avatarProvider"]

            # Simulate Text to speech avatar pipeline or so 

            video_path = "default_avatar.mp4"

            return jsonify({
                'status': "success",
                'video_url': f'{app_url}video/' + video_path,
            })


    except Exception as e:
        print(e)


@app.route('/handleUserQuery', methods=["POST", "OPTIONS"])
@cross_origin(supports_credentials=True)
def handle_user_query_route():
    try:

        if "messages" in request.form:
            messages_str = request.form['messages']
            messages = None
            if messages_str:
                messages = json.loads(messages_str)  # Convert JSON string back to array
                message = messages[-1]['content']
                pipe_out = pipeline.process(message)
                print(messages)

        # Simulate Search Call (you have to embed the user query in your search pipeline and then perform RAG)

        # Simulate LLM CALL After receiving Search scall

        # Simulate Avatar (Text to speech is already included in the new pipeline)

        # Return video

        with open("temp/result_video.mp4", 'rb') as f:
            video_bytes = f.read()
            # Emit the video chunk to the client
            socketio.emit('video_chunk', video_bytes)  # Send as hex string

        with open("temp/result_voice.mp4", 'rb') as f:
            video_bytes = f.read()
            # Emit the video chunk to the client
            socketio.emit('video_chunk', video_bytes)  # Send as hex string

        with open("temp/result_voice2.mp4", 'rb') as f:
            video_bytes = f.read()
            # Emit the video chunk to the client
            socketio.emit('video_chunk', video_bytes)  # Send as hex string

        return jsonify({'status': "success",
                        'text_response': "Hello! How can I assist you today? Feel free to ask me anything, whether it's about general knowledge, recommendations, or troubleshooting. I'm here to help!"})

    except Exception as e:
        print(e)


@app.route('/video/<file_path>', methods=['GET'])
def serve_video(file_path):
    # Serve video file from local storage
    try:
        video_path = f'temp/{file_path}'
        return send_file(video_path, mimetype='video/mp4')

    except Exception as e:
        print(e)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
