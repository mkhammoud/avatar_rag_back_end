import json
import time
from flask import Flask, jsonify, send_file, request
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, send, emit

from app.core.InnerSQLs import InnerSQLite
from app.core.Pipeline import Pipeline
from app.core.pipes.SQLRetrieval import SQLRetrieval
from app.core.pipes.VLLMPipe import VLLMPipe
from app.main import init_retrieval_pipe, init_llm
from socketio_setup import socketio, init_socketio
import threading
from app.core import middlewares
from avatar_utils import chunk_string,process_chunks_with_limit
import ast 

app = Flask("Avatar RAG Backend")
app_url = "http://localhost:5000/"
CORS(app, origins=['*'])

init_socketio(app)

global_config={
"max_text_character_chunk_size":200,
"max_text_to_speech_avatar_thread":2
}

def init_retrieval_pipe():
    # vllm serve meta-llama/Llama-3.2-1B-Instruct --dtype auto --api-key token-abc123
    pipe = SQLRetrieval(
        embedding_path='./post_embeddings.index',
        ids_path='./post_ids.npy',
        sql=InnerSQLite('./knowledge.db')
    )
    return pipe


def init_llm():
    pipe = VLLMPipe('meta-llama/Llama-3.2-1B-Instruct')
    return pipe


def init_pipline():
    pipeline = Pipeline()
    pipeline.queue(init_retrieval_pipe())
    #pipeline.queue(init_llm())
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
        if "avatarProvider" in request.form:
            avatarProvider=request.form["avatarProvider"]
        else:
            avatarProvider="local"


        if "messages" in request.form:
            messages_str = request.form['messages']
            if messages_str:
                messages = json.loads(messages_str)  # Convert JSON string back to array
                message = messages[-1]['content']

                print("LAST MESSAGE",message)

                pipe_out = pipeline.process(message, middlewares.contextualize(messages))
                
                try:
                    x_dict = ast.literal_eval(pipe_out)
                    pipe_out = x_dict['content']
                except Exception as e:
                    print(e)
                
                chunks = chunk_string(pipe_out,max_chunk_size=global_config['max_text_character_chunk_size'])
                
                if avatarProvider=="local":
                    print("LOCAL")
                    threading.Thread(target=process_chunks_with_limit, args=(chunks,global_config['max_text_to_speech_avatar_thread'])).start()
            
                return jsonify({'status': "success",
                                'text_response': pipe_out})

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
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
