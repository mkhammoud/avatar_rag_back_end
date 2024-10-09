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
import re
from avatar.speech import synthesize
import threading
from app.core import middlewares

app = Flask("Avatar RAG Backend")
app_url = "http://localhost:5000/"
CORS(app, origins=['*'])

init_socketio(app)

def chunk_string(text, max_chunk_size=300):
    # Use regex to split by words, while keeping the words intact.
    words = re.findall(r'\S+\s*', text)
    
    chunks = []
    current_chunk = ""
    
    for word in words:
        # If adding the next word exceeds the max_chunk_size, finalize the current chunk.
        if len(current_chunk) + len(word) > max_chunk_size:
            chunks.append(current_chunk.strip() + ".")
            current_chunk = word
        else:
            current_chunk += word
    
    # Add the last chunk if there's any leftover content
    if current_chunk:
        chunks.append(current_chunk.strip() + ".")
    
    return chunks

def process_chunks_with_limit(chunks, max_threads=1):
    active_threads = []
    
    for chunk in chunks:
        # If the number of active threads reaches the limit, wait for one to finish
        while len(active_threads) >= max_threads:
            for thread in active_threads:
                if not thread.is_alive():
                    active_threads.remove(thread)
            time.sleep(0.1)  # Give a little delay to reduce CPU usage during the loop

        # Start a new thread for the current chunk
        thread = threading.Thread(target=synthesize, args=(chunk,))
        thread.start()
        active_threads.append(thread)

    # Wait for all remaining threads to finish
    for thread in active_threads:
        thread.join()

def run_process_in_thread(chunks, max_threads):
    # Run the chunk processing function in a separate thread
    process_thread = threading.Thread(target=process_chunks_with_limit, args=(chunks, max_threads))
    process_thread.start()
    return process_thread  # Return the thread if you want to join it later

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
        if "messages" in request.form:
            messages_str = request.form['messages']
            if messages_str:
                messages = json.loads(messages_str)  # Convert JSON string back to array
                message = messages[-1]['content']

                print("LAST MESSAGE",message)

                pipe_out = pipeline.process(message, middlewares.contextualize(messages))
                
                chunks = chunk_string(pipe_out)
                
                threading.Thread(target=process_chunks_with_limit, args=(chunks,)).start()
            
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
