from io import BytesIO
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
from avatar.speech import synthesize_from_audio, process_avatar

app = Flask("Avatar RAG Backend")
app_url = "http://localhost:5000/"
CORS(app, origins=['*'])

init_socketio(app)

global_config={
"max_text_character_chunk_size":200,
"max_text_to_speech_avatar_thread":1,
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
            print(avatarId)
            video_path = f"idle_avatars/{avatarId}.mp4"
            print(video_path)
            return send_file(video_path,mimetype='video/mp4')


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

                    voiceId=request.form["voiceId"]
                    avatarId=request.form["avatarId"]
                    
                    threading.Thread(target=process_chunks_with_limit, args=(chunks,global_config['max_text_to_speech_avatar_thread'],avatarId,voiceId,)).start()
            
                return jsonify({'status': "success",
                                'text_response': pipe_out})

    except Exception as e:
        print(e)


@app.route('/synthesize_single_from_audio', methods=["POST", "OPTIONS"])
@cross_origin(supports_credentials=True)
def synthesize_from_audio_route():
    try:

       avatarId=None 
       if "avatarId" in request.form:
          avatarId=request.form["avatarId"]   
     
       if "audio" in request.files:
           audio=request.files["audio"]

           result=synthesize_from_audio(audio,avatarId)
           video_object = BytesIO(result)
           return send_file(video_object,mimetype='video/mp4') 

    except Exception as e:
        print(e)


@app.route('/process_avatar', methods=["POST", "OPTIONS"])
@cross_origin(supports_credentials=True)
def process_avatar_route():
    try:

       if "avatarVideo" in request.files:
           avatarVideo=request.files["avatarVideo"]
           create_perfect_loop=request.form["create_perfect_loop"]

           result=process_avatar(avatarVideo,create_perfect_loop)
           print(result)
           return jsonify(result)

    except Exception as e:
        print(e)





    except Exception as e:
        print(e)


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
