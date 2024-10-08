
import json
from flask import Flask, jsonify, send_file,request
from flask_cors import CORS, cross_origin

app = Flask("Avatar RAG Backend")
app_url="http://localhost:5000/"
CORS(app, origins=['*'])


# GET IDLE AVATAR FOR LOCAL VIDEO 

@app.route('/getIdleAvatar', methods=["POST","OPTIONS"])
@cross_origin(supports_credentials=True)
def get_idle_avatar_route():
    try:
        if "avatarId" in request.form:
            
            avatarId=request.form["avatarId"]
            avatarProvider=request.form["avatarProvider"]

            # Simulate Text to speech avatar pipeline or so 

            video_path="default_avatar.mp4"

            return jsonify({
            'status':"success",
            'video_url':f'{app_url}video/'+video_path,
            })      


    except Exception as e:
        print(e)


@app.route('/handleUserQuery', methods=["POST","OPTIONS"])
@cross_origin(supports_credentials=True)
def handle_user_query_route():
    try:

        if "messages" in request.form:
            messages_str = request.form['messages']
            messages=None
            if messages_str:
                messages = json.loads(messages_str)  # Convert JSON string back to array
                print(messages)

        # Simulate Search Call (you have to embed the user query in your search pipeline and then perform RAG)

        # Simulate LLM CALL After receiving Search scall

        # Simulate Avatar (Text to speech is already included in the new pipeline)

        # Return video

        video_path="result_video.mp4"
        return jsonify({'status':"success",'video_url':f'{app_url}video/'+video_path,'text_response':"Hello! How can I assist you today? Feel free to ask me anything, whether it's about general knowledge, recommendations, or troubleshooting. I'm here to help!"})


    except Exception as e:
        print(e)

@app.route('/video/<filename>', methods=['GET'])
def serve_video(filename):
    # Serve video file from local storage
    video_path = f'temp/{filename}'
    return send_file(video_path, mimetype='video/mp4')



if __name__ == '__main__':
    app.run()