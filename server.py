
from flask import Flask, jsonify, send_file,request
from flask_cors import CORS, cross_origin

app = Flask("Avatar RAG Backend")
app_url="http://localhost:5000/"
CORS(app, origins=['*'])

@app.route('/getIdleAvatar', methods=["POST","OPTIONS"])
@cross_origin(supports_credentials=True)
def get_idle_avatar_route():
    try:
        if "avatarId" in request.form:
            
            avatarId=request.form["avatarId"]

            # Simulate Text to speech avatar pipeline or so 

            video_path="idle_video.mp4"

            return jsonify({
            'status':"success",
            'video_url':f'{app_url}video/'+video_path
            })      


    except Exception as e:
        print(e)


@app.route('/handleUserQuery', methods=["POST","OPTIONS"])
@cross_origin(supports_credentials=True)
def handle_user_query_route():
    try:
            
        userTextInput=request.form["userTextInput"]

        # Simulate Search Call (you have to embed the user query in your search pipeline and then perform RAG)

        # Simulate LLM CALL After receiving Search scall

        # Simulate Avatar (Text to speech is already included in the new pipeline)

        # Return video

        video_path="result_video.mp4"
        print(video_path)
        return jsonify({'status':"success",'video_url':f'{app_url}video/'+video_path})

    except Exception as e:
        print(e)

@app.route('/video/<filename>', methods=['GET'])
def serve_video(filename):
    # Serve video file from local storage
    video_path = f'temp/{filename}'
    return send_file(video_path, mimetype='video/mp4')



if __name__ == '__main__':
    app.run()