1) Step 1: install pickle --> pip install pickle

2) Step 2: run server --> python server.py

3) Step 3: Make an an API Call to process your first video:

Use one of the following option

1) CURL: curl --location 'http://127.0.0.1:38888/save_avatar' \
--form 'avatar_video=@"/path/to/file"'

2) Python: import requests

url = "http://127.0.0.1:38888/save_avatar"

payload = {}
files=[
  ('avatar_video',('file',open('/path/to/file','rb'),'application/octet-stream'))
]
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)


4) Step 4: Make sure face_detection_results and avatar_videos and avatar_videos_frames have .pkl information

5) Step 5: Make an api call to synthesize

a) avatar_id is just the video name without extension

b) If you include the audio it will use it for speech generation, if you don't specify audio you must specify text to generate speech, optionally you can specify an voice_id value between 1 and 7509 but 7306 the default is the best


avatar_id is 

Using curl: 

curl --location 'http://127.0.0.1:38888/synthesize' \
--form 'avatar_id=@"/path/to/file"' \
--form 'audio=@"/C:/Mohamad_Hammoud/Work/Airport/Audio Creation/output_audio.wav"' \
--form 'voice_id="7306"' \
--form 'text="Hello this is a test so please read it in now"'

Using python:

import requests

url = "http://127.0.0.1:38888/synthesize"

payload = {'voice_id': '7306',
'text': 'Hello this is a test so please read it in now'}
files=[
  ('avatar_id',('file',open('/path/to/file','rb'),'application/octet-stream')),
  ('audio',('output_audio.wav',open('/C:/Mohamad_Hammoud/Work/Airport/Audio Creation/output_audio.wav','rb'),'audio/wav'))
]
headers = {}

response = requests.request("POST", url, headers=headers, data=payload, files=files)

print(response.text)

