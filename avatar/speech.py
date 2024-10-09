from io import BytesIO
import pickle
import re
import time
from flask import Flask, request, jsonify, send_file
import json
# start a web server

from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, avatar.audio as audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, avatar.face_detection
from avatar.models import Wav2Lip
import platform
import gc
from dotenv import load_dotenv
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
from werkzeug.utils import secure_filename
from socketio_setup import socketio

#app = Flask(__name__)

load_dotenv()

class Args:
    def __init__(self):
        self.checkpoint_path = "avatar/checkpoints/wav2lip_gan.pth"
        self.face = "avatar_videos/lisa_casual_1080.mp4"
        self.audio = "avatar/input/default_audio.wav"
        self.outfile = 'avatar/results/result_voice.mp4'
        self.static = False
        self.fps = 25.
        self.pads = [0, 10, 0, 0]
        self.face_det_batch_size = 1
        self.wav2lip_batch_size = 200
        self.resize_factor = 1
        self.crop = [0, -1, 0, -1]
        self.box = [-1, -1, -1, -1]
        self.rotate = False
        self.nosmooth = False
        self.img_size = 96
        self.default_avatar_id="lisa_casual_1080"

args = Args()

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))


def _load(checkpoint_path):
    global model
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    global model
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

def load_model_and_dataset(voice_id=None):
    global text_to_speech_model
    global text_to_speech_embeddings_dataset

    if not text_to_speech_model:
        text_to_speech_model = pipeline("text-to-speech", "microsoft/speecht5_tts",device=device)
    
    if not text_to_speech_embeddings_dataset:
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    
    if voice_id:
        return torch.tensor(embeddings_dataset[int(voice_id)]["xvector"]).unsqueeze(0)
    else:
        return torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    # Return speaker embedding for later use (can be modified to load other embeddings if needed)


model_path ="avatar/checkpoints/wav2lip_gan.pth"


model = load_model(model_path)

# Global value for the text-to-speech model
text_to_speech_model = None
text_to_speech_embeddings_dataset=None

load_model_and_dataset()

face_det_results=None

full_frames=None


if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    args.static = True

def get_smoothened_boxes(boxes, T):
    global model
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

# CAN BE SKIPPED BECAUSE YOU KNOW THE VIDEO BEFOREHAND
def face_detect(images):
    global model


    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)

    batch_size = args.face_det_batch_size

    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))

        
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    fnr = 0
    x1 = 0
    y1 = 0
    x2 = 100
    y2 = 100
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
            print(f'Face not detected in {fnr}! Ensure the video contains a face in all the frames.')
        # raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
        else:
            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])
        fnr = fnr + 1

    boxes = np.array(results)
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
    
    del detector

    return results


def datagen(frames, mels):
    global model
    global face_det_results

    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    # CAN BE SKIPPED BECAUSE WE KNOW THE VIDEO BEFOREHAND
    if not face_det_results:
        if args.box[0] == -1:
            if not args.static:
                face_det_results = face_detect(frames)  # BGR2RGB for CNN face detection
            else:
                face_det_results = face_detect([frames[0]])
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = args.box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    # CANNOT BE SKIPPED BECAUSE IT CONTAINS THE AUDIO CHUNKS

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        # CAN BE SKIPPED BUT NOT LIKELY IMPORTANT ENOUGH OR AFFECTING THE PROCESS
        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch
 

def main():
    global model
    global full_frames
    
    start_time = time.time()

    
    # (NOT RELEVANT FOR US) IF THE FACE AGRUMENT WAS NOT PROVIDED   
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    # (NOT RELEVANT FOR US AS WE WILL SEND A VIDEO ALWAYS) IF THE FACE AGRUMENT WAS PROVIDED AS AN IMAGE
    else:
        if not full_frames:
            if args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
                full_frames = [cv2.imread(args.face)]

            # (TO SKIP) EXTRACTING ALL THE FRAMES OF THE VIDEO        
            else:
                print("Not full frames")
                full_frames=extract_frames_from_video(args.face)

    print("Number of frames available for inference: " + str(len(full_frames)))

    # (NOT RELEVANT FOR US AS WE WILL ALWAYS SEND WAV AUDIO) EXTRACTING RAW AUDIO FROM AUDIO ARGUMENT IF NOT PROVIDEDI IN WAV FORMAT
    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

        subprocess.call(command, shell=True)
        args.audio = 'temp/temp.wav'

    # (CANNOT BE SKIPPED RELATED TO AUDIO) Loading the Audio + Mel spectrogram converts the audio waveform into a Mel spectrogram, highlighting frequencies important to human hearing.

    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80. / args.fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    # (CANNOT BE SKIPPED) CHOOSING THE NUMBER OF FRAMES ACCORDING TO THE NUMBER OF AUDIO CHUNKS
    full_frames = full_frames[:len(mel_chunks)]

    batch_size = args.wav2lip_batch_size
 
    # (CANNOT BE SKIPPED) GENERATION FUNCTION BECAUSE IT CONTAINS THE AUDIO CHUNKCS
    gen = datagen(full_frames.copy(), mel_chunks)

    clear_memory()

  
    # processes batches of frames and audio to generate a video,
    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                    total=int(
                                                                        np.ceil(float(len(mel_chunks)) / batch_size)))):



        if i == 0:

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.avi',
                                  cv2.VideoWriter_fourcc(*'DIVX'), args.fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)



        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.


        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)


    out.release()

    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')
    end_time = time.time()
    duration = end_time - start_time
    print(f"Duration: {duration} seconds")


def get_avatar_face_detection_results(avatar_id):
    try:
        avatar_face_det_path=f"avatar/face_detection_results/{avatar_id}.pkl"
        print(avatar_face_det_path)
        with open(avatar_face_det_path, 'rb') as f:
            face_det_results=pickle.load(f)
            return face_det_results

    except Exception as e:
        print(e)

def get_avatar_video(avatar_id):
    try:
        avatar_video_path=f"avatar/avatar_videos/{avatar_id}.mp4"
        return avatar_video_path

    except Exception as e:
        print(e)

def generate_avatar_face_detection(avatar_video):
    global full_frames
    global face_det_results

    try:
            avatar_video.seek(0)

            # Secure the filename
            filename = secure_filename(avatar_video.filename)

            # Extract the file name without the extension
            name_without_ext = os.path.splitext(filename)[0]

            # Specify the path where you want to save the file
            save_path = os.path.join('avatar_videos', filename)
            
            avatar_video.save(save_path)
            args.face=save_path
            print(save_path)

            full_frames= extract_frames_from_video(save_path)
            
            frames=full_frames.copy()

            avatar_videos_frames_path = f"avatar_videos_frames/{name_without_ext}.pkl"
            
            # Write the face detection results to the pickle file
            with open(avatar_videos_frames_path, 'wb') as f:
                pickle.dump(frames, f)

            if args.box[0] == -1:
                if not args.static:
                    face_det_results = face_detect(frames)  # BGR2RGB for CNN face detection
                else:
                    face_det_results = face_detect([frames[0]])
            else:
                print('Using the specified bounding box instead of face detection...')
                y1, y2, x1, x2 = args.box
                face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
            
            avatar_face_det_path = f"face_detection_results/{name_without_ext}.pkl"
            
            # Write the face detection results to the pickle file
            with open(avatar_face_det_path, 'wb') as f:
                pickle.dump(face_det_results, f)


    except Exception as e:
        print(e)
        return jsonify({"status":"failure"})



def generate_speech(text, voice_id=None):
    global text_to_speech_model
    
    speaker_embeddings=load_model_and_dataset(voice_id)
    
    # Generate speech
    speech = text_to_speech_model(text, forward_params={"speaker_embeddings": speaker_embeddings})
    
    # Write audio to a bytes buffer (in-memory)
    buffer = BytesIO()
    sf.write(buffer, speech["audio"], samplerate=speech["sampling_rate"], format="WAV")
    
    # Get the audio blob from the buffer
    buffer.seek(0)
    return buffer

def get_avatar_videos_frames(avatar_id):
    avatar_video_frames_path=f"avatar/avatar_videos_frames/{avatar_id}.pkl"
    with open(avatar_video_frames_path, 'rb') as f:
        full_frames=pickle.load(f)
        return full_frames


'''
@app.route('/save_avatar', methods=['POST'])
def save_avatar():

    try:
        if "avatar_video" in request.files:
            avatar_video=request.files["avatar_video"]

        if avatar_video: 
            result= generate_avatar_face_detection(avatar_video)
            return jsonify({"status":"success","message":"Avatar Created Successfully"})

    except Exception as e:
        print(e)
        return jsonify({"status":"failure"})'''


def extract_frames_from_video(avatar_video_path):

    video_stream = cv2.VideoCapture(avatar_video_path)
    args.fps = video_stream.get(cv2.CAP_PROP_FPS)

    print('Reading video frames...')

    full_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        if args.resize_factor > 1:
            frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))

        if args.rotate:
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

        y1, y2, x1, x2 = args.crop
        if x2 == -1: x2 = frame.shape[1]
        if y2 == -1: y2 = frame.shape[0]

        frame = frame[y1:y2, x1:x2]

        full_frames.append(frame)
    return full_frames


def clear_memory():
    # Clear CPU memory
    torch.cuda.empty_cache()  # Clear GPU memory (if using GPU)
    gc.collect() 


def synthesize(text,avatar_id=None,voice_id=None):

    try:
        global face_det_results
        global full_frames

        if not avatar_id:
            avatar_id=args.default_avatar_id
        
        if avatar_id:
            avatar_video_path=get_avatar_video(avatar_id)
            args.face=avatar_video_path
            full_frames= get_avatar_videos_frames(avatar_id)
            face_det_results=get_avatar_face_detection_results(avatar_id)

        speech_blob=generate_speech(text,voice_id)
        audio_path=f"avatar/input/audio.wav"
        
        with open(audio_path, "wb") as f:
            f.write(speech_blob.read())
        args.audio = audio_path

        main()

        with open(args.outfile,"rb") as f:
            video_bytes = f.read()
            socketio.emit('video_chunk', video_bytes)
        
        return text
    
    except Exception as e:
        print(e)
'''
if __name__ == '__main__':
    main_load_model()

    app.run(host='0.0.0.0', port=38888,debug=False)'''