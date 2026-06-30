import os
import sys
import importlib.util

class DummyPkgResources:
    @staticmethod
    def resource_filename(module_name, resource_name):
        spec = importlib.util.find_spec(module_name)
        if spec and spec.origin:
            return os.path.join(os.path.dirname(spec.origin), resource_name)
        return ""

# Monkey-patch pkg_resources to fix the "No module named 'pkg_resources'" error 
# caused by Hugging Face's stripped-down Python environments
if 'pkg_resources' not in sys.modules:
    sys.modules['pkg_resources'] = DummyPkgResources

import cv2
import time
import json
import numpy as np
import base64
import urllib.request
import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pymongo import MongoClient
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from openai import OpenAI
import re
from datetime import timedelta
from dotenv import load_dotenv
import urllib.parse
from ytmusicapi import YTMusic
from curl_cffi import requests as cffi_requests
import requests
import socket
from pytubefix import YouTube
import urllib3.util.connection as urllib3_cn

def allowed_gai_family():
    return socket.AF_INET
urllib3_cn.allowed_gai_family = allowed_gai_family

from werkzeug.middleware.proxy_fix import ProxyFix

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Fix for Hugging Face Spaces iframe and HTTPS
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_SAMESITE='None',
    PERMANENT_SESSION_LIFETIME=timedelta(days=30)
)

# MongoDB Connection
def get_db():
    client = MongoClient(os.getenv("MONGO_URI"))
    return client.safedriving

# Login Required Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# MediaPipe Setup (New Tasks API, dynamically downloading the model if missing)
MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe Face Landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5)

face_mesh = vision.FaceLandmarker.create_from_options(options)
# MediaPipe Eye Landmark Indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144] # User's right eye
LEFT_EYE = [362, 385, 387, 263, 373, 380] # User's left eye

# Global State
system_status = {
    "state": "NORMAL",
    "ear": 1.0,
    "drowsy_counter": 0
}

_dialogue_state = 'none'
_dialogue_start_time = 0

def set_dialogue(state):
    global _dialogue_state, _dialogue_start_time
    _dialogue_state = state
    _dialogue_start_time = time.time()

def get_dialogue():
    return _dialogue_state

# Voice State & Tracking Variables
active_listening = False
start_closed_time = None
last_announced_state = "NORMAL"

# Adjusted EAR Threshold for MediaPipe (MediaPipe EAR tends to be slightly different than dlib)
EAR_THRESHOLD = 0.20 
smoothed_ear = 1.0  # For Exponential Moving Average
EMA_ALPHA = 0.4     # Smoothing factor (Lower = smoother but slightly delayed)

def eye_aspect_ratio_mediapipe(landmarks, eye_indices, img_w, img_h):
    # Convert normalized landmarks to pixel coordinates
    pts = [np.array([landmarks[i].x * img_w, landmarks[i].y * img_h]) for i in eye_indices]
    
    # Compute EAR
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    
    if h == 0: return 0.0
    return (v1 + v2) / (2.0 * h), pts

def enhance_low_light(frame):
    """ Applies CLAHE to improve contrast in dark environments """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# FLASK ROUTES
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form.get('confirm_password', '')
        
        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return render_template('signup.html')
            
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('signup.html')
        
        db = get_db()
        users = db.users
        
        if users.find_one({'username': username}):
            flash('Username already exists.', 'danger')
        else:
            users.insert_one({
                'username': username,
                'password': generate_password_hash(password)
            })
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
            
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        db = get_db()
        users = db.users
        user = users.find_one({'username': username})
        
        if user and check_password_hash(user['password'], password):
            session.permanent = True
            session['user_id'] = str(user['_id'])
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')
            
    return render_template('login.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        username = request.form['username']
        new_password = request.form['new_password']
        
        if len(new_password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return render_template('reset_password.html')
            
        db = get_db()
        users = db.users
        
        user = users.find_one({'username': username})
        if user:
            users.update_one({'username': username}, {'$set': {'password': generate_password_hash(new_password)}})
            flash('Password reset successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Username not found.', 'danger')
            
    return render_template('reset_password.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    return jsonify(system_status)

@app.route('/api/music_url', methods=['GET'])
def api_music_url():
    video_id = request.args.get('video_id')
    if not video_id:
        return jsonify({"error": "No video_id provided"}), 400
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        yt = YouTube(url, use_oauth=False, allow_oauth_cache=False)
        audio_stream = yt.streams.get_audio_only()
        
        return jsonify({
            "url": audio_stream.url,
            "title": yt.title,
            "artist": yt.author,
            "thumbnail": yt.thumbnail_url
        })
    except Exception as e:
        print(f"[pytubefix error] {repr(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/frame', methods=['POST'])
def api_frame():
    global start_closed_time, last_announced_state, smoothed_ear
    
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image provided"}), 400
        
    # Decode base64 image
    img_data = base64.b64decode(data['image'].split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # 1. Low Light Enhancement (CLAHE)
    frame = enhance_low_light(frame)
    
    # 2. Process with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = frame.shape
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = face_mesh.detect(mp_image)
    
    avg_ear = smoothed_ear
    alert_triggered = None
    left_eye_points_out = []
    right_eye_points_out = []
    
    if results.face_landmarks:
        landmarks = results.face_landmarks[0]
        
        # Calculate EAR
        right_ear, right_eye_pts = eye_aspect_ratio_mediapipe(landmarks, RIGHT_EYE, img_w, img_h)
        left_ear, left_eye_pts = eye_aspect_ratio_mediapipe(landmarks, LEFT_EYE, img_w, img_h)
        
        raw_avg_ear = (left_ear + right_ear) / 2.0
        
        # 3. Signal Smoothing (EMA) to combat low camera resolution jitters
        smoothed_ear = (EMA_ALPHA * raw_avg_ear) + ((1 - EMA_ALPHA) * smoothed_ear)
        avg_ear = smoothed_ear
        system_status["ear"] = avg_ear
        
        # Formatting for frontend drawing (if needed)
        left_eye_points_out = [(int(pt[0]), int(pt[1])) for pt in left_eye_pts]
        right_eye_points_out = [(int(pt[0]), int(pt[1])) for pt in right_eye_pts]
        
        # Alert Logic
        if avg_ear < EAR_THRESHOLD:
            if start_closed_time is None:
                start_closed_time = time.time()
            else:
                duration = time.time() - start_closed_time
                
                # 5s — first wake-up nudge
                if duration >= 5.0 and last_announced_state == "NORMAL":
                    system_status["state"] = "LEVEL_1"
                    alert_triggered = "Wake up! Please stay focused on the road."
                    last_announced_state = "LEVEL_1"
                
                # 8s — serious warning
                elif duration >= 8.0 and last_announced_state == "LEVEL_1":
                    system_status["state"] = "LEVEL_3"
                    system_status["drowsy_counter"] += 1
                    last_announced_state = "LEVEL_3"
                    
                    if system_status["drowsy_counter"] >= 3 and get_dialogue() == 'none':
                        set_dialogue('asking_rest')
                        alert_triggered = "You have been drowsy multiple times. I strongly recommend you take a rest or have an energy drink. Should I remind you to pull over? Say yes or no."
                    else:
                        alert_triggered = "Please pull over safely and take a rest."
        else:
            start_closed_time = None
            if system_status["state"] != "NORMAL":
                system_status["state"] = "NORMAL"
                last_announced_state = "NORMAL"
    else:
        # No face detected
        start_closed_time = None
        if system_status["state"] != "NORMAL":
            system_status["state"] = "NORMAL"
            last_announced_state = "NORMAL"

    # Contact emergency if no dialogue response
    action = None
    if _dialogue_state in ['asking_rest', 'asking_song']:
        if time.time() - _dialogue_start_time > 15.0:
            set_dialogue('none')
            alert_triggered = "No response detected. Contacting emergency services on 911."
            system_status["state"] = "CRITICAL"
            action = "call_emergency"

    return jsonify({
        "ear": avg_ear,
        "state": system_status["state"],
        "drowsy_counter": system_status["drowsy_counter"],
        "alert": alert_triggered,
        "action": action,
        "left_eye": left_eye_points_out,
        "right_eye": right_eye_points_out
    })

@app.route('/api/voice', methods=['POST'])
def api_voice():
    global active_listening
    data = request.json
    text = data.get('text', '').lower().strip()
    
    if not text:
        return jsonify({})
        
    print(f'[Voice API] Received: "{text}"')
    
    dialogue = get_dialogue()

    # Dialogue logic
    if dialogue == 'asking_rest':
        if any(w in text for w in ['yes', 'yeah', 'sure', 'okay', 'ok', 'please', 'fine']):
            set_dialogue('none')
            return jsonify({"speak": "Please find a safe place to pull over and take a break. Your safety matters."})
        elif any(w in text for w in ['no', 'nope', 'nah', 'not', "don't", 'refuse']):
            set_dialogue('asking_song')
            return jsonify({"speak": "Alright. Would you like me to play an energetic song to keep you awake? Just tell me the song or artist name."})
        return jsonify({})

    if dialogue == 'asking_song':
        stripped = re.sub(r'[^\w\s]', '', text).strip()
        if len(stripped) > 1:
            set_dialogue('none')
            try:
                ytmusic = YTMusic()
                    
                results = ytmusic.search(f"{stripped} energetic lyrics", filter="videos", limit=10)
                video_ids = [res['videoId'] for res in results if 'videoId' in res]
                if video_ids:
                    return jsonify({"speak": f"Playing {stripped} to keep you alert.", "action": "play_native", "video_ids": video_ids[:20], "title": stripped})
                else:
                    return jsonify({"speak": "I couldn't find that song."})
            except Exception as e:
                print(f"[YouTube Search Error] {repr(e)}")
                # Fallback to yt-dlp search if ytmusicapi is blocked
                try:
                    cmd = ["yt-dlp", "--impersonate", "chrome", "--force-ipv4", "--get-id", "--flat-playlist"]
                    cmd.append(f"ytsearch10:{stripped} energetic")
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
                    ids = result.stdout.strip().split('\n')
                    video_ids = [vid.strip() for vid in ids if len(vid.strip()) == 11]
                    if video_ids:
                         return jsonify({"speak": f"Playing {stripped} to keep you alert.", "action": "play_native", "video_ids": video_ids[:20], "title": stripped})
                except Exception as ex:
                    print(f"[Fallback Search Error] {repr(ex)}")
                    
                return jsonify({"speak": "I had trouble searching for the song on YouTube Music. It might be blocking automated requests."})
        return jsonify({})

    # Normal wake-word logic
    clean_text = re.sub(r'[^\w\s]', '', text).strip()
    if 'lara thanks' in clean_text or 'lara thank you' in clean_text:
        active_listening = False
        return jsonify({"speak": "You're welcome."})
    elif active_listening and clean_text in ['thanks', 'thank you']:
        active_listening = False
        return jsonify({"speak": "You're welcome."})
        
    command = ""
    if active_listening:
        command = clean_text
        if command in ['okay bye', 'ok bye', 'bye', 'goodbye', 'stop listening']:
            active_listening = False
            return jsonify({"speak": "Goodbye."})
    else:
        if 'lara' not in text:
            return jsonify({})
        idx = text.find('lara') + len('lara')
        command = re.sub(r'[^\w\s]', '', text[idx:]).strip()

    if len(command) < 3:
        active_listening = True
        return jsonify({"speak": "Yes?"})

    print(f'[Voice API] Command: "{command}"')

    # Media Commands (Native Player Control)
    cmd = command.lower()
    if 'pause' in cmd:
        return jsonify({"speak": "Pausing.", "action": "pause_native"})
    if 'resume' in cmd or cmd == 'play':
        return jsonify({"speak": "Resuming.", "action": "resume_native"})
    if 'stop' in cmd:
        return jsonify({"speak": "Stopping.", "action": "stop_native"})
    if 'next' in cmd or 'skip' in cmd:
        return jsonify({"speak": "Skipping.", "action": "next_native"})
    if 'previous' in cmd or 'back' in cmd:
        return jsonify({"speak": "Going back.", "action": "prev_native"})
    if 'volume up' in cmd or 'louder' in cmd or 'increase volume' in cmd:
        return jsonify({"speak": "Volume up.", "action": "vol_up_native"})
    if 'volume down' in cmd or 'quieter' in cmd or 'decrease volume' in cmd:
        return jsonify({"speak": "Volume down.", "action": "vol_down_native"})
    if 'unmute' in cmd:
        return jsonify({"speak": "Unmuting.", "action": "unmute_native"})
    elif 'mute' in cmd:
        return jsonify({"speak": "Muting.", "action": "mute_native"})

    if 'play ' in command.lower():
        idx = command.lower().find('play ') + 5
        song = command[idx:].strip()
        if song.lower() in ['music', 'song', 'the music', 'the song', 'it']:
            return jsonify({"speak": "Resuming.", "action": "resume_native"})
        if song.lower() in ['another', 'another one', 'another song', 'something else', 'a different song', 'different song', 'this song', 'this']:
            return jsonify({"speak": "Skipping.", "action": "next_native"})
        if song:
            try:
                # Inject a Chrome-impersonated session to bypass YouTube bot blocking on Hugging Face Spaces
                custom_session = cffi_requests.Session(impersonate="chrome")
                ytmusic = YTMusic(requests_session=custom_session)
                
                # Generic search
                results = ytmusic.search(f"{song} lyrics", filter="videos", limit=10)
                video_ids = [res['videoId'] for res in results if 'videoId' in res]
                if video_ids:
                    return jsonify({"speak": f"Playing {song}.", "action": "play_native", "video_ids": video_ids[:20], "title": song})
                else:
                    return jsonify({"speak": "I couldn't find that song."})
            except Exception as e:
                print(f"[YouTube Search Error] {repr(e)}")
                return jsonify({"speak": "I had trouble searching for the song on YouTube Music. It might be blocking automated requests."})

    # LLM Interaction
    api_key = os.getenv('API_KEY', '')
    if not api_key:
        return jsonify({"speak": "No API key is configured.", "text": "**Offline Mode** — No API key configured."})
        
    try:
        llm = OpenAI(api_key=api_key, base_url='https://integrate.api.nvidia.com/v1')
        SYS_PROMPT = (
            'You are Lara, a voice assistant and copilot for a driver. '
            'You MUST ONLY assist with driving, road-related questions, directions, routes, and playing music. '
            'If the user asks about ANYTHING else, politely refuse to answer and state that you can only help with driving and music. '
            'Provide helpful, concise, and clear answers so the driver can stay focused on the road. '
            'Keep your responses brief (1-2 sentences max) and do not use conversational filler.'
        )
        resp = llm.chat.completions.create(
            model='meta/llama-3.1-8b-instruct',
            max_tokens=120,
            messages=[
                {'role': 'system', 'content': SYS_PROMPT},
                {'role': 'user', 'content': command},
            ]
        )
        reply = resp.choices[0].message.content.strip()
        print(f'[Voice API] Reply: "{reply}"')
        return jsonify({"speak": reply, "text": reply})
    except Exception as exc:
        print(f'[Voice API] LLM error: {exc}')
        return jsonify({
            "speak": "Sorry, I had a network error.", 
            "error": str(exc),
            "text": f"**LLM API Error:** {str(exc)}"
        })


@app.route('/api/chat', methods=['POST'])
def api_chat():
    # Keep standard text-chat endpoint working for the UI form
    data = request.json
    prompt = data.get('prompt', '')
    api_key = os.getenv('API_KEY', '')
    
    if not api_key:
        return jsonify({"response": "**Offline Mode** — No API key configured."})
        
    try:
        client = OpenAI(api_key=api_key, base_url="https://integrate.api.nvidia.com/v1")
        sys_prompt = (
            'You are Lara, a voice assistant and copilot for a driver. '
            'You MUST ONLY assist with driving, road-related questions, directions, routes, and playing music. '
            'If the user asks about ANYTHING else, politely refuse to answer and state that you can only help with driving and music. '
            'Provide helpful, concise, and clear answers so the driver can stay focused on the road. '
            'Keep your responses brief (1-2 sentences max) and do not use conversational filler.'
        )
        completion = client.chat.completions.create(
            model='meta/llama-3.1-8b-instruct',
            max_tokens=300,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        response_text = completion.choices[0].message.content
        return jsonify({
            "response": response_text,
            "speak": re.sub(r'\*+', '', response_text).strip()
        })
    except Exception as exc:
        print(f'[Chat API] LLM error: {exc}')
        return jsonify({"response": f"**LLM API Error:** {str(exc)}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Threaded=True handles multiple browser connections safely
    app.run(debug=False, host='0.0.0.0', port=7860, threaded=True, use_reloader=False)