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
import face_recognition
import base64
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import re
from dotenv import load_dotenv
import urllib.parse
import urllib.request
import subprocess

load_dotenv()

app = Flask(__name__)

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

# Voice State
active_listening = False
start_closed_time = None
last_announced_state = "NORMAL"

# Face Tracking Logic
EAR_THRESHOLD = 0.22

def eye_aspect_ratio(eye_points):
    p1, p2, p3, p4, p5, p6 = [np.array(pt) for pt in eye_points]
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    if h == 0: return 0.0
    return (v1 + v2) / (2.0 * h)


# Flask Routes

@app.route('/')
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
        cmd = ["yt-dlp", "-J", "-f", "bestaudio", f"https://www.youtube.com/watch?v={video_id}"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        return jsonify({
            "url": info.get("url"),
            "title": info.get("title"),
            "artist": info.get("uploader"),
            "thumbnail": info.get("thumbnail")
        })
    except Exception as e:
        print(f"[yt-dlp error] {repr(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/frame', methods=['POST'])
def api_frame():
    global start_closed_time, last_announced_state
    
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image provided"}), 400
        
    # Decode base64 image
    img_data = base64.b64decode(data['image'].split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    scale = 0.5
    small_rgb = cv2.resize(rgb_frame, (0, 0), fx=scale, fy=scale)
    
    face_landmarks_list = face_recognition.face_landmarks(small_rgb)
    avg_ear = 1.0
    alert_triggered = None
    
    if face_landmarks_list:
        landmarks = face_landmarks_list[0]
        left_eye_raw = landmarks.get('left_eye', [])
        right_eye_raw = landmarks.get('right_eye', [])
        
        if len(left_eye_raw) == 6 and len(right_eye_raw) == 6:
            left_eye = [(int(x/scale), int(y/scale)) for x, y in left_eye_raw]
            right_eye = [(int(x/scale), int(y/scale)) for x, y in right_eye_raw]
            
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            
            system_status["ear"] = avg_ear
            
            if avg_ear < EAR_THRESHOLD:
                if start_closed_time is None:
                    start_closed_time = time.time()
                else:
                    duration = time.time() - start_closed_time
                    
                    # 3 s — first wake-up nudge
                    if duration >= 3.0 and last_announced_state == "NORMAL":
                        system_status["state"] = "LEVEL_1"
                        alert_triggered = "Wake up! Please stay focused on the road."
                        last_announced_state = "LEVEL_1"
                    
                    # 7 s — serious warning
                    elif duration >= 7.0 and last_announced_state == "LEVEL_1":
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
        "action": action
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
                query = urllib.parse.quote_plus(stripped + ' energetic')
                html = urllib.request.urlopen(f'https://www.youtube.com/results?search_query={query}').read().decode()
                video_ids = re.findall(r'watch\?v=(\S{11})', html)
                if video_ids:
                    return jsonify({"speak": f"Playing {stripped} to keep you alert.", "action": "play_native", "video_ids": video_ids[:20], "title": stripped})
                else:
                    return jsonify({"speak": "I couldn't find that song."})
            except Exception as e:
                print(f"[YouTube Search Error] {repr(e)}")
                return jsonify({"speak": "I had trouble searching for the song on YouTube Music. It might be blocking automated requests."})
        return jsonify({})

    # Normal wake-word logic
    command = ""
    if active_listening:
        command = re.sub(r'[^\w\s]', '', text).strip()
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

    if command.lower().startswith('play '):
        song = command[5:].strip()
        if song.lower() in ['music', 'song', 'the music', 'the song', 'it']:
            return jsonify({"speak": "Resuming.", "action": "resume_native"})
        if song:
            try:
                query = urllib.parse.quote_plus(song)
                html = urllib.request.urlopen(f'https://www.youtube.com/results?search_query={query}').read().decode()
                video_ids = re.findall(r'watch\?v=(\S{11})', html)
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