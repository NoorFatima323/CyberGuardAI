# ---------------------- streamlit_app.py ----------------------
import os, io, json, tempfile, re, requests, shutil
import numpy as np
import librosa, soundfile as sf
from pydub import AudioSegment

import streamlit as st
import whisper

# ---------------------- Load Whisper ----------------------
WHISPER_MODEL_NAME = "small"
st.sidebar.info("Loading Whisper model (~30s)...")
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
st.sidebar.success("✅ Whisper loaded")

# ---------------------- Groq API ----------------------
GROQ_CHAT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_KEY = os.environ.get("GROQ_API_KEY", None)
st.sidebar.info(f"GROQ key available? {'YES' if GROQ_KEY else 'NO'}")

# ---------------------- Utility functions ----------------------
def save_numpy_audio_to_wav(audio_data, out_path):
    if audio_data is None:
        raise ValueError("No audio provided")
    y, sr = audio_data
    sf.write(out_path, y, sr)
    return out_path

def extract_audio_features(wav_path, sr=16000):
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    rms = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    pitch = 0.0
    try:
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
        f0_nonan = f0[~np.isnan(f0)]
        if len(f0_nonan) > 0:
            pitch = float(np.median(f0_nonan))
    except:
        pitch = 0.0
    tempo = 0.0
    try:
        tempo = float(librosa.beat.tempo(y=y, sr=sr, aggregate=None)[0])
    except:
        tempo = 0.0
    duration = float(librosa.get_duration(y=y, sr=sr))
    return {"rms": rms, "zcr": zcr, "pitch": pitch, "tempo": tempo, "duration": duration}

def heuristic_emotion_label(features):
    if not isinstance(features, dict):
        return "unknown"
    dur = features.get("duration", 0)
    if dur < 0.35:
        return "too_short"
    rms = features.get("rms", 0)
    pitch = features.get("pitch", 0)
    tempo = features.get("tempo", 0)
    if rms > 0.03 and pitch > 200:
        return "angry/urgent"
    if tempo > 140 or (rms > 0.025 and tempo > 110):
        return "pressured/fast"
    if rms < 0.007 and tempo < 80:
        return "calm/soft"
    return "neutral"

def build_groq_prompt(transcription, emotion, features):
    return f"""
You are an assistant that MUST return JSON only. Analyze the short phone call below.

Transcription: \"\"\"{transcription}\"\"\"
Estimated voice emotion: {emotion}
Audio features: {json.dumps(features)}

Task:
- Choose label from: "SPAM", "FRAUD", "LEGITIMATE", "UNKNOWN".
- Provide confidence (0-100).
- Provide 2-4 concise reasons (array).
- Suggest one short safe response user can say (<= 20 words).

Return JSON only with keys: label, confidence, reasons, safe_response.
"""

def call_groq_chat(api_key, prompt, model="mixtral-8x7b", max_tokens=300, temperature=0.0, timeout=30):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a JSON-output assistant and must return JSON only."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": 1
    }
    resp = requests.post(GROQ_CHAT_ENDPOINT, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def local_heuristic_classifier(transcription, emotion, features):
    txt = (transcription or "").lower()
    reasons = []
    score = 40
    fraud_keywords = ["bank","account","password","otp","verify","ssn","social security","transfer","wire","pin"]
    spam_keywords = ["prize","winner","congratulations","free","claim","limited time","act now","offer"]
    if any(k in txt for k in fraud_keywords):
        reasons.append("mentions account/verification keywords")
        score += 30
    if any(k in txt for k in spam_keywords):
        reasons.append("uses promotional/prize language")
        score += 20
    if "call from" in txt or "we are calling" in txt:
        reasons.append("unspecified caller identity")
        score += 10
    if emotion and ("angry" in emotion or "pressured" in emotion):
        reasons.append("urgent/pressured tone detected")
        score += 10
    if score >= 80:
        label = "FRAUD"
    elif score >= 60:
        label = "SPAM"
    elif score >= 45:
        label = "POTENTIAL_SPAM"
    else:
        label = "LEGITIMATE"
    confidence = int(min(95, score + 5))
    if not reasons:
        reasons = ["No clear spam/fraud indicators in this short sample."]
    return {"label": label, "confidence": confidence, "reasons": reasons, "safe_response": "I don't share personal info on calls; please send in writing."}

# ---------------------- Main pipeline ----------------------
def analyze_pipeline(audio_file, manual_text, groq_model_choice="mixtral-8x7b"):
    tmpdir = tempfile.mkdtemp()
    try:
        wav_out = os.path.join(tmpdir, "call.wav")
        if audio_file is None:
            return {"error": "No audio uploaded."}
        # Convert uploaded audio to WAV
        y, sr = librosa.load(audio_file, sr=16000, mono=True)
        save_numpy_audio_to_wav((y, sr), wav_out)
    except Exception as e:
        return {"error": f"Audio save failed: {e}"}

    try:
        t = whisper_model.transcribe(wav_out)
        transcription = t.get("text","").strip()
    except Exception as e:
        transcription = f"[transcription_error: {e}]"

    transcription_combined = transcription
    if manual_text and manual_text.strip():
        transcription_combined = (transcription + " " + manual_text.strip()).strip()

    try:
        features = extract_audio_features(wav_out)
    except Exception as e:
        features = {"error": str(e)}

    emotion = heuristic_emotion_label(features) if isinstance(features, dict) else "unknown"

    # Classification
    if GROQ_KEY:
        try:
            prompt = build_groq_prompt(transcription_combined or "[no transcription]", emotion, features)
            api_resp = call_groq_chat(GROQ_KEY, prompt, model=groq_model_choice)
            assistant_text = None
            if "choices" in api_resp and len(api_resp["choices"])>0:
                ch = api_resp["choices"][0]
                if "message" in ch and "content" in ch["message"]:
                    assistant_text = ch["message"]["content"]
                elif "text" in ch:
                    assistant_text = ch["text"]
            if assistant_text:
                m = re.search(r"\{.*\}", assistant_text, flags=re.DOTALL)
                if m:
                    try:
                        classification = json.loads(m.group(0))
                    except:
                        classification = local_heuristic_classifier(transcription_combined, emotion, features)
                else:
                    classification = local_heuristic_classifier(transcription_combined, emotion, features)
            else:
                classification = local_heuristic_classifier(transcription_combined, emotion, features)
        except:
            classification = local_heuristic_classifier(transcription_combined, emotion, features)
    else:
        classification = local_heuristic_classifier(transcription_combined, emotion, features)

    result = {
        "transcription": transcription,
        "transcription_combined": transcription_combined,
        "emotion": emotion,
        "features": features,
        "classification": classification
    }

    try:
        shutil.rmtree(tmpdir)
    except:
        pass

    return result

# ---------------------- Streamlit UI ----------------------
st.title("🚨 Live-call Spam & Fraud Detector")
st.write("Upload a short audio snippet (<= ~25s) and optionally paste call text.")

audio_file = st.file_uploader("🎤 Upload audio file (wav, mp3, m4a)", type=["wav","mp3","m4a"])
manual_text = st.text_area("Optional: paste call text", placeholder="e.g. 'Caller: OTP needed...'")
groq_model_choice = st.selectbox("Select Groq model (backend key used)", ["mixtral-8x7b","gpt-4o-mini"], index=0)

if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        result = analyze_pipeline(audio_file, manual_text, groq_model_choice)
        st.subheader("📊 Full JSON Result")
        st.json(result)
        st.text("Transcription: " + result.get("transcription",""))
        st.text("Emotion label: " + result.get("emotion",""))
        st.subheader("Audio Features")
        st.json(result.get("features",{}))
