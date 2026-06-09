#!/usr/bin/env python3
# Kalki AI - Multilingual Voice Assistant (pure sounddevice, no pyaudio)

import os
import sys
import io
import threading
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Core libraries
import speech_recognition as sr
import google.generativeai as genai
from gtts import gTTS
import pygame
import sounddevice as sd
import numpy as np

# ========== API KEY MANAGEMENT ==========
CONFIG_FILE = Path(__file__).parent / "kalki_key.txt"

def get_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            key = f.read().strip()
            if key:
                return key
    print("\n" + "="*50)
    print("🔑 Gemini API Key Required")
    print("Get your free key from: https://aistudio.google.com/apikey")
    print("="*50)
    key = input("Paste your API key and press Enter: ").strip()
    if key:
        with open(CONFIG_FILE, "w") as f:
            f.write(key)
        print("✅ Key saved.\n")
        return key
    else:
        print("❌ No key provided. Exiting.")
        sys.exit(1)

GEMINI_API_KEY = get_api_key()
genai.configure(api_key=GEMINI_API_KEY)

# ========== LANGUAGE DETECTION ==========
LANGUAGE_DETECTION = {
    'hi': {'keywords': ['नमस्ते', 'धन्यवाद', 'कैसे'], 'name': 'Hindi', 'gtts_code': 'hi'},
    'en': {'keywords': ['hello', 'thank', 'you', 'how'], 'name': 'English', 'gtts_code': 'en'},
    'mr': {'keywords': ['नमस्कार', 'धन्यवाद', 'कसे'], 'name': 'Marathi', 'gtts_code': 'mr'},
    'bn': {'keywords': ['নমস্কার', 'ধন্যবাদ', 'কেমন'], 'name': 'Bengali', 'gtts_code': 'bn'},
    'ta': {'keywords': ['வணக்கம்', 'நன்றி', 'எப்படி'], 'name': 'Tamil', 'gtts_code': 'ta'},
    'te': {'keywords': ['నమస్కారం', 'ధన్యవాదాలు', 'ఎలా'], 'name': 'Telugu', 'gtts_code': 'te'},
    'gu': {'keywords': ['નમસ્તે', 'આભાર', 'કેમ'], 'name': 'Gujarati', 'gtts_code': 'gu'},
    'pa': {'keywords': ['ਸਤ ਸ੍ਰੀ ਅਕਾਲ', 'ਧੰਨਵਾਦ', 'ਕਿਵੇਂ'], 'name': 'Punjabi', 'gtts_code': 'pa'},
    'ml': {'keywords': ['നമസ്കാരം', 'നന്ദി', 'എങ്ങനെ'], 'name': 'Malayalam', 'gtts_code': 'ml'},
    'kn': {'keywords': ['ನಮಸ್ಕಾರ', 'ಧನ್ಯವಾದ', 'ಹೇಗೆ'], 'name': 'Kannada', 'gtts_code': 'kn'},
    'or': {'keywords': ['ନମସ୍କାର', 'ଧନ୍ୟବାଦ', 'କେମିତି'], 'name': 'Odia', 'gtts_code': 'or'},
    'as': {'keywords': ['নমস্কাৰ', 'ধন্যবাদ', 'কেমন'], 'name': 'Assamese', 'gtts_code': 'as'},
    'ur': {'keywords': ['سلام', 'شکریہ', 'کیسے'], 'name': 'Urdu', 'gtts_code': 'ur'},
    'sa': {'keywords': ['नमस्कारः', 'धन्यवाद', 'कथम्'], 'name': 'Sanskrit', 'gtts_code': 'sa'},
}

# ========== VOICE ASSISTANT CLASS ==========
class KalkiVoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.current_language = 'en'
        self.conversation_context = []
        self.running = True
        self.sample_rate = 16000  # Hz

        # Calibrate with ambient noise (record 2 seconds)
        print("🔧 Calibrating microphone (2 seconds of ambient noise)...")
        self.calibrate(duration=2)
        print("✅ Ready! Speak in any supported language.\n")

        pygame.mixer.init()

    def record_audio(self, duration=5):
        """Record audio using sounddevice, return bytes (WAV in memory)"""
        try:
            print("🎤 Listening...")
            recording = sd.rec(int(duration * self.sample_rate),
                               samplerate=self.sample_rate,
                               channels=1,
                               dtype='int16')
            sd.wait()
            # Convert numpy array to bytes
            return recording.tobytes()
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return None

    def calibrate(self, duration=2):
        """Record ambient noise to set energy threshold"""
        try:
            recording = sd.rec(int(duration * self.sample_rate),
                               samplerate=self.sample_rate,
                               channels=1,
                               dtype='int16')
            sd.wait()
            rms = np.sqrt(np.mean(recording**2))
            self.recognizer.energy_threshold = max(rms * 1.5, 200)
            print(f"   Ambient noise level: {int(rms)} -> threshold set to {int(self.recognizer.energy_threshold)}")
        except Exception as e:
            print(f"   Calibration failed: {e}")

    def listen(self):
        """Record and transcribe using Google Speech Recognition"""
        raw_bytes = self.record_audio(duration=5)
        if not raw_bytes:
            return None
        try:
            # Create AudioData object from raw bytes (16-bit PCM, mono, sample_rate)
            audio = sr.AudioData(raw_bytes, self.sample_rate, 2)  # 2 bytes per sample
            text = self.recognizer.recognize_google(audio, show_all=False)
            print(f"✅ You said: {text}")
            return text
        except sr.UnknownValueError:
            print("🤔 Could not understand.")
        except sr.RequestError as e:
            print(f"❌ Recognition error: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")
        return None

    def detect_language(self, text):
        if not text:
            return self.current_language, 'English'
        scores = {}
        for code, info in LANGUAGE_DETECTION.items():
            score = sum(1 for kw in info['keywords'] if kw.lower() in text.lower())
            scores[code] = score
        best = max(scores.items(), key=lambda x: x[1]) if scores else ('en', 0)
        if best[1] == 0:
            return self.current_language, LANGUAGE_DETECTION.get(self.current_language, {}).get('name', 'English')
        return best[0], LANGUAGE_DETECTION[best[0]]['name']

    def get_gemini_response(self, user_input, lang_code):
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            prompts = {
                'hi': "आप एक मददगार AI सहायक हैं। हिंदी में संवाद करें।",
                'en': "You are a helpful assistant. Answer naturally.",
                'mr': "तुम एक मदतनीस AI सहायक आहात. मराठीत उत्तर द्या.",
                'bn': "আপনি একজন সহায়ক AI সহকারী। বাংলায় উত্তর দিন।",
            }
            system = prompts.get(lang_code, prompts['en'])
            context = "\n".join([f"User: {ctx['text']}\nAI: {ctx['response']}" for ctx in self.conversation_context[-3:]])
            full = f"{system}\n\n{context}\nUser: {user_input}\nAI:"
            response = model.generate_content(full)
            return response.text.strip()
        except Exception as e:
            return f"Sorry, error: {str(e)}"

    def speak(self, text, lang_code):
        try:
            lang_info = LANGUAGE_DETECTION.get(lang_code, LANGUAGE_DETECTION['en'])
            tts_lang = lang_info['gtts_code']
            print(f"🤖 {lang_info['name']}: {text}")

            def play():
                tts = gTTS(text=text, lang=tts_lang, slow=False)
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                pygame.mixer.music.load(fp, 'mp3')
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(50)

            threading.Thread(target=play, daemon=True).start()
        except Exception as e:
            print(f"TTS Error: {e}")

    def run(self):
        print("\n🚀 KALKI AI - Voice Assistant (sounddevice + gTTS)")
        print("💡 Say 'goodbye', 'exit', or 'quit' to stop.\n")
        self.speak("Hello! I am Kalki, your multilingual voice assistant. How can I help you?", 'en')

        while self.running:
            user_text = self.listen()
            if not user_text:
                continue

            if any(word in user_text.lower() for word in ['exit', 'quit', 'goodbye', 'bye', 'बंद', 'विदा']):
                self.speak("Goodbye! Have a great day!", 'en')
                break

            new_lang, lang_name = self.detect_language(user_text)
            if new_lang != self.current_language:
                print(f"🔄 Switching to {lang_name}")
                self.current_language = new_lang

            print("💭 Thinking...")
            ai_response = self.get_gemini_response(user_text, self.current_language)
            self.speak(ai_response, self.current_language)

            self.conversation_context.append({'text': user_text, 'response': ai_response})
            if len(self.conversation_context) > 5:
                self.conversation_context.pop(0)

if __name__ == "__main__":
    assistant = KalkiVoiceAssistant()
    assistant.run()
