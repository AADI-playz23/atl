import speech_recognition as sr
import google.generativeai as genai
from gtts import gTTS
import pygame
import io
import re
import threading
from collections import Counter

# Configure Gemini
GEMINI_API_KEY = "AIzaSyAB6GQYyjCYURc--HOkXB41vAU41HIankM"
genai.configure(api_key=GEMINI_API_KEY)

# Enhanced Language Mapping with better detection
LANGUAGE_DETECTION = {
    # Hindi and variations
    'hi': {
        'keywords': ['नमस्ते', 'धन्यवाद', 'कैसे', 'हैं', 'हूं', 'है', 'मैं', 'तुम', 'आप'],
        'patterns': [r'[\u0900-\u097F]'],
        'name': 'Hindi',
        'gtts_code': 'hi'
    },
    # English
    'en': {
        'keywords': ['hello', 'thank', 'you', 'how', 'are', 'what', 'where', 'when'],
        'patterns': [r'[a-zA-Z]'],
        'name': 'English', 
        'gtts_code': 'en'
    },
    # Marathi
    'mr': {
        'keywords': ['नमस्कार', 'धन्यवाद', 'कसे', 'आहात', 'मी', 'तू', 'तुम्ही'],
        'patterns': [r'[\u0900-\u097F]'],
        'name': 'Marathi',
        'gtts_code': 'mr'
    },
    # Bengali
    'bn': {
        'keywords': ['নমস্কার', 'ধন্যবাদ', 'কেমন', 'আছেন', 'আমি', 'তুমি', 'আপনি'],
        'patterns': [r'[\u0980-\u09FF]'],
        'name': 'Bengali',
        'gtts_code': 'bn'
    },
    # Tamil
    'ta': {
        'keywords': ['வணக்கம்', 'நன்றி', 'எப்படி', 'இருக்கிறீர்கள்', 'நான்', 'நீங்கள்'],
        'patterns': [r'[\u0B80-\u0BFF]'],
        'name': 'Tamil',
        'gtts_code': 'ta'
    },
    # Telugu
    'te': {
        'keywords': ['నమస్కారం', 'ధన్యవాదాలు', 'ఎలా', 'ఉన్నారు', 'నేను', 'మీరు'],
        'patterns': [r'[\u0C00-\u0C7F]'],
        'name': 'Telugu',
        'gtts_code': 'te'
    },
    # Gujarati
    'gu': {
        'keywords': ['નમસ્તે', 'આભાર', 'કેમ', 'છો', 'હું', 'તમે'],
        'patterns': [r'[\u0A80-\u0AFF]'],
        'name': 'Gujarati',
        'gtts_code': 'gu'
    },
    # Punjabi
    'pa': {
        'keywords': ['ਸਤ ਸ੍ਰੀ ਅਕਾਲ', 'ਧੰਨਵਾਦ', 'ਕਿਵੇਂ', 'ਹੋ', 'ਮੈਂ', 'ਤੁਸੀਂ'],
        'patterns': [r'[\u0A00-\u0A7F]'],
        'name': 'Punjabi',
        'gtts_code': 'pa'
    },
    # Malayalam
    'ml': {
        'keywords': ['നമസ്കാരം', 'നന്ദി', 'എങ്ങനെ', 'ഉണ്ട്', 'ഞാൻ', 'നിങ്ങൾ'],
        'patterns': [r'[\u0D00-\u0D7F]'],
        'name': 'Malayalam',
        'gtts_code': 'ml'
    },
    # Kannada
    'kn': {
        'keywords': ['ನಮಸ್ಕಾರ', 'ಧನ್ಯವಾದ', 'ಹೇಗೆ', 'ಇದ್ದೀರಾ', 'ನಾನು', 'ನೀವು'],
        'patterns': [r'[\u0C80-\u0CFF]'],
        'name': 'Kannada',
        'gtts_code': 'kn'
    },
    # Odia
    'or': {
        'keywords': ['ନମସ୍କାର', 'ଧନ୍ୟବାଦ', 'କେମିତି', 'ଅଛନ୍ତି', 'ମୁଁ', 'ଆପଣ'],
        'patterns': [r'[\u0B00-\u0B7F]'],
        'name': 'Odia',
        'gtts_code': 'or'
    },
    # Assamese
    'as': {
        'keywords': ['নমস্কাৰ', 'ধন্যবাদ', 'কেমন', 'আছে', 'মই', 'আপুনি'],
        'patterns': [r'[\u0980-\u09FF]'],
        'name': 'Assamese',
        'gtts_code': 'as'
    },
    # Urdu
    'ur': {
        'keywords': ['سلام', 'شکریہ', 'کیسے', 'ہیں', 'میں', 'آپ'],
        'patterns': [r'[\u0600-\u06FF]'],
        'name': 'Urdu',
        'gtts_code': 'ur'
    },
    # Sanskrit
    'sa': {
        'keywords': ['नमस्कारः', 'धन्यवाद', 'कथम्', 'अस्ति', 'अहम्', 'भवान्'],
        'patterns': [r'[\u0900-\u097F]'],
        'name': 'Sanskrit',
        'gtts_code': 'sa'
    },
    # Chhattisgarhi
    'hne': {
        'keywords': ['राम राम', 'धन्यबाद', 'का हाव', 'हस', 'हम', 'तू'],
        'patterns': [r'[\u0900-\u097F]'],
        'name': 'Chhattisgarhi',
        'gtts_code': 'hi'  # Fallback to Hindi
    }
}

class AdvancedVoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.current_language = 'en'
        self.language_confidence = {}
        self.conversation_context = []
        
        # Calibrate microphone
        print("🔧 Calibrating microphone for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("✅ Microphone calibrated!")
        
        # Initialize pygame for audio
        pygame.mixer.init()
    
    def enhanced_listen(self):
        """Advanced listening with multiple recognition attempts"""
        try:
            with self.microphone as source:
                print("🎤 Listening... (Speak now)")
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=10)
            
            # Try multiple recognition methods
            text = self.try_multiple_recognition(audio)
            return text
            
        except sr.WaitTimeoutError:
            print("⏰ No speech detected")
            return None
        except Exception as e:
            print(f"❌ Listening error: {e}")
            return None
    
    def try_multiple_recognition(self, audio):
        """Try different speech recognition approaches"""
        recognition_methods = [
            self.recognize_with_google,
            self.recognize_with_language_hint,
            self.recognize_with_auto_detect
        ]
        
        for method in recognition_methods:
            try:
                text = method(audio)
                if text and len(text.strip()) > 2:  # Valid response
                    print(f"✅ Recognized: {text}")
                    return text
            except Exception as e:
                continue
        
        return None
    
    def recognize_with_google(self, audio):
        """Basic Google recognition with auto-detect"""
        return self.recognizer.recognize_google(audio, show_all=False)
    
    def recognize_with_language_hint(self, audio):
        """Recognition with language hint based on current language"""
        language_hints = {
            'hi': 'hi-IN', 'en': 'en-US', 'mr': 'mr-IN', 'bn': 'bn-IN',
            'ta': 'ta-IN', 'te': 'te-IN', 'gu': 'gu-IN', 'pa': 'pa-IN',
            'ml': 'ml-IN', 'kn': 'kn-IN', 'or': 'or-IN', 'as': 'as-IN',
            'ur': 'ur-PK', 'sa': 'sa-IN'
        }
        
        hint = language_hints.get(self.current_language, 'en-US')
        return self.recognizer.recognize_google(audio, language=hint)
    
    def recognize_with_auto_detect(self, audio):
        """Try recognition with multiple language hints"""
        common_languages = ['hi-IN', 'en-US', 'mr-IN', 'bn-IN', 'ta-IN', 'te-IN']
        
        for lang in common_languages:
            try:
                return self.recognizer.recognize_google(audio, language=lang)
            except:
                continue
        return None
    
    def detect_language_intelligently(self, text):
        """Advanced language detection using multiple methods"""
        if not text:
            return 'en', 'English'
        
        text_lower = text.lower()
        scores = {}
        
        # Method 1: Script detection
        script_scores = self.detect_by_script(text)
        for lang, score in script_scores.items():
            scores[lang] = scores.get(lang, 0) + score * 0.4
        
        # Method 2: Keyword matching
        keyword_scores = self.detect_by_keywords(text_lower)
        for lang, score in keyword_scores.items():
            scores[lang] = scores.get(lang, 0) + score * 0.4
        
        # Method 3: Context awareness
        context_score = self.detect_by_context()
        for lang, score in context_score.items():
            scores[lang] = scores.get(lang, 0) + score * 0.2
        
        # Get best match
        if scores:
            best_lang = max(scores.items(), key=lambda x: x[1])
            if best_lang[1] > 0.3:  confidence threshold
                lang_info = LANGUAGE_DETECTION[best_lang[0]]
                return best_lang[0], lang_info['name']
        
        # Fallback to current language or English
        return self.current_language, LANGUAGE_DETECTION.get(self.current_language, {}).get('name', 'English')
    
    def detect_by_script(self, text):
        """Detect language by Unicode script ranges"""
        scores = {}
        
        for lang_code, lang_info in LANGUAGE_DETECTION.items():
            for pattern in lang_info['patterns']:
                if re.search(pattern, text):
                    scores[lang_code] = scores.get(lang_code, 0) + 1
        
        return scores
    
    def detect_by_keywords(self, text):
        """Detect language by common keywords"""
        scores = {}
        
        for lang_code, lang_info in LANGUAGE_DETECTION.items():
            for keyword in lang_info['keywords']:
                if keyword.lower() in text:
                    scores[lang_code] = scores.get(lang_code, 0) + 1
        
        return scores
    
    def detect_by_context(self):
        """Use conversation context for language detection"""
        if not self.conversation_context:
            return {self.current_language: 1}
        
        # Count recent language usage
        recent_langs = [ctx['lang'] for ctx in self.conversation_context[-3:]]
        lang_counts = Counter(recent_langs)
        
        if lang_counts:
            most_common = lang_counts.most_common(1)[0]
            return {most_common[0]: most_common[1] / 3.0}
        
        return {self.current_language: 1}
    
    def get_gemini_response(self, user_input, language_code='en'):
        """Get AI response with language context"""
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            # Language-specific system prompts
            system_prompts = {
                'hi': "आप एक मददगार AI सहायक हैं। हिंदी में संवाद करें और स्वाभाविक, बातचीत जैसी भाषा का प्रयोग करें।",
                'en': "You are a helpful AI assistant. Respond in natural, conversational language.",
                'mr': "तुम एक मदतनीस AI सहायक आहात. मराठीत स्वाभाविक संवाद साधा.",
                'bn': "আপনি একজন সহায়ক AI সহকারী। বাংলায় প্রাকৃতিক কথোপকথনমূলক ভাষা ব্যবহার করুন।",
                'ta': "நீங்கள் ஒரு உதவிகரமான AI உதவியாளர். தமிழில் இயல்பான உரையாடல் மொழியைப் பயன்படுத்தவும்.",
                'te': "మీరు సహాయకారి AI అసిస్టెంట్. తెలుగులో సహజమైన సంభాషణ భాషను ఉపయోగించండి.",
                'as': "আপুনি এজন সহায়ক AI সহকৰ্মী। অসমীয়াত স্বাভাৱিক কথোপকথনৰ ভাষা ব্যৱহাৰ কৰক।",
                'hne': "तउ एक मददगार AI सहायक हवे। छत्तीसगढ़ी में स्वाभाविक बतियात भाषा इस्तेमाल करे।"
            }
            
            system_prompt = system_prompts.get(language_code, system_prompts['en'])
            
            # Add conversation context
            context = "\n".join([f"User: {ctx['text']}\nAI: {ctx['response']}" 
                               for ctx in self.conversation_context[-2:]])
            
            full_prompt = f"{system_prompt}\n\n{context}\nUser: {user_input}\nAI:"
            
            response = model.generate_content(full_prompt)
            return response.text.strip()
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def speak_with_pygame(self, text, language_code='en'):
        """Convert text to speech with language support"""
        try:
            lang_info = LANGUAGE_DETECTION.get(language_code, LANGUAGE_DETECTION['en'])
            print(f"🤖 {lang_info['name']}: {text}")
            
            # Clean text
            clean_text = re.sub(r'[*_#`]', '', text)
            
            # Use appropriate TTS language code
            tts_lang = lang_info.get('gtts_code', 'en')
            
            # Generate speech
            tts = gTTS(text=clean_text, lang=tts_lang, slow=False)
            audio_file = io.BytesIO()
            tts.write_to_fp(audio_file)
            audio_file.seek(0)
            
            # Play audio
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
            pygame.mixer.music.load(audio_file, 'mp3')
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
                
        except Exception as e:
            print(f"❌ TTS Error: {e}")
    
    def update_conversation_context(self, user_text, ai_response, language):
        """Maintain conversation context"""
        self.conversation_context.append({
            'text': user_text,
            'response': ai_response,
            'lang': language
        })
        
        # Keep only last 5 conversations
        if len(self.conversation_context) > 5:
            self.conversation_context.pop(0)
    
    def show_language_status(self):
        """Display current language status"""
        lang_info = LANGUAGE_DETECTION.get(self.current_language, LANGUAGE_DETECTION['en'])
        print(f"🌐 Current Language: {lang_info['name']} ({self.current_language})")
    
    def run(self):
        """Main assistant loop"""
        print("🚀 ADVANCED VOICE AI ASSISTANT")
        print("✨ Speaks like Google Assistant with Multi-language Support")
        print("=" * 60)
        
        # Show supported languages
        print("\n🗣️  SUPPORTED LANGUAGES:")
        languages = list(LANGUAGE_DETECTION.keys())
        for i in range(0, len(languages), 4):
            print("   " + " | ".join(f"{lang}: {LANGUAGE_DETECTION[lang]['name']}" 
                                   for lang in languages[i:i+4]))
        
        print("\n💡 Just speak naturally in any supported language!")
        print("   I'll automatically detect and respond in your language.")
        print("=" * 60)
        
        # Welcome message
        welcome_text = "Hello! I'm your advanced voice assistant. I can understand and speak multiple languages. How can I help you today?"
        self.speak_with_pygame(welcome_text, 'en')
        
        while True:
            try:
                print("\n" + "-" * 40)
                self.show_language_status()
                print("💬 Speak now...")
                
                # Listen for speech
                user_text = self.enhanced_listen()
                
                if not user_text:
                    continue
                
                # Check for exit commands
                exit_commands = {
                    'en': ['exit', 'quit', 'stop', 'goodbye'],
                    'hi': ['बंद', 'रुको', 'समाप्त', 'विदा'],
                    'mr': ['थांब', 'बंद', 'संपव', 'विदा'],
                    'bn': ['বন্ধ', 'থামো', 'সমাপ্ত', 'বিদায়'],
                    'ta': ['நிறுத்து', 'முடி', 'போதும்', 'விடைபெறுகிறேன்'],
                    'te': ['నిలిపి', 'ముగించు', 'పూర్తి', 'వీడ్కోలు'],
                    'as': ['বন্ধ', 'থওক', 'সমাপ্ত', 'বিদায়'],
                    'hne': ['रुक', 'बंद', 'खतम', 'अलविदा']
                }
                
                should_exit = False
                for lang, commands in exit_commands.items():
                    if any(cmd in user_text.lower() for cmd in commands):
                        should_exit = True
                        break
                
                if should_exit:
                    goodbye_text = {
                        'en': "Goodbye! Have a great day!",
                        'hi': "अलविदा! आपका दिन शुभ हो!",
                        'mr': "नमस्कार! तुमचा दिवस चांगला जावो!",
                        'bn': "বিদায়! আপনার দিনটি ভালো কাটুক!",
                        'ta': "பிரியாவிடை! உங்கள் நாள் நலமாக அமையட்டும்!",
                        'te': "వీడ్కోలు! మీ రోజు శుభంగా జరుగుతుందని కోరుకుంటున్నాను!",
                        'as': "বিদায়! আপোনাৰ দিনটো শুভ হওক!",
                        'hne': "अलविदा! राउर दिन बढ़िया जाव!"
                    }
                    farewell = goodbye_text.get(self.current_language, goodbye_text['en'])
                    self.speak_with_pygame(farewell, self.current_language)
                    break
                
                # Detect language
                detected_lang, lang_name = self.detect_language_intelligently(user_text)
                
                if detected_lang != self.current_language:
                    print(f"🔄 Language switched: {lang_name}")
                    self.current_language = detected_lang
                
                # Get AI response
                print(f"💭 Processing in {lang_name}...")
                ai_response = self.get_gemini_response(user_text, self.current_language)
                
                # Speak response
                self.speak_with_pygame(ai_response, self.current_language)
                
                # Update context
                self.update_conversation_context(user_text, ai_response, self.current_language)
                
            except KeyboardInterrupt:
                print("\n👋 Shutting down...")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                continue

if __name__ == "__main__":
    assistant = AdvancedVoiceAssistant()
    assistant.run()