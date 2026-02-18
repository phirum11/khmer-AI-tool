"""
AI Speech-to-Text Pro - Full Enterprise Version
- Triple engine: faster-whisper (local) + Vosk (offline) + Google Cloud API
- Advanced UI with dark/light theme toggle
- Real-time progress with percentage display
- Export to multiple formats (TXT, SRT, VTT, JSON)
- Audio waveform visualization
- Keyboard shortcuts
- Settings persistence
- Session management
- Advanced Text-to-Speech with voice customization
- Enhanced Khmer language support with forced detection
- Copy functionality for selected text and entire transcript
- OPTIMIZED for SPEED (1-5 minute processing)
"""

import sys
import os
import json
import tempfile
import asyncio
import wave
from datetime import timedelta, datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
import traceback

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QProgressBar, QGroupBox, QLineEdit,
    QRadioButton, QButtonGroup, QCheckBox, QSplitter, QTabWidget,
    QTextEdit, QSpinBox, QDoubleSpinBox, QDialog, QDialogButtonBox,
    QFormLayout, QMenuBar, QMenu, QStatusBar, QToolBar, QStyleFactory,
    QFrame ,QProgressDialog
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QUrl, QSettings, QTimer, QDateTime 
)
from PyQt6.QtGui import (
    QFont, QPalette, QColor, QAction, QKeySequence, QTextCursor, QCursor ,QPainter ,QPen
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

# Transcription engines
import faster_whisper
from vosk import Model as VoskModel, KaldiRecognizer

# Google Cloud Speech (optional)
try:
    from google.cloud import speech_v1
    from google.cloud.speech_v1 import RecognitionConfig, RecognitionAudio
    from google.api_core.exceptions import GoogleAPICallError
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False

# TTS
import edge_tts

# Audio processing
try:
    import pydub
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

# Global exception handler
def global_excepthook(exc_type, exc_value, exc_traceback):
    print("=" * 60)
    print("CRASH DETECTED!")
    print("=" * 60)
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    print("=" * 60)
    input("Press Enter to exit...")

sys.excepthook = global_excepthook

# ==================== LANGUAGE SUPPORT CONFIGURATION ====================

# Define which languages are supported by each engine
ENGINE_SUPPORT = {
    "faster-whisper": {
        "supported": ["en", "zh", "km", "ja", "ko", "fr", "de", "es", "ru", "ar"],
        "notes": "Khmer may default to Chinese due to limited training data",
        "requires_internet": False,
        "cost": "Free"
    },
    "vosk": {
        "supported": ["en", "zh", "fr", "de", "es", "ru", "ja", "ko"],
        "notes": "Khmer NOT supported",
        "requires_internet": False,
        "cost": "Free"
    },
    "google": {
        "supported": ["en", "zh", "km", "ja", "ko", "fr", "de", "es", "ru", "ar", "th", "vi"],
        "notes": "Best for Khmer",
        "requires_internet": True,
        "cost": "Paid ($0.016/min)"
    }
}

# ==================== DATA CLASSES ====================

@dataclass
class TranscriptionSegment:
    """Represents a transcribed segment with metadata."""
    start: float
    end: float
    text: str
    confidence: float = 1.0
    speaker: Optional[str] = None
    detected_language: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptionSegment':
        return cls(**data)

@dataclass
class TranscriptionSession:
    """Represents a complete transcription session."""
    file_path: str
    file_name: str
    file_size: int
    duration: float
    language: str
    engine: str
    segments: List[TranscriptionSegment]
    created_at: str
    modified_at: str
    detected_language: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'file_path': self.file_path,
            'file_name': self.file_name,
            'file_size': self.file_size,
            'duration': self.duration,
            'language': self.language,
            'engine': self.engine,
            'segments': [s.to_dict() for s in self.segments],
            'created_at': self.created_at,
            'modified_at': self.modified_at,
            'detected_language': self.detected_language
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptionSession':
        segments = [TranscriptionSegment.from_dict(s) for s in data['segments']]
        return cls(
            file_path=data['file_path'],
            file_name=data['file_name'],
            file_size=data['file_size'],
            duration=data['duration'],
            language=data['language'],
            engine=data['engine'],
            segments=segments,
            created_at=data['created_at'],
            modified_at=data['modified_at'],
            detected_language=data.get('detected_language')
        )

# ==================== THREAD WORKERS ====================

class ModelLoaderThread(QThread):
    """Load faster-whisper model in background with progress."""
    progress = pyqtSignal(str)
    model_loaded = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, model_size: str = "tiny"):
        super().__init__()
        self.model_size = model_size
    
    def run(self):
        try:
            self.progress.emit(f"🔄 Downloading/loading {self.model_size} model...")
            self.progress.emit("⏳ This may take a few minutes on first run...")
            self.progress.emit("📦 Model size: ~75MB for tiny, ~300MB for base")
            
            # Check if model is already downloaded
            import os
            from pathlib import Path
            
            # Common cache locations
            cache_dirs = [
                Path.home() / ".cache" / "huggingface" / "hub",
                Path.home() / ".cache" / "faster_whisper",
                Path.home() / ".cache" / "whisper"
            ]
            
            for cache_dir in cache_dirs:
                if cache_dir.exists():
                    self.progress.emit(f"📁 Checking cache: {cache_dir}")
                    # Count existing model files
                    model_files = list(cache_dir.glob(f"*{self.model_size}*"))
                    if model_files:
                        self.progress.emit(f"✅ Found existing {self.model_size} model in cache")
            
            self.progress.emit("🌐 Connecting to Hugging Face hub...")
            
            # Configure model loading with better error handling
            try:
                model = faster_whisper.WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",  # Use int8 for better compatibility
                    download_root=None,    # Uses default cache directory
                    num_workers=2,         # Reduced for stability
                    cpu_threads=4,         # Balanced for most systems
                    local_files_only=False  # Allow downloading
                )
            except Exception as e:
                if "404" in str(e) or "not found" in str(e).lower():
                    # Try alternative model name
                    self.progress.emit("⚠️ Model not found, trying alternative...")
                    alt_models = {
                        "tiny": "tiny.en",
                        "base": "base.en",
                        "small": "small.en"
                    }
                    alt_name = alt_models.get(self.model_size, self.model_size)
                    model = faster_whisper.WhisperModel(
                        alt_name,
                        device="cpu",
                        compute_type="int8",
                        download_root=None,
                        num_workers=2,
                        cpu_threads=4
                    )
                else:
                    raise e
            
            self.progress.emit("✅ Model loaded successfully!")
            self.progress.emit("🚀 Ready for transcription")
            self.model_loaded.emit(model)
            
        except ImportError as e:
            self.error.emit(f"📦 faster-whisper not properly installed: {str(e)}\n\nPlease run: pip install --upgrade faster-whisper")
            
        except Exception as e:
            error_msg = str(e)
            
            # Categorize errors for better user feedback
            if "Connection" in error_msg or "timeout" in error_msg.lower() or "network" in error_msg.lower():
                self.error.emit(
                    "🌐 NETWORK ERROR\n\n"
                    "Cannot connect to download model. Please:\n"
                    "1. Check your internet connection\n"
                    "2. Disable VPN/proxy if active\n"
                    "3. Try again later\n\n"
                    f"Technical details: {error_msg}"
                )
                
            elif "404" in error_msg or "not found" in error_msg.lower():
                self.error.emit(
                    f"❌ MODEL NOT FOUND\n\n"
                    f"Model '{self.model_size}' could not be found.\n\n"
                    f"Available models: tiny, base, small, medium, large\n\n"
                    f"Technical details: {error_msg}"
                )
                
            elif "disk" in error_msg.lower() or "space" in error_msg.lower():
                self.error.emit(
                    "💾 DISK SPACE ERROR\n\n"
                    "Not enough disk space to download the model.\n"
                    "Please free up space and try again.\n\n"
                    f"Technical details: {error_msg}"
                )
                
            elif "permission" in error_msg.lower():
                self.error.emit(
                    "🔒 PERMISSION ERROR\n\n"
                    "Cannot write to cache directory.\n"
                    "Please check folder permissions:\n"
                    f"{Path.home() / '.cache'}\n\n"
                    f"Technical details: {error_msg}"
                )
                
            else:
                self.error.emit(
                    f"❌ FAILED TO LOAD MODEL\n\n"
                    f"Error: {error_msg}\n\n"
                    f"Suggestions:\n"
                    f"• Try a different model size (tiny, base, small)\n"
                    f"• Use Vosk or Google Cloud API instead\n"
                    f"• Check if faster-whisper is updated: pip install --upgrade faster-whisper"
                )

class FasterWhisperThread(QThread):
    """Optimized faster-whisper transcription for speed."""
    segment_ready = pyqtSignal(TranscriptionSegment)
    progress = pyqtSignal(int, int, float)
    language_detected = pyqtSignal(str, float)
    finished = pyqtSignal(TranscriptionSession)
    error = pyqtSignal(str)
    
    def __init__(self, model, file_path: str, language: Optional[str], 
                 task: str = "transcribe", vad: bool = True):
        super().__init__()
        self.model = model
        self.file_path = file_path
        self.language = language
        self.task = task
        self.vad = vad
        self._running = True
        self.detected_language = None
    
    def run(self):
        try:
            file_size = os.path.getsize(self.file_path)
            file_name = os.path.basename(self.file_path)
            
            duration = 0
            if AUDIO_PROCESSING_AVAILABLE:
                try:
                    audio = pydub.AudioSegment.from_file(self.file_path)
                    duration = len(audio) / 1000.0
                except:
                    pass
            
            # SPEED OPTIMIZATION: Use smaller beam size for faster processing
            # SPEED OPTIMIZATION: Disable VAD if not needed
            # SPEED OPTIMIZATION: Use lower temperature for faster sampling
            
            # Special handling for Khmer
            if self.language == "km":
                segments, info = self.model.transcribe(
                    self.file_path,
                    language="km",  # Force Khmer
                    task=self.task,
                    vad_filter=self.vad,
                    beam_size=3,  # Reduced from 5 for speed
                    best_of=3,    # Reduced from 5 for speed
                    temperature=0.0,  # Use 0 for deterministic faster output
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False,  # Set to False for speed
                    initial_prompt=None,
                    prefix=None,
                    hotwords=None,
                    without_timestamps=False,
                    word_timestamps=False  # Disable for speed if not needed
                )
                self.detected_language = info.language
                self.language_detected.emit(info.language, info.language_probability)
            else:
                # Regular handling for other languages
                segments, info = self.model.transcribe(
                    self.file_path,
                    language=self.language,
                    task=self.task,
                    vad_filter=self.vad,
                    beam_size=3,  # Reduced for speed
                    best_of=3,    # Reduced for speed
                    temperature=0.0,  # Deterministic faster output
                    condition_on_previous_text=False,  # Speed optimization
                    word_timestamps=False  # Speed optimization
                )
                self.detected_language = info.language
                self.language_detected.emit(info.language, info.language_probability)
            
            # Process segments
            segments_list = list(segments)
            total_segments = len(segments_list)
            
            # SPEED OPTIMIZATION: Batch process segments
            for i, seg in enumerate(segments_list):
                if not self._running:
                    break
                
                segment = TranscriptionSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    detected_language=self.detected_language
                )
                self.segment_ready.emit(segment)
                
                # Update progress less frequently for speed
                if i % max(1, total_segments // 20) == 0 or i == total_segments - 1:
                    self.progress.emit(i + 1, total_segments, seg.end)
            
            if self._running:
                session = TranscriptionSession(
                    file_path=self.file_path,
                    file_name=file_name,
                    file_size=file_size,
                    duration=duration,
                    language=self.language if self.language else "auto",
                    engine="faster-whisper",
                    segments=segments_list,
                    created_at=QDateTime.currentDateTime().toString(Qt.DateFormat.ISODate),
                    modified_at=QDateTime.currentDateTime().toString(Qt.DateFormat.ISODate),
                    detected_language=self.detected_language
                )
                self.finished.emit(session)
            
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"Transcription error: {str(e)}")
    
    def stop(self):
        self._running = False

class GoogleSpeechThread(QThread):
    """Optimized Google Cloud Speech-to-API transcription thread."""
    segment_ready = pyqtSignal(TranscriptionSegment)
    progress = pyqtSignal(int, int, float)
    language_detected = pyqtSignal(str, float)
    finished = pyqtSignal(TranscriptionSession)
    error = pyqtSignal(str)
    
    def __init__(self, api_key: str, file_path: str, language: str, 
                 use_enhanced: bool = True):
        super().__init__()
        self.api_key = api_key
        self.file_path = file_path
        self.language = language
        self.use_enhanced = use_enhanced
        self._running = True
    
    def run(self):
        if not GOOGLE_CLOUD_AVAILABLE:
            self.error.emit("Google Cloud Speech library not installed")
            return
        
        try:
            file_size = os.path.getsize(self.file_path)
            file_name = os.path.basename(self.file_path)
            
            client = speech_v1.SpeechClient(
                client_options={"api_key": self.api_key}
            )
            
            with open(self.file_path, 'rb') as f:
                content = f.read()
            
            # Configure recognition with Khmer support
            config = {
                "encoding": speech_v1.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
                "sample_rate_hertz": 16000,
                "language_code": self.language if self.language != "auto" else "km",  # Default to Khmer for auto
                "enable_automatic_punctuation": True,
                "enable_word_time_offsets": True,
                "enable_speaker_diarization": True,
                "diarization_speaker_count": 2,
                "model": "latest_long" if self.use_enhanced else "command_and_search",
                "use_enhanced": self.use_enhanced,
                "speech_contexts": [{
                    "phrases": []  # Add custom phrases if needed
                }]
            }
            
            # For auto language detection, include Khmer
            if self.language == "auto":
                config["language_codes"] = ["km", "en-US", "zh", "ja", "ko"]
            
            audio = speech_v1.RecognitionAudio(content=content)
            
            # SPEED OPTIMIZATION: Use short recognition for files under 1 minute
            if file_size < 10 * 1024 * 1024:  # Under 10MB
                response = client.recognize(config=config, audio=audio)
                results = response.results
                total_results = len(results)
                
                segments = []
                detected_lang = None
                
                for i, result in enumerate(results):
                    if not self._running:
                        break
                    
                    alternative = result.alternatives[0]
                    
                    # Get detected language if available
                    if hasattr(result, 'language_code'):
                        detected_lang = result.language_code
                        self.language_detected.emit(result.language_code, 1.0)
                    
                    segment = TranscriptionSegment(
                        start=0.0,
                        end=0.0,
                        text=alternative.transcript,
                        confidence=alternative.confidence,
                        detected_language=detected_lang
                    )
                    self.segment_ready.emit(segment)
                    segments.append(segment)
                    
                    self.progress.emit(i + 1, total_results, 0)
            else:
                # Use long running recognize for larger files
                operation = client.long_running_recognize(config=config, audio=audio)
                response = operation.result(timeout=300)
                
                total_results = len(response.results)
                segments = []
                detected_lang = None
                
                for i, result in enumerate(response.results):
                    if not self._running:
                        break
                    
                    alternative = result.alternatives[0]
                    
                    # Get detected language if available
                    if hasattr(result, 'language_code'):
                        detected_lang = result.language_code
                        self.language_detected.emit(result.language_code, 1.0)
                    
                    if alternative.words:
                        # Group words into larger chunks for fewer segments
                        words = list(alternative.words)
                        chunk_size = 20  # Increased from 10 for fewer segments
                        
                        for j in range(0, len(words), chunk_size):
                            group = words[j:j+chunk_size]
                            start_time = group[0].start_time.total_seconds()
                            end_time = group[-1].end_time.total_seconds()
                            text = ' '.join(w.word for w in group)
                            
                            speaker = None
                            if hasattr(group[0], 'speaker_tag'):
                                speaker = f"SPEAKER_{group[0].speaker_tag}"
                            
                            segment = TranscriptionSegment(
                                start=start_time,
                                end=end_time,
                                text=text,
                                confidence=alternative.confidence,
                                speaker=speaker,
                                detected_language=detected_lang
                            )
                            self.segment_ready.emit(segment)
                            segments.append(segment)
                    else:
                        segment = TranscriptionSegment(
                            start=0.0,
                            end=0.0,
                            text=alternative.transcript,
                            confidence=alternative.confidence,
                            detected_language=detected_lang
                        )
                        self.segment_ready.emit(segment)
                        segments.append(segment)
                    
                    # Update progress less frequently
                    if i % max(1, total_results // 20) == 0 or i == total_results - 1:
                        self.progress.emit(i + 1, total_results, 0)
            
            if self._running:
                duration = 0
                if AUDIO_PROCESSING_AVAILABLE:
                    try:
                        audio = pydub.AudioSegment.from_file(self.file_path)
                        duration = len(audio) / 1000.0
                    except:
                        pass
                
                session = TranscriptionSession(
                    file_path=self.file_path,
                    file_name=file_name,
                    file_size=file_size,
                    duration=duration,
                    language=self.language if self.language != "auto" else "km",
                    engine="google-cloud",
                    segments=segments,
                    created_at=QDateTime.currentDateTime().toString(Qt.DateFormat.ISODate),
                    modified_at=QDateTime.currentDateTime().toString(Qt.DateFormat.ISODate),
                    detected_language=detected_lang
                )
                self.finished.emit(session)
            
        except GoogleAPICallError as e:
            self.error.emit(f"Google API error: {str(e)}")
        except Exception as e:
            self.error.emit(f"Unexpected error: {str(e)}")
    
    def stop(self):
        self._running = False

class VoskThread(QThread):
    """Optimized Vosk offline transcription thread."""
    segment_ready = pyqtSignal(TranscriptionSegment)
    progress = pyqtSignal(int, int, float)
    language_detected = pyqtSignal(str, float)
    finished = pyqtSignal(TranscriptionSession)
    error = pyqtSignal(str)
    
    def __init__(self, model_path: str, file_path: str, language: str):
        super().__init__()
        self.model_path = model_path
        self.file_path = file_path
        self.language = language
        self._running = True
        self.temp_wav = None
    
    def run(self):
        try:
            file_size = os.path.getsize(self.file_path)
            file_name = os.path.basename(self.file_path)
            
            duration = 0
            if AUDIO_PROCESSING_AVAILABLE:
                try:
                    audio = pydub.AudioSegment.from_file(self.file_path)
                    duration = len(audio) / 1000.0
                except:
                    pass
            
            self.progress.emit(0, 100, 0)
            self.progress.emit(20, 100, 0)  # Jump to 20%
            
            model = VoskModel(self.model_path)
            
            self.progress.emit(30, 100, 0)
            
            # SPEED OPTIMIZATION: Pre-convert to WAV in memory if possible
            if not self.file_path.endswith('.wav'):
                audio = pydub.AudioSegment.from_file(self.file_path)
                audio = audio.set_channels(1).set_frame_rate(16000)
                
                # Use memory buffer instead of temp file for speed
                import io
                wav_buffer = io.BytesIO()
                audio.export(wav_buffer, format='wav')
                wav_buffer.seek(0)
                wf = wave.open(wav_buffer, 'rb')
                self.temp_wav = None
            else:
                wf = wave.open(self.file_path, 'rb')
                self.temp_wav = None
            
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                self.error.emit("Audio must be mono PCM")
                wf.close()
                return
            
            rec = KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(True)
            
            self.progress.emit(40, 100, 0)
            
            segments = []
            
            # SPEED OPTIMIZATION: Process in larger chunks
            chunk_size = 8000  # Increased from 4000
            total_frames = wf.getnframes()
            frames_processed = 0
            last_progress = 0
            
            while self._running:
                data = wf.readframes(chunk_size)
                if len(data) == 0:
                    break
                
                frames_processed += chunk_size
                
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if result.get('text'):
                        if result.get('result'):
                            words = result['result']
                            if words:
                                start = words[0]['start']
                                end = words[-1]['end']
                                text = ' '.join(w['word'] for w in words)
                                
                                segment = TranscriptionSegment(
                                    start=start,
                                    end=end,
                                    text=text,
                                    confidence=1.0,
                                    detected_language=self.language
                                )
                                self.segment_ready.emit(segment)
                                segments.append(segment)
                
                # Update progress less frequently
                if total_frames > 0:
                    progress_pct = 40 + int((frames_processed / total_frames) * 50)
                    if progress_pct - last_progress >= 5:  # Update every 5%
                        self.progress.emit(progress_pct, 100, frames_processed / wf.getframerate())
                        last_progress = progress_pct
            
            # Get final result
            final_result = json.loads(rec.FinalResult())
            if final_result.get('text'):
                segment = TranscriptionSegment(
                    start=duration - 2 if duration > 2 else 0,
                    end=duration,
                    text=final_result['text'],
                    confidence=1.0,
                    detected_language=self.language
                )
                self.segment_ready.emit(segment)
                segments.append(segment)
            
            wf.close()
            
            if self._running:
                self.progress.emit(100, 100, duration)
                self.language_detected.emit(self.language, 1.0)
                
                session = TranscriptionSession(
                    file_path=self.file_path,
                    file_name=file_name,
                    file_size=file_size,
                    duration=duration,
                    language=self.language,
                    engine="vosk",
                    segments=segments,
                    created_at=QDateTime.currentDateTime().toString(Qt.DateFormat.ISODate),
                    modified_at=QDateTime.currentDateTime().toString(Qt.DateFormat.ISODate),
                    detected_language=self.language
                )
                self.finished.emit(session)
            
        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"Vosk error: {str(e)}")
    
    def stop(self):
        self._running = False

class TTSThread(QThread):
    """Enhanced TTS thread with multiple voice options."""
    audio_ready = pyqtSignal(str)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    
    VOICE_MAP = {
        "en": ["en-US-JennyNeural", "en-US-GuyNeural", "en-GB-SoniaNeural", "en-AU-NatashaNeural"],
        "zh": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural", "zh-TW-HsiaoChenNeural"],
        "km": ["km-KH-SreymomNeural"],
        "ja": ["ja-JP-NanamiNeural", "ja-JP-KeitaNeural"],
        "ko": ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural"],
        "fr": ["fr-FR-DeniseNeural", "fr-FR-HenriNeural"],
        "de": ["de-DE-KatjaNeural", "de-DE-ConradNeural"],
        "es": ["es-ES-ElviraNeural", "es-ES-AlvaroNeural"]
    }
    
    def __init__(self, text: str, language: str, voice: str = None,
                 rate: int = 0, volume: int = 0):
        super().__init__()
        self.text = text
        self.language = language
        self.custom_voice = voice
        self.rate = rate
        self.volume = volume
    
    def run(self):
        try:
            # Get available voices for language
            voices = self.VOICE_MAP.get(self.language, self.VOICE_MAP["en"])
            
            # Use custom voice if provided and valid, otherwise use first available
            if self.custom_voice and self.custom_voice in voices:
                voice = self.custom_voice
            else:
                voice = voices[0]
                self.custom_voice = voice
            
            # Create temp file
            fd, path = tempfile.mkstemp(suffix='.mp3')
            os.close(fd)
            
            self.progress.emit(f"Generating speech with {voice}...")
            
            # Configure TTS
            communicate = edge_tts.Communicate(self.text, voice)
            
            # Apply rate and volume if specified
            if self.rate != 0 or self.volume != 0:
                rate_str = f"+{self.rate}%" if self.rate > 0 else f"{self.rate}%"
                volume_str = f"+{self.volume}%" if self.volume > 0 else f"{self.volume}%"
                communicate = edge_tts.Communicate(
                    self.text, 
                    voice,
                    rate=rate_str,
                    volume=volume_str
                )
            
            # Generate speech
            async def _generate():
                await communicate.save(path)
            
            asyncio.run(_generate())
            
            self.progress.emit("Speech generated successfully")
            self.audio_ready.emit(path)
            
        except Exception as e:
            self.error.emit(f"TTS failed: {str(e)}")

# ==================== CUSTOM WIDGETS ====================

class WaveformWidget(QWidget):
    """Simple waveform visualization widget."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)
        self.segments = []
        self.current_position = 0
        self.setStyleSheet("background-color: #2d2d3a; border-radius: 5px;")
    
    def set_segments(self, segments: List[TranscriptionSegment]):
        self.segments = segments
        self.update()
    
    def set_position(self, position: float):
        self.current_position = position
        self.update()
    
    def paintEvent(self, event):
        if not self.segments:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        painter.fillRect(0, 0, width, height, QColor(45, 45, 58))
        
        total_duration = max(s.end for s in self.segments)
        
        for segment in self.segments:
            x1 = int((segment.start / total_duration) * width)
            x2 = int((segment.end / total_duration) * width)
            
            color = QColor(109, 93, 252)
            painter.fillRect(x1, 5, x2 - x1, height - 10, color)
            painter.setPen(QPen(QColor(255, 255, 255, 50), 1))
            painter.drawRect(x1, 5, x2 - x1, height - 10)
        
        if self.current_position > 0:
            x = int((self.current_position / total_duration) * width)
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawLine(x, 0, x, height)

class SessionManager:
    """Manages transcription sessions (save/load)."""
    
    def __init__(self):
        self.sessions_dir = Path.home() / '.ai_transcriber' / 'sessions'
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
    
    def save_session(self, session: TranscriptionSession, name: Optional[str] = None):
        if name is None:
            name = f"{session.file_name}_{session.created_at.replace(':', '-')}"
        
        file_path = self.sessions_dir / f"{name}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
        
        return str(file_path)
    
    def load_session(self, file_path: str) -> Optional[TranscriptionSession]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return TranscriptionSession.from_dict(data)
        except Exception as e:
            print(f"Failed to load session: {e}")
            return None

# ==================== DIALOGS ====================
class SettingsDialog(QDialog):
    """Settings dialog for advanced configuration."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(550)
        self.main_window = parent  # Store reference to main window
        
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        
        # ==================== GENERAL TAB ====================
        general_tab = QWidget()
        general_layout = QFormLayout(general_tab)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "System"])
        general_layout.addRow("Theme:", self.theme_combo)
        
        self.auto_save = QCheckBox("Auto-save sessions")
        self.auto_save.setChecked(True)
        general_layout.addRow("", self.auto_save)
        
        self.save_format = QComboBox()
        self.save_format.addItems(["JSON", "TXT", "SRT", "All"])
        general_layout.addRow("Save format:", self.save_format)
        
        # Session directory info
        session_dir = Path.home() / '.ai_transcriber' / 'sessions'
        session_label = QLabel(f"📁 {session_dir}")
        session_label.setWordWrap(True)
        session_label.setStyleSheet("font-size: 8pt; color: gray;")
        general_layout.addRow("Session dir:", session_label)
        
        tabs.addTab(general_tab, "General")
        
        # ==================== TRANSCRIPTION TAB ====================
        trans_tab = QWidget()
        trans_layout = QFormLayout(trans_tab)
        
        # Default engine selection
        self.default_engine = QComboBox()
        self.default_engine.addItems(["faster-whisper", "Vosk", "Google Cloud"])
        trans_layout.addRow("Default engine:", self.default_engine)
        
        # Model size selection
        self.model_size = QComboBox()
        self.model_size.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_size.setCurrentText("tiny")
        self.model_size.setToolTip(
            "tiny: Fastest, 75MB\n"
            "base: Balanced, 150MB\n"
            "small: Good accuracy, 300MB\n"
            "medium: Better accuracy, 1.5GB\n"
            "large: Best accuracy, 3GB"
        )
        trans_layout.addRow("Model size:", self.model_size)
        
        # Model info label
        self.model_info = QLabel("ℹ️ tiny: 75MB, fastest for most uses")
        self.model_info.setStyleSheet("color: #4CAF50; font-size: 8pt;")
        self.model_size.currentTextChanged.connect(self.update_model_info)
        trans_layout.addRow("", self.model_info)
        
        # Download model button with status
        download_layout = QHBoxLayout()
        self.download_btn = QPushButton("⬇️ Download Model Now")
        self.download_btn.clicked.connect(self.download_model)
        self.download_btn.setMinimumHeight(35)
        download_layout.addWidget(self.download_btn)
        
        # Check if model exists
        self.model_status_label = QLabel(self.check_model_status())
        self.model_status_label.setStyleSheet("font-size: 8pt;")
        download_layout.addWidget(self.model_status_label)
        download_layout.addStretch()
        trans_layout.addRow("", download_layout)
        
        # Cache location info
        cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
        cache_label = QLabel(f"📦 Cache: {cache_dir}")
        cache_label.setWordWrap(True)
        cache_label.setStyleSheet("font-size: 8pt; color: gray;")
        trans_layout.addRow("", cache_label)
        
        # VAD filter
        self.vad_filter = QCheckBox("Enable VAD filter (voice activity detection)")
        self.vad_filter.setChecked(False)
        self.vad_filter.setToolTip("Helps filter out non-speech parts, but slower")
        trans_layout.addRow("", self.vad_filter)
        
        # Speed mode
        self.speed_mode = QCheckBox("Speed Mode (faster, lower accuracy)")
        self.speed_mode.setChecked(True)
        self.speed_mode.setToolTip("Enables optimizations for 2-5x faster processing")
        trans_layout.addRow("", self.speed_mode)
        
        # Compute type
        self.compute_type = QComboBox()
        self.compute_type.addItems(["int8", "int8_float16", "float16", "float32"])
        self.compute_type.setCurrentText("int8")
        self.compute_type.setToolTip(
            "int8: Fastest, lowest memory\n"
            "int8_float16: Balanced\n"
            "float16: Better accuracy, more memory\n"
            "float32: Best accuracy, most memory"
        )
        trans_layout.addRow("Compute type:", self.compute_type)
        
        tabs.addTab(trans_tab, "Transcription")
        
        # ==================== VOSK TAB ====================
        vosk_tab = QWidget()
        vosk_layout = QFormLayout(vosk_tab)
        
        # Vosk models directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, 'models')
        
        vosk_models_label = QLabel(f"📁 {models_dir}")
        vosk_models_label.setWordWrap(True)
        vosk_layout.addRow("Models directory:", vosk_models_label)
        
        # Check for existing models
        vosk_models = {
            "en": "vosk-model-small-en-us-0.15",
            "zh": "vosk-model-small-cn-0.22",
            "fr": "vosk-model-small-fr-0.22",
            "de": "vosk-model-small-de-0.15",
            "es": "vosk-model-small-es-0.42",
            "ru": "vosk-model-small-ru-0.22",
            "ja": "vosk-model-small-ja-0.22",
            "ko": "vosk-model-ko-0.22"
        }
        
        vosk_status = QTextEdit()
        vosk_status.setReadOnly(True)
        vosk_status.setMaximumHeight(150)
        
        status_text = "Installed Vosk models:\n"
        found_any = False
        for lang, model_name in vosk_models.items():
            model_path = os.path.join(models_dir, model_name)
            if os.path.exists(model_path):
                status_text += f"✅ {lang}: {model_name}\n"
                found_any = True
            else:
                status_text += f"❌ {lang}: {model_name} (not found)\n"
        
        if not found_any:
            status_text += "\n⚠️ No Vosk models found. Download from:\nhttps://alphacephei.com/vosk/models"
        
        vosk_status.setText(status_text)
        vosk_layout.addRow("Model status:", vosk_status)
        
        # Download link
        vosk_link = QLabel('<a href="https://alphacephei.com/vosk/models">📥 Download Vosk Models</a>')
        vosk_link.setOpenExternalLinks(True)
        vosk_layout.addRow("", vosk_link)
        
        tabs.addTab(vosk_tab, "Vosk")
        
        # ==================== GOOGLE CLOUD TAB ====================
        google_tab = QWidget()
        google_layout = QFormLayout(google_tab)
        
        self.google_api_key = QLineEdit()
        self.google_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.google_api_key.setPlaceholderText("Enter your Google Cloud API key")
        google_layout.addRow("API Key:", self.google_api_key)
        
        # Show/hide API key
        self.show_key_check = QCheckBox("Show API key")
        self.show_key_check.toggled.connect(
            lambda checked: self.google_api_key.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        google_layout.addRow("", self.show_key_check)
        
        # Enhanced model
        self.enhanced_model = QCheckBox("Use enhanced model (better accuracy, costs more)")
        self.enhanced_model.setChecked(False)
        google_layout.addRow("", self.enhanced_model)
        
        # Pricing info
        pricing_label = QLabel(
            "💰 Pricing: $0.016 per minute (standard)\n"
            "💰 Enhanced: $0.024 per minute"
        )
        pricing_label.setStyleSheet("color: #FF9800; font-size: 8pt;")
        google_layout.addRow("", pricing_label)
        
        # Documentation link
        docs_link = QLabel('<a href="https://cloud.google.com/speech-to-text/pricing">📄 Pricing Details</a>')
        docs_link.setOpenExternalLinks(True)
        google_layout.addRow("", docs_link)
        
        tabs.addTab(google_tab, "Google Cloud")
        
        # ==================== TTS TAB ====================
        tts_tab = QWidget()
        tts_layout = QFormLayout(tts_tab)
        
        self.tts_rate = QSpinBox()
        self.tts_rate.setRange(-50, 50)
        self.tts_rate.setValue(0)
        self.tts_rate.setSuffix("%")
        self.tts_rate.setToolTip("Adjust speech speed (-50% to +50%)")
        tts_layout.addRow("Speech rate:", self.tts_rate)
        
        self.tts_volume = QSpinBox()
        self.tts_volume.setRange(-50, 50)
        self.tts_volume.setValue(0)
        self.tts_volume.setSuffix("%")
        self.tts_volume.setToolTip("Adjust volume (-50% to +50%)")
        tts_layout.addRow("Volume:", self.tts_volume)
        
        self.auto_play = QCheckBox("Auto-play TTS after generation")
        self.auto_play.setChecked(True)
        tts_layout.addRow("", self.auto_play)
        
        # Default voice per language
        self.default_voice_en = QComboBox()
        self.default_voice_en.addItems(["Jenny (Female)", "Guy (Male)", "Sonia (UK)", "Natasha (AU)"])
        tts_layout.addRow("English voice:", self.default_voice_en)
        
        self.default_voice_zh = QComboBox()
        self.default_voice_zh.addItems(["Xiaoxiao (Female)", "Yunxi (Male)", "HsiaoChen (Taiwan)"])
        tts_layout.addRow("Chinese voice:", self.default_voice_zh)
        
        self.default_voice_km = QComboBox()
        self.default_voice_km.addItems(["Sreymom (Female)"])
        tts_layout.addRow("Khmer voice:", self.default_voice_km)
        
        tabs.addTab(tts_tab, "Text-to-Speech")
        
        # ==================== ABOUT TAB ====================
        about_tab = QWidget()
        about_layout = QVBoxLayout(about_tab)
        
        about_text = QTextEdit()
        about_text.setReadOnly(True)
        about_text.setHtml("""
        <h2>AI Speech-to-Text Pro</h2>
        <p><b>Version 3.0.0</b></p>
        <p>A powerful speech-to-text application with:</p>
        <ul>
        <li>🎯 <b>faster-whisper</b> - Local processing (75MB-3GB models)</li>
        <li>🗣️ <b>Vosk</b> - Offline, free (40MB-1.3GB models)</li>
        <li>☁️ <b>Google Cloud</b> - High accuracy cloud API</li>
        </ul>
        <p><b>Features:</b></p>
        <ul>
        <li>⚡ Speed Mode - 2-5x faster processing</li>
        <li>🌐 Multi-language support including Khmer</li>
        <li>🔊 Advanced TTS with voice customization</li>
        <li>📊 Real-time progress with percentage</li>
        <li>📋 Copy functionality for text</li>
        <li>📄 Export to TXT, SRT, VTT, JSON</li>
        </ul>
        <p><b>Cache location:</b><br>
        📁 {cache_dir}</p>
        <p><b>Session location:</b><br>
        📁 {session_dir}</p>
        <p>Built with PyQt6 and ❤️</p>
        """.format(
            cache_dir=Path.home() / '.cache' / 'huggingface' / 'hub',
            session_dir=Path.home() / '.ai_transcriber' / 'sessions'
        ))
        about_layout.addWidget(about_text)
        
        tabs.addTab(about_tab, "About")
        
        layout.addWidget(tabs)
        
        # Dialog buttons
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("💾 Save Settings")
        self.save_btn.clicked.connect(self.accept)
        self.save_btn.setMinimumHeight(35)
        button_layout.addWidget(self.save_btn)
        
        self.cancel_btn = QPushButton("❌ Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setMinimumHeight(35)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Load existing settings if parent is MainWindow
        if parent and hasattr(parent, 'settings'):
            self.load_from_main_window()
    
    def update_model_info(self, model_size: str):
        """Update model info based on selected size."""
        info = {
            "tiny": "📦 75MB - Fastest, good for testing",
            "base": "📦 150MB - Balanced speed/accuracy",
            "small": "📦 300MB - Good accuracy",
            "medium": "📦 1.5GB - Better accuracy, slower",
            "large": "📦 3GB - Best accuracy, slowest"
        }
        self.model_info.setText(f"ℹ️ {info.get(model_size, '')}")
        self.model_status_label.setText(self.check_model_status(model_size))
    
    def check_model_status(self, model_size: str = None) -> str:
        """Check if model is already downloaded."""
        if model_size is None:
            model_size = self.model_size.currentText()
        
        cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
        if cache_dir.exists():
            # Look for model files
            model_files = list(cache_dir.glob(f"*{model_size}*"))
            if model_files:
                return "✅ Model cached"
        return "⬇️ Not downloaded"
    
    def download_model(self):
        """Download the selected model manually with better progress tracking."""
        model_size = self.model_size.currentText()
        
        # Check if already downloaded
        cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
        model_files = list(cache_dir.glob(f"*{model_size}*")) if cache_dir.exists() else []
        
        if model_files:
            reply = QMessageBox.question(
                self,
                "Model Already Exists",
                f"Model '{model_size}' appears to be already downloaded.\n\nDownload again?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        # Create a temporary thread to download with better progress
        class DownloadThread(QThread):
            progress = pyqtSignal(str, int)  # message, percentage
            finished = pyqtSignal(bool, str)
            
            def __init__(self, model_size):
                super().__init__()
                self.model_size = model_size
            
            def run(self):
                try:
                    self.progress.emit(f"Starting download of {self.model_size} model...", 10)
                    self.progress.emit("This may take a few minutes...", 20)
                    
                    # This will trigger the download
                    model = faster_whisper.WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type="int8",
                        download_root=None
                    )
                    
                    self.progress.emit("Verifying download...", 90)
                    self.progress.emit("Model ready!", 100)
                    self.finished.emit(True, f"{self.model_size} model downloaded successfully!")
                    
                except Exception as e:
                    self.finished.emit(False, str(e))
        
        self.download_btn.setEnabled(False)
        self.download_btn.setText("⏳ Starting download...")
        
        # Create progress dialog
        self.progress_dialog = QProgressDialog("Downloading model...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle("Model Download")
        self.progress_dialog.setMinimumWidth(400)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.setAutoReset(True)
        
        self.download_thread = DownloadThread(model_size)
        self.download_thread.progress.connect(self.update_download_progress)
        self.download_thread.finished.connect(self.on_download_finished)
        self.download_thread.start()
    
    def update_download_progress(self, message: str, percentage: int):
        """Update download progress dialog."""
        self.progress_dialog.setLabelText(message)
        self.progress_dialog.setValue(percentage)
        self.download_btn.setText(message)
    
    def on_download_finished(self, success: bool, message: str):
        """Handle download completion."""
        self.progress_dialog.close()
        self.download_btn.setEnabled(True)
        
        if success:
            self.download_btn.setText("✅ Download Complete")
            self.model_status_label.setText("✅ Model cached")
            QMessageBox.information(
                self, 
                "Download Successful", 
                f"✅ {message}\n\nYou can now use faster-whisper for transcription."
            )
        else:
            self.download_btn.setText("⬇️ Download Model Now")
            self.model_status_label.setText("❌ Download failed")
            
            error_msg = str(message)
            if "Connection" in error_msg or "timeout" in error_msg.lower():
                QMessageBox.critical(
                    self,
                    "Download Failed",
                    f"❌ Network error. Please check your internet connection.\n\n{error_msg}"
                )
            elif "disk" in error_msg.lower() or "space" in error_msg.lower():
                QMessageBox.critical(
                    self,
                    "Download Failed",
                    f"❌ Not enough disk space. Please free up space and try again.\n\n{error_msg}"
                )
            else:
                QMessageBox.critical(
                    self,
                    "Download Failed",
                    f"❌ Failed to download model:\n\n{error_msg}"
                )
    
    def load_from_main_window(self):
        """Load settings from main window."""
        if not self.main_window:
            return
        
        # Load theme
        theme = self.main_window.settings.value('theme', 'Dark')
        index = self.theme_combo.findText(theme)
        if index >= 0:
            self.theme_combo.setCurrentIndex(index)
        
        # Load auto-save
        auto_save = self.main_window.settings.value('auto_save', True, type=bool)
        self.auto_save.setChecked(auto_save)
        
        # Load save format
        save_format = self.main_window.settings.value('save_format', 'JSON')
        index = self.save_format.findText(save_format)
        if index >= 0:
            self.save_format.setCurrentIndex(index)
        
        # Load default engine
        default_engine = self.main_window.settings.value('default_engine', 'faster-whisper')
        engine_map = {
            'faster-whisper': 0,
            'Vosk': 1,
            'google': 2,
            'Google Cloud': 2
        }
        self.default_engine.setCurrentIndex(engine_map.get(default_engine, 0))
        
        # Load model size
        model_size = self.main_window.settings.value('model_size', 'tiny')
        index = self.model_size.findText(model_size)
        if index >= 0:
            self.model_size.setCurrentIndex(index)
        
        # Load VAD filter
        vad_filter = self.main_window.settings.value('vad_filter', False, type=bool)
        self.vad_filter.setChecked(vad_filter)
        
        # Load speed mode
        speed_mode = self.main_window.settings.value('speed_mode', True, type=bool)
        self.speed_mode.setChecked(speed_mode)
        
        # Load compute type
        compute_type = self.main_window.settings.value('compute_type', 'int8')
        index = self.compute_type.findText(compute_type)
        if index >= 0:
            self.compute_type.setCurrentIndex(index)
        
        # Load Google API key
        google_api_key = self.main_window.settings.value('google_api_key', '')
        self.google_api_key.setText(google_api_key)
        
        # Load enhanced model
        enhanced = self.main_window.settings.value('enhanced_model', False, type=bool)
        self.enhanced_model.setChecked(enhanced)
        
        # Load TTS settings
        tts_rate = self.main_window.settings.value('tts_rate', 0, type=int)
        self.tts_rate.setValue(tts_rate)
        
        tts_volume = self.main_window.settings.value('tts_volume', 0, type=int)
        self.tts_volume.setValue(tts_volume)
        
        auto_play = self.main_window.settings.value('auto_play_tts', True, type=bool)
        self.auto_play.setChecked(auto_play)
    
    def get_settings(self) -> Dict[str, Any]:
        """Get all settings as dictionary."""
        return {
            # General
            'theme': self.theme_combo.currentText(),
            'auto_save': self.auto_save.isChecked(),
            'save_format': self.save_format.currentText(),
            
            # Transcription
            'default_engine': self.default_engine.currentText(),
            'model_size': self.model_size.currentText(),
            'vad_filter': self.vad_filter.isChecked(),
            'speed_mode': self.speed_mode.isChecked(),
            'compute_type': self.compute_type.currentText(),
            
            # Google Cloud
            'google_api_key': self.google_api_key.text(),
            'enhanced_model': self.enhanced_model.isChecked(),
            
            # TTS
            'tts_rate': self.tts_rate.value(),
            'tts_volume': self.tts_volume.value(),
            'auto_play_tts': self.auto_play.isChecked(),
            
            # Voice defaults (you can expand these)
            'default_voice_en': self.default_voice_en.currentText(),
            'default_voice_zh': self.default_voice_zh.currentText(),
            'default_voice_km': self.default_voice_km.currentText(),
        }
class EngineInfoDialog(QDialog):
    """Dialog showing engine capabilities for selected language."""
    
    def __init__(self, parent=None, language="km"):
        super().__init__(parent)
        self.setWindowTitle(f"Engine Support for {language.upper()}")
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml(self.get_engine_info_html(language))
        layout.addWidget(info_text)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
    
    def get_engine_info_html(self, language):
        html = f"<h2>Engine Support for {language.upper()}</h2>"
        html += "<table border='1' cellpadding='5'>"
        html += "<tr><th>Engine</th><th>Supported</th><th>Notes</th><th>Cost</th></tr>"
        
        for engine, info in ENGINE_SUPPORT.items():
            supported = "✅" if language in info["supported"] else "❌"
            html += f"<tr>"
            html += f"<td>{engine}</td>"
            html += f"<td>{supported}</td>"
            html += f"<td>{info['notes']}</td>"
            html += f"<td>{info['cost']}</td>"
            html += f"</tr>"
        
        html += "</table>"
        return html

class TTSDialog(QDialog):
    """Text-to-Speech settings dialog."""
    
    def __init__(self, parent=None, text="", language="km"):
        super().__init__(parent)
        self.setWindowTitle("Text-to-Speech Settings")
        self.setMinimumWidth(500)
        self.main_window = parent  # This stores the parent (MainWindow)
        
        layout = QVBoxLayout(self)
        
        # Text preview
        text_group = QGroupBox("📝 Text to Speak")
        text_layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(text)
        self.text_edit.setMaximumHeight(100)
        text_layout.addWidget(self.text_edit)
        text_group.setLayout(text_layout)
        layout.addWidget(text_group)
        
        # Voice settings
        settings_group = QGroupBox("🎤 Voice Settings")
        settings_layout = QFormLayout()
        
        # Language selection
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("English", "en")
        self.lang_combo.addItem("Chinese", "zh")
        self.lang_combo.addItem("Khmer", "km")
        self.lang_combo.addItem("Japanese", "ja")
        self.lang_combo.addItem("Korean", "ko")
        self.lang_combo.addItem("French", "fr")
        self.lang_combo.addItem("German", "de")
        self.lang_combo.addItem("Spanish", "es")
        
        # Set current language
        index = self.lang_combo.findData(language)
        if index >= 0:
            self.lang_combo.setCurrentIndex(index)
        
        settings_layout.addRow("Language:", self.lang_combo)
        
        # Voice selection (will update based on language)
        self.voice_combo = QComboBox()
        self.update_voices()
        self.lang_combo.currentIndexChanged.connect(self.update_voices)
        settings_layout.addRow("Voice:", self.voice_combo)
        
        # Speed control
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(-50, 50)
        self.speed_spin.setValue(0)
        self.speed_spin.setSuffix("%")
        settings_layout.addRow("Speed:", self.speed_spin)
        
        # Volume control
        self.volume_spin = QSpinBox()
        self.volume_spin.setRange(-50, 50)
        self.volume_spin.setValue(0)
        self.volume_spin.setSuffix("%")
        settings_layout.addRow("Volume:", self.volume_spin)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.preview_btn = QPushButton("🔊 Preview")
        self.preview_btn.clicked.connect(self.preview_tts)
        button_layout.addWidget(self.preview_btn)
        
        self.speak_btn = QPushButton("🗣️ Speak")
        self.speak_btn.setDefault(True)
        self.speak_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.speak_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
    
    def update_voices(self):
        """Update voice list based on selected language."""
        lang = self.lang_combo.currentData()
        
        voices = {
            "en": ["en-US-JennyNeural", "en-US-GuyNeural", "en-GB-SoniaNeural", "en-AU-NatashaNeural"],
            "zh": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural", "zh-TW-HsiaoChenNeural"],
            "km": ["km-KH-SreymomNeural"],
            "ja": ["ja-JP-NanamiNeural", "ja-JP-KeitaNeural"],
            "ko": ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural"],
            "fr": ["fr-FR-DeniseNeural", "fr-FR-HenriNeural"],
            "de": ["de-DE-KatjaNeural", "de-DE-ConradNeural"],
            "es": ["es-ES-ElviraNeural", "es-ES-AlvaroNeural"]
        }
        
        self.voice_combo.clear()
        for voice in voices.get(lang, voices["en"]):
            self.voice_combo.addItem(voice, voice)
    
    def preview_tts(self):
        """Preview TTS with current settings."""
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No Text", "Please enter some text to preview.")
            return
        
        lang = self.lang_combo.currentData()
        voice = self.voice_combo.currentData()
        speed = self.speed_spin.value()
        volume = self.volume_spin.value()
        
        # Check if main_window exists and has the required methods
        if self.main_window and hasattr(self.main_window, 'status_label'):
            self.main_window.status_label.setText("🔊 Generating preview...")
            
            # Create and start TTS thread
            self.preview_thread = TTSThread(text, lang, voice, speed, volume)
            self.preview_thread.audio_ready.connect(self.main_window.play_tts)
            self.preview_thread.error.connect(self.main_window.on_tts_error)
            self.preview_thread.start()
        else:
            # Fallback if no main window reference
            QMessageBox.information(self, "Preview", 
                f"Preview would generate:\nLanguage: {lang}\nVoice: {voice}\nText: {text[:50]}...")
    
    def get_settings(self):
        """Get TTS settings."""
        return {
            'text': self.text_edit.toPlainText().strip(),
            'language': self.lang_combo.currentData(),
            'voice': self.voice_combo.currentData(),
            'speed': self.speed_spin.value(),
            'volume': self.volume_spin.value()
        }
# ==================== MAIN WINDOW ====================

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        print("=" * 60)
        print("DEBUG: Application starting...")
        print("=" * 60)
        
        try:
            print("DEBUG: Calling super().__init__()")
            super().__init__()
            print("DEBUG: super().__init__() completed")
            
            print("DEBUG: Initializing variables")
            self.model = None
            self.current_session = None
            self.player = None
            self.audio_output = None
            self.current_thread = None
            self.segments: List[TranscriptionSegment] = []
            self.temp_files = []
            self.settings = QSettings('AITranscriber', 'SpeechToTextPro')
            self.session_manager = SessionManager()
            print(f"DEBUG: Session manager created")
            print("DEBUG: Variables initialized")
            
            print("DEBUG: Calling setup_ui()")
            self.setup_ui()
            print("DEBUG: setup_ui() completed")
            
            print("DEBUG: Calling setup_menu()")
            self.setup_menu()
            print("DEBUG: setup_menu() completed")
            
            print("DEBUG: Calling setup_toolbar()")
            self.setup_toolbar()
            print("DEBUG: setup_toolbar() completed")
            
            print("DEBUG: Calling setup_statusbar()")
            self.setup_statusbar()
            print("DEBUG: setup_statusbar() completed")
            
            print("DEBUG: Calling update_engine_ui()")
            self.update_engine_ui()
            print("DEBUG: update_engine_ui() completed")
            
            print("DEBUG: Calling load_settings()")
            self.load_settings()
            print("DEBUG: load_settings() completed")
            
            print("DEBUG: Setting up QTimer for load_model")
            self.load_timer = QTimer()
            self.load_timer.singleShot(500, self.load_model)
            print(f"DEBUG: QTimer created")
            
            print("DEBUG: Calling apply_theme()")
            self.apply_theme()
            print("DEBUG: apply_theme() completed")
            
            print("DEBUG: Calling setup_connections()")
            self.setup_connections()
            print("DEBUG: setup_connections() completed")
            
            # Setup copy shortcuts
            self.setup_copy_shortcuts()
            
            print("=" * 60)
            print("DEBUG: Application initialized successfully!")
            print("=" * 60)
            
        except Exception as e:
            print("\n" + "!" * 60)
            print(f"CRASH IN __init__: {e}")
            print("!" * 60)
            traceback.print_exc()
            print("!" * 60)
            input("Press Enter to exit...")
            sys.exit(1)
    
    def setup_ui(self):
        """Create the main UI."""
        self.setWindowTitle("AI Speech-to-Text Pro")
        self.setMinimumSize(1200, 800)
        
        # Create status labels
        self.model_status = QLabel("Model: Not loaded")
        self.lang_status = QLabel("Language: --")
        self.segment_status = QLabel("Segments: 0")
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Top toolbar area
        toolbar_layout = QHBoxLayout()
        
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("padding: 5px; background-color: #2d2d3a; border-radius: 3px;")
        toolbar_layout.addWidget(self.file_label, 1)
        toolbar_layout.addStretch()
        
       # Top toolbar area - WITHOUT theme button
        toolbar_layout = QHBoxLayout()

        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("padding: 5px; background-color: #2d2d3a; border-radius: 3px;")
        toolbar_layout.addWidget(self.file_label, 1)
        toolbar_layout.addStretch()

        # Theme button removed - will be handled by unified app

        main_layout.addLayout(toolbar_layout)
        
        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        
        # File selection group
        file_group = QGroupBox("📁 Audio File")
        file_layout = QVBoxLayout()
        
        self.select_btn = QPushButton("📂 Select Audio File")
        self.select_btn.setMinimumHeight(40)
        self.select_btn.clicked.connect(self.select_file)
        file_layout.addWidget(self.select_btn)
        
        self.file_details = QLabel("")
        self.file_details.setWordWrap(True)
        self.file_details.setStyleSheet("color: #b0b0b0; font-size: 9pt;")
        file_layout.addWidget(self.file_details)
        
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)
        
        # Engine selection group
        engine_group = QGroupBox("⚙️ Transcription Engine")
        engine_layout = QVBoxLayout()
        
        self.engine_group = QButtonGroup()
        
        self.local_radio = QRadioButton("🎯 faster-whisper (Local)")
        self.local_radio.setChecked(True)
        self.engine_group.addButton(self.local_radio)
        engine_layout.addWidget(self.local_radio)
        
        self.vosk_radio = QRadioButton("🗣️ Vosk (Offline, Free)")
        self.engine_group.addButton(self.vosk_radio)
        engine_layout.addWidget(self.vosk_radio)
        
        self.google_radio = QRadioButton("☁️ Google Cloud API")
        self.engine_group.addButton(self.google_radio)
        engine_layout.addWidget(self.google_radio)
        
        self.api_key_layout = QHBoxLayout()
        self.api_key_layout.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Enter Google Cloud API key")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_layout.addWidget(self.api_key_input)
        engine_layout.addLayout(self.api_key_layout)
        
        # Engine info button
        self.engine_info_btn = QPushButton("ℹ️ Engine Support")
        self.engine_info_btn.clicked.connect(self.show_engine_info)
        engine_layout.addWidget(self.engine_info_btn)
        
        engine_group.setLayout(engine_layout)
        left_layout.addWidget(engine_group)
        
        # Settings group
        settings_group = QGroupBox("🔧 Settings")
        settings_layout = QFormLayout()
        
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("Auto Detect", "auto")
        self.lang_combo.addItem("English", "en")
        self.lang_combo.addItem("Chinese", "zh")
        self.lang_combo.addItem("Khmer", "km")
        self.lang_combo.addItem("Japanese", "ja")
        self.lang_combo.addItem("Korean", "ko")
        self.lang_combo.addItem("French", "fr")
        self.lang_combo.addItem("German", "de")
        self.lang_combo.addItem("Spanish", "es")
        self.lang_combo.currentIndexChanged.connect(self.on_language_changed)
        settings_layout.addRow("Language:", self.lang_combo)
        
        self.vad_check = QCheckBox("Enable VAD filter")
        self.vad_check.setChecked(False)  # Disabled by default for speed
        settings_layout.addRow("", self.vad_check)
        
        self.enhanced_check = QCheckBox("Use enhanced model")
        self.enhanced_check.setChecked(False)  # Disabled by default for speed
        settings_layout.addRow("", self.enhanced_check)
        
        # SPEED OPTIMIZATION: Add speed mode indicator
        self.speed_indicator = QLabel("⚡ Speed Mode: Enabled")
        self.speed_indicator.setStyleSheet("color: #4CAF50; font-weight: bold;")
        settings_layout.addRow("", self.speed_indicator)
        
        settings_group.setLayout(settings_layout)
        left_layout.addWidget(settings_group)
        
        # Transcribe button
        self.transcribe_btn = QPushButton("🎤 START TRANSCRIPTION")
        self.transcribe_btn.setMinimumHeight(50)
        self.transcribe_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.transcribe_btn.clicked.connect(self.transcribe)
        self.transcribe_btn.setEnabled(False)
        left_layout.addWidget(self.transcribe_btn)
        
        # Progress group
        progress_group = QGroupBox("📊 Progress")
        progress_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        progress_layout.addWidget(self.status_label)
        
        self.percentage_label = QLabel("0%")
        self.percentage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.percentage_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #4CAF50;")
        self.percentage_label.setVisible(False)
        progress_layout.addWidget(self.percentage_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFormat("%p%")
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        left_layout.addWidget(progress_group)
        
        splitter.addWidget(left_panel)
        
        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.waveform = WaveformWidget()
        right_layout.addWidget(self.waveform)
        
        self.tab_widget = QTabWidget()
        
        self.list_widget = QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setFont(QFont("Consolas", 10))
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.tab_widget.addTab(self.list_widget, "List View")
        
        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        self.text_view.setFont(QFont("Consolas", 10))
        self.tab_widget.addTab(self.text_view, "Text View")
        
        right_layout.addWidget(self.tab_widget)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶️ Play")
        self.play_btn.clicked.connect(self.play_pause)
        self.play_btn.setEnabled(False)
        action_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.clicked.connect(self.stop_playback)
        self.stop_btn.setEnabled(False)
        action_layout.addWidget(self.stop_btn)
        
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.VLine)
        separator1.setFrameShadow(QFrame.Shadow.Sunken)
        separator1.setMaximumWidth(2)
        action_layout.addWidget(separator1)
        
        self.speak_btn = QPushButton("🔊 Speak Selected")
        self.speak_btn.clicked.connect(self.speak_selected)
        self.speak_btn.setEnabled(False)
        action_layout.addWidget(self.speak_btn)
        
        # Copy buttons
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.Shape.VLine)
        separator3.setFrameShadow(QFrame.Shadow.Sunken)
        separator3.setMaximumWidth(2)
        action_layout.addWidget(separator3)

        self.copy_btn = QPushButton("📋 Copy")
        self.copy_btn.clicked.connect(self.copy_selected)
        self.copy_btn.setEnabled(False)
        action_layout.addWidget(self.copy_btn)

        self.copy_all_btn = QPushButton("📋 Copy All")
        self.copy_all_btn.clicked.connect(self.copy_all)
        self.copy_all_btn.setEnabled(False)
        action_layout.addWidget(self.copy_all_btn)

        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.VLine)
        separator2.setFrameShadow(QFrame.Shadow.Sunken)
        separator2.setMaximumWidth(2)
        action_layout.addWidget(separator2)
        
        self.export_btn = QPushButton("📄 Export")
        self.export_btn.clicked.connect(self.export_menu)
        self.export_btn.setEnabled(False)
        action_layout.addWidget(self.export_btn)
        
        self.save_btn = QPushButton("💾 Save Session")
        self.save_btn.clicked.connect(self.save_session)
        self.save_btn.setEnabled(False)
        action_layout.addWidget(self.save_btn)
        
        self.clear_btn = QPushButton("🗑️ Clear")
        self.clear_btn.clicked.connect(self.clear_results)
        action_layout.addWidget(self.clear_btn)
        
        action_layout.addStretch()
        right_layout.addLayout(action_layout)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])
        main_layout.addWidget(splitter)
        
        # Connect radio buttons
        self.local_radio.toggled.connect(self.update_engine_ui)
        self.vosk_radio.toggled.connect(self.update_engine_ui)
        self.google_radio.toggled.connect(self.update_engine_ui)
        
        self.update_engine_ui()
    
    def setup_menu(self):
        """Create menu bar."""
        print("  DEBUG: setup_menu() started")
        try:
            menubar = self.menuBar()
            
            file_menu = menubar.addMenu("&File")
            
            open_action = QAction("📂 Open Audio File", self)
            open_action.setShortcut(QKeySequence.StandardKey.Open)
            open_action.triggered.connect(self.select_file)
            file_menu.addAction(open_action)
            
            file_menu.addSeparator()
            
            save_action = QAction("💾 Save Session", self)
            save_action.setShortcut(QKeySequence.StandardKey.Save)
            save_action.triggered.connect(self.save_session)
            file_menu.addAction(save_action)
            
            load_action = QAction("📂 Load Session", self)
            load_action.setShortcut(QKeySequence.StandardKey.Open)
            load_action.triggered.connect(self.load_session)
            file_menu.addAction(load_action)
            
            file_menu.addSeparator()
            
            export_menu = file_menu.addMenu("📄 Export As")
            
            txt_action = QAction("TXT File", self)
            txt_action.triggered.connect(lambda: self.export_format("txt"))
            export_menu.addAction(txt_action)
            
            srt_action = QAction("SRT File", self)
            srt_action.triggered.connect(lambda: self.export_format("srt"))
            export_menu.addAction(srt_action)
            
            vtt_action = QAction("VTT File", self)
            vtt_action.triggered.connect(lambda: self.export_format("vtt"))
            export_menu.addAction(vtt_action)
            
            json_action = QAction("JSON File", self)
            json_action.triggered.connect(lambda: self.export_format("json"))
            export_menu.addAction(json_action)
            
            file_menu.addSeparator()
            
            exit_action = QAction("❌ Exit", self)
            exit_action.setShortcut(QKeySequence.StandardKey.Quit)
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)
            
            edit_menu = menubar.addMenu("&Edit")
            
            copy_action = QAction("📋 Copy", self)
            copy_action.setShortcut(QKeySequence.StandardKey.Copy)
            copy_action.triggered.connect(self.copy_selected)
            edit_menu.addAction(copy_action)
            
            select_all_action = QAction("🔍 Select All", self)
            select_all_action.setShortcut(QKeySequence.StandardKey.SelectAll)
            select_all_action.triggered.connect(self.select_all)
            edit_menu.addAction(select_all_action)
            
            edit_menu.addSeparator()
            
            clear_action = QAction("🗑️ Clear", self)
            clear_action.triggered.connect(self.clear_results)
            edit_menu.addAction(clear_action)
            
            tools_menu = menubar.addMenu("&Tools")
            
            settings_action = QAction("⚙️ Settings", self)
            settings_action.setShortcut(QKeySequence.StandardKey.Preferences)
            settings_action.triggered.connect(self.show_settings)
            tools_menu.addAction(settings_action)
            
            tools_menu.addSeparator()
            
            # Add Engine Info to Tools menu
            engine_info_action = QAction("ℹ️ Engine Support Info", self)
            engine_info_action.triggered.connect(self.show_engine_info)
            tools_menu.addAction(engine_info_action)
            
            # Add Quick TTS to Tools menu
            quick_tts_action = QAction("⚡ Quick TTS", self)
            quick_tts_action.setShortcut("Ctrl+Shift+T")
            quick_tts_action.triggered.connect(self.quick_tts)
            tools_menu.addAction(quick_tts_action)
            
            help_menu = menubar.addMenu("&Help")
            
            about_action = QAction("ℹ️ About", self)
            about_action.triggered.connect(self.show_about)
            help_menu.addAction(about_action)
            
            print("  DEBUG: setup_menu() completed")
            
        except Exception as e:
            print(f"  ERROR in setup_menu: {e}")
            traceback.print_exc()
            raise
    
    def setup_toolbar(self):
        """Create toolbar."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        open_action = QAction("📂 Open", self)
        open_action.triggered.connect(self.select_file)
        toolbar.addAction(open_action)
        
        save_action = QAction("💾 Save", self)
        save_action.triggered.connect(self.save_session)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        play_action = QAction("▶️ Play", self)
        play_action.triggered.connect(self.play_pause)
        toolbar.addAction(play_action)
        
        stop_action = QAction("⏹️ Stop", self)
        stop_action.triggered.connect(self.stop_playback)
        toolbar.addAction(stop_action)
        
        toolbar.addSeparator()
        
        speak_action = QAction("🔊 Speak", self)
        speak_action.triggered.connect(self.speak_selected)
        toolbar.addAction(speak_action)
        
        # Quick TTS button
        quick_tts_action = QAction("⚡ Quick TTS", self)
        quick_tts_action.triggered.connect(self.quick_tts)
        toolbar.addAction(quick_tts_action)
    
    def setup_statusbar(self):
        """Create status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.addPermanentWidget(self.model_status)
        self.statusbar.addPermanentWidget(self.lang_status)
        self.statusbar.addPermanentWidget(self.segment_status)
    
    def setup_connections(self):
        """Connect signals and slots."""
        if self.player:
            self.player.positionChanged.connect(self.on_position_changed)
            self.player.playbackStateChanged.connect(self.on_playback_state_changed)
    
    # ==================== CORE FUNCTIONALITY ====================
    
    def load_model(self):
        """Load faster-whisper model in background."""
        print("🔥🔥🔥 load_model STARTED 🔥🔥🔥")
        try:
            self.status_label.setText("⏳ Loading faster-whisper model...")
            self.model_status.setText("Model: Loading...")
            
            # SPEED OPTIMIZATION: Use tiny model by default for speed
            model_size = self.settings.value('model_size', 'tiny')
            print(f"🔥 Model size: {model_size}")
            
            print("🔥 Creating ModelLoaderThread")
            self.model_loader = ModelLoaderThread(model_size)
            
            print("🔥 Connecting signals")
            self.model_loader.progress.connect(self.on_load_progress)
            self.model_loader.model_loaded.connect(self.on_model_loaded)
            self.model_loader.error.connect(self.on_model_error)
            
            print("🔥 Starting thread")
            self.model_loader.start()
            print("🔥 Thread started")
            
        except Exception as e:
            print(f"🔥🔥🔥 ERROR in load_model: {e}")
            traceback.print_exc()

    def on_load_progress(self, message):
        """Handle model loading progress."""
        print(f"📢 Progress: {message}")
        self.status_label.setText(message)

    def on_model_loaded(self, model):
        """Handle successful model loading."""
        print("✅✅✅ MODEL LOADED SUCCESSFULLY! ✅✅✅")
        self.model = model
        self.status_label.setText("✅ Model ready (Speed Mode)")
        self.model_status.setText("Model: Ready (faster-whisper tiny)")
        self.transcribe_btn.setEnabled(True)

    def on_model_error(self, error_msg):
        """Handle model loading error."""
        self.status_label.setText("⚠️ Model failed to load")
        self.model_status.setText("Model: Failed")
        
        # Create a more helpful error dialog
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Model Error")
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setText("faster-whisper model could not be loaded.")
        
        # Parse the error message for better guidance
        if "NETWORK ERROR" in error_msg:
            msg_box.setInformativeText(
                "Network connection issue detected.\n\n"
                "The model needs to be downloaded first (75-300MB).\n"
                "Please check your internet connection and try again."
            )
        elif "MODEL NOT FOUND" in error_msg:
            msg_box.setInformativeText(
                "The requested model size is not available.\n\n"
                "Please go to Settings → Transcription and select a different model size:\n"
                "• tiny (fastest, 75MB)\n"
                "• base (balanced, 150MB)\n"
                "• small (accurate, 300MB)"
            )
        elif "DISK SPACE" in error_msg:
            msg_box.setInformativeText(
                "Not enough disk space to download the model.\n\n"
                "Please free up space and try again."
            )
        elif "PERMISSION" in error_msg:
            msg_box.setInformativeText(
                "Permission denied when trying to save the model.\n\n"
                f"Please check permissions for:\n{Path.home() / '.cache'}"
            )
        else:
            msg_box.setInformativeText(
                f"Error details: {error_msg}\n\n"
                "You can still use Vosk or Google Cloud API for transcription.\n\n"
                "To fix this issue:\n"
                "1. Go to Tools → Settings → Transcription\n"
                "2. Click 'Download Model Now'\n"
                "3. Or try a different model size"
            )
        
        msg_box.setDetailedText(error_msg)
        
        # Add helpful buttons
        settings_btn = msg_box.addButton("Open Settings", QMessageBox.ButtonRole.ActionRole)
        close_btn = msg_box.addButton("Close", QMessageBox.ButtonRole.RejectRole)
        
        msg_box.exec()
        
        if msg_box.clickedButton() == settings_btn:
            self.show_settings()
        
        self.transcribe_btn.setEnabled(True)
    
    def select_file(self):
        """Select audio file with details."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            self.settings.value('last_directory', ''),
            "Audio Files (*.mp3 *.wav *.m4a *.flac *.ogg *.aac);;All Files (*.*)"
        )
        
        if file_path:
            self.audio_file = file_path
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            
            if file_size < 1024:
                size_str = f"{file_size} B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"
            
            self.file_label.setText(file_name)
            self.file_details.setText(f"Size: {size_str}")
            
            self.settings.setValue('last_directory', os.path.dirname(file_path))
            self.clear_results()
    
    def on_language_changed(self, index):
        """Handle language selection change."""
        language = self.lang_combo.currentData()
        if language == "km":
            self.show_khmer_recommendation()
    
    def show_khmer_recommendation(self):
        """Show recommendation for Khmer language."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Khmer Language Recommendation")
        msg.setText("For Khmer transcription, we recommend:")
        msg.setInformativeText(
            "☁️ Google Cloud API: Best accuracy (costs $0.016/min)\n"
            "🎯 faster-whisper: Free but may default to Chinese\n"
            "🗣️ Vosk: Not available for Khmer\n\n"
            "Your current engine selection will be checked when you start transcription."
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
    
    def show_engine_info(self):
        """Show engine support information."""
        language = self.lang_combo.currentData()
        dialog = EngineInfoDialog(self, language)
        dialog.exec()
    
    def transcribe(self):
        """Start transcription with selected engine."""
        if not hasattr(self, 'audio_file') or not self.audio_file:
            QMessageBox.warning(self, "No File", "Please select an audio file first.")
            return
        
        language = self.lang_combo.currentData()
        
        # Check if selected engine supports the language
        if self.vosk_radio.isChecked() and language == "km":
            reply = QMessageBox.question(
                self,
                "Vosk Khmer Support",
                "Vosk does NOT support Khmer language.\n\n"
                "Would you like to switch to Google Cloud API for better Khmer support?\n"
                "(Click No to continue with faster-whisper)",
                QMessageBox.StandardButton.Yes | 
                QMessageBox.StandardButton.No | 
                QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.google_radio.setChecked(True)
            elif reply == QMessageBox.StandardButton.No:
                self.local_radio.setChecked(True)
            else:
                self.reset_ui()
                return
        
        self.transcribe_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.clear_results()
        
        self.status_label.setText("⏳ Initializing transcription...")
        self.progress_bar.setVisible(True)
        self.percentage_label.setVisible(True)
        self.percentage_label.setText("0%")
        self.progress_bar.setRange(0, 0)
        
        if self.google_radio.isChecked():
            self.transcribe_with_google(language)
        elif self.vosk_radio.isChecked():
            self.transcribe_with_vosk(language)
        else:
            self.transcribe_with_faster_whisper(language)
    
    def transcribe_with_faster_whisper(self, language):
        """Optimized transcription using faster-whisper with language forcing."""
        if self.model is None:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Model Not Loaded")
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setText("faster-whisper model is not loaded.")
            msg_box.setInformativeText(
                "This usually happens on first run as the model needs to be downloaded.\n\n"
                "Options:\n"
                "1. Go to Settings → Transcription → Click 'Download Model Now'\n"
                "2. Use Vosk or Google Cloud API instead\n"
                "3. Restart the application to try again"
            )
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()
            self.reset_ui()
            return
        
        # Special handling for Khmer
        if language == "km":
            reply = QMessageBox.question(
                self,
                "Khmer Language Notice",
                "Whisper models have limited Khmer training data.\n"
                "The transcription may default to Chinese if the audio is unclear.\n\n"
                "Options:\n"
                "• Click 'Yes' to continue with faster-whisper (may show Chinese)\n"
                "• Click 'No' to switch to Google Cloud API (better for Khmer)\n"
                "• Click 'Cancel' to abort",
                QMessageBox.StandardButton.Yes | 
                QMessageBox.StandardButton.No | 
                QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.No:
                self.google_radio.setChecked(True)
                self.transcribe_with_google(language)
                return
            elif reply == QMessageBox.StandardButton.Cancel:
                self.reset_ui()
                return
        
        # Force the language parameter strictly
        whisper_language = language if language != "auto" else None
        
        if language == "km":
            whisper_language = "km"
            self.status_label.setText("⏳ Forcing Khmer language (speed optimized)...")
        
        self.current_thread = FasterWhisperThread(
            self.model,
            self.audio_file,
            whisper_language,
            task="transcribe",
            vad=self.vad_check.isChecked()
        )
        
        self.current_thread.language_detected.connect(
            lambda detected_lang, confidence: self.check_khmer_detection(detected_lang, confidence, language)
        )
        
        self.current_thread.segment_ready.connect(self.add_segment)
        self.current_thread.progress.connect(self.update_progress)
        self.current_thread.finished.connect(self.on_transcription_finished)
        self.current_thread.error.connect(self.on_transcription_error)
        self.current_thread.start()
        
        self.status_label.setText("⚡ Speed mode active - processing...")

    def check_khmer_detection(self, detected_lang: str, confidence: float, requested_lang: str):
        """Check if the detected language matches what we requested."""
        self.lang_status.setText(f"Language: {detected_lang} ({confidence:.1%})")
        
        # If we requested Khmer but got something else
        if requested_lang == "km" and detected_lang != "km":
            print(f"⚠️ Warning: Requested Khmer but model detected {detected_lang} with {confidence:.1%} confidence")
            
            # Show non-blocking warning
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Language Mismatch")
            msg_box.setText(f"The model detected '{detected_lang}' instead of Khmer.")
            msg_box.setInformativeText(
                f"This often happens with Khmer audio due to limited training data.\n"
                f"The transcription may be in {detected_lang}.\n\n"
                f"Try using Google Cloud API for better Khmer accuracy."
            )
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()

    def transcribe_with_google(self, language):
        """Transcribe using Google Cloud API."""
        api_key = self.api_key_input.text().strip()
        
        if not api_key:
            QMessageBox.warning(self, "API Key Required",
                               "Please enter your Google Cloud API key.")
            self.reset_ui()
            return
        
        if not GOOGLE_CLOUD_AVAILABLE:
            QMessageBox.critical(self, "Missing Library",
                               "google-cloud-speech not installed.\n"
                               "Install: pip install google-cloud-speech")
            self.reset_ui()
            return
        
        self.current_thread = GoogleSpeechThread(
            api_key,
            self.audio_file,
            language,
            use_enhanced=self.enhanced_check.isChecked()  # Disable enhanced for speed
        )
        self.current_thread.segment_ready.connect(self.add_segment)
        self.current_thread.progress.connect(self.update_progress)
        self.current_thread.language_detected.connect(self.on_language_detected)
        self.current_thread.finished.connect(self.on_transcription_finished)
        self.current_thread.error.connect(self.on_transcription_error)
        self.current_thread.start()
    
    def transcribe_with_vosk(self, language):
        """Transcribe using Vosk offline engine."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, 'models')
        
        # Vosk supported languages (Khmer NOT supported)
        vosk_models = {
            "en": "vosk-model-small-en-us-0.15",
            "zh": "vosk-model-small-cn-0.22",
            "fr": "vosk-model-small-fr-0.22",
            "de": "vosk-model-small-de-0.15",
            "es": "vosk-model-small-es-0.42",
            "ru": "vosk-model-small-ru-0.22",
            "ja": "vosk-model-small-ja-0.22",
            "ko": "vosk-model-ko-0.22"
        }
        
        if language not in vosk_models:
            QMessageBox.critical(
                self,
                "Language Not Supported",
                f"Vosk does not support {language} language.\n\n"
                f"Supported languages: {', '.join(vosk_models.keys())}\n\n"
                f"Please use faster-whisper or Google Cloud API for {language}."
            )
            self.reset_ui()
            return
        
        model_name = vosk_models[language]
        model_path = os.path.join(models_dir, model_name)
        
        if not os.path.exists(model_path):
            QMessageBox.critical(
                self, 
                "Model Not Found",
                f"Vosk model not found at:\n{model_path}\n\n"
                f"Please download from:\nhttps://alphacephei.com/vosk/models\n\n"
                f"Required model: {model_name}.zip\n\n"
                f"After downloading, extract to:\n{model_path}"
            )
            self.reset_ui()
            return
        
        self.status_label.setText(f"⏳ Loading Vosk {language} model...")
        
        self.current_thread = VoskThread(
            model_path,
            self.audio_file,
            language
        )
        self.current_thread.segment_ready.connect(self.add_segment)
        self.current_thread.progress.connect(self.update_progress)
        self.current_thread.language_detected.connect(self.on_language_detected)
        self.current_thread.finished.connect(self.on_transcription_finished)
        self.current_thread.error.connect(self.on_transcription_error)
        self.current_thread.start()
    
    def add_segment(self, segment: TranscriptionSegment):
        """Add transcribed segment to UI."""
        self.segments.append(segment)
        
        time_str = self.format_time(segment.start)
        speaker_prefix = f"[{segment.speaker}] " if segment.speaker else ""
        
        # Add language indicator if different from requested
        lang_indicator = ""
        if segment.detected_language and segment.detected_language != self.lang_combo.currentData():
            lang_indicator = f" [{segment.detected_language}]"
        
        display_text = f"{speaker_prefix}[{time_str}]{lang_indicator} {segment.text}"
        
        item = QListWidgetItem(display_text)
        item.setData(Qt.ItemDataRole.UserRole, segment)
        
        if segment.speaker:
            colors = {
                "SPEAKER_0": QColor(109, 93, 252),
                "SPEAKER_1": QColor(76, 175, 80)
            }
            item.setForeground(colors.get(segment.speaker, Qt.GlobalColor.white))
        elif segment.detected_language and segment.detected_language != self.lang_combo.currentData():
            # Highlight mismatched language
            item.setForeground(QColor(255, 165, 0))  # Orange
        
        self.list_widget.addItem(item)
        self.text_view.append(display_text)
        self.list_widget.scrollToBottom()
        self.text_view.moveCursor(QTextCursor.MoveOperation.End)
        
        self.segment_status.setText(f"Segments: {len(self.segments)}")
    
    def update_progress(self, current: int, total: int, current_time: float):
        """Update progress bar with percentage."""
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
            self.progress_bar.setFormat(f"%p% ({current}/{total})")
            
            self.percentage_label.setVisible(True)
            self.percentage_label.setText(f"{percentage}%")
            
            if current_time > 0:
                time_str = self.format_time(current_time)
                self.status_label.setText(f"⏳ Transcribing... {percentage}% at {time_str}")
            else:
                self.status_label.setText(f"⏳ Transcribing... {percentage}%")
    
    def on_language_detected(self, language: str, confidence: float):
        """Handle detected language."""
        self.lang_status.setText(f"Language: {language} ({confidence:.1%})")
    
    def on_transcription_finished(self, session: TranscriptionSession):
        """Handle successful transcription."""
        self.current_session = session
        
        self.status_label.setText(f"✅ Complete - {len(session.segments)} segments (100%)")
        self.progress_bar.setVisible(False)
        self.percentage_label.setVisible(False)
        
        self.waveform.set_segments(session.segments)
        
        # Enable all buttons
        self.play_btn.setEnabled(True)
        self.speak_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.copy_btn.setEnabled(True)
        self.copy_all_btn.setEnabled(True)
        
        self.reset_ui()
    
    def on_transcription_error(self, error_msg: str):
        """Handle transcription error."""
        self.status_label.setText("❌ Transcription failed")
        self.progress_bar.setVisible(False)
        self.percentage_label.setVisible(False)
        self.reset_ui()
        QMessageBox.critical(self, "Error", error_msg)
    
    def reset_ui(self):
        """Reset UI after transcription."""
        self.transcribe_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
    
    # ==================== PLAYBACK ====================
    
    def on_item_clicked(self, item: QListWidgetItem):
        """Handle item click with context menu."""
        segment = item.data(Qt.ItemDataRole.UserRole)
        if segment and self.audio_file:
            menu = QMenu(self)
            
            play_from_here = menu.addAction("▶️ Play from here to end")
            play_segment_only = menu.addAction("⏵ Play this segment only")
            play_full = menu.addAction("🎵 Play full audio")
            
            action = menu.exec(QCursor.pos())
            
            if action == play_from_here:
                self.play_from_segment(segment.start)
            elif action == play_segment_only:
                self.play_from_segment(segment.start, segment.end)
            elif action == play_full:
                self.play_from_segment(0)
    
    def on_item_double_clicked(self, item: QListWidgetItem):
        """Handle double click - play segment."""
        segment = item.data(Qt.ItemDataRole.UserRole)
        if segment and self.audio_file:
            self.play_from_segment(segment.start, segment.end)
    
    def play_from_segment(self, start: float = 0, end: float = None):
        """Play audio from specific start time."""
        if not self.audio_file:
            return
        
        try:
            if not self.player:
                self.player = QMediaPlayer()
                self.audio_output = QAudioOutput()
                self.player.setAudioOutput(self.audio_output)
            
            if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.player.stop()
                try:
                    self.player.positionChanged.disconnect()
                except:
                    pass
            
            self.player.setSource(QUrl.fromLocalFile(self.audio_file))
            self.player.setPosition(int(start * 1000))
            
            if end is not None:
                def check_position(position):
                    if position >= int(end * 1000):
                        self.player.stop()
                        self.play_btn.setText("▶️ Play")
                        self.status_label.setText("✅ Playback finished")
                        try:
                            self.player.positionChanged.disconnect(check_position)
                        except:
                            pass
                
                self.player.positionChanged.connect(check_position)
                self.status_label.setText(f"▶️ Playing segment: {self.format_time(start)} - {self.format_time(end)}")
            else:
                self.status_label.setText(f"▶️ Playing from {self.format_time(start)} to end")
            
            self.player.play()
            self.play_btn.setText("⏸️ Pause")
            self.waveform.set_position(start)
            
        except Exception as e:
            QMessageBox.warning(self, "Playback Error", str(e))

    def play_pause(self):
        """Toggle play/pause."""
        if not self.player:
            if self.audio_file:
                self.play_from_segment(0)
            return
        
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.play_btn.setText("▶️ Play")
        else:
            self.player.play()
            self.play_btn.setText("⏸️ Pause")

    def stop_playback(self):
        """Stop playback."""
        if self.player:
            self.player.stop()
            self.play_btn.setText("▶️ Play")
            self.waveform.set_position(0)
            self.status_label.setText("⏹️ Playback stopped")
            try:
                self.player.positionChanged.disconnect()
            except:
                pass
    
    def on_position_changed(self, position: int):
        """Handle position change."""
        if self.current_session:
            self.waveform.set_position(position / 1000.0)
    
    def on_playback_state_changed(self, state: QMediaPlayer.PlaybackState):
        """Handle playback state change."""
        self.play_btn.setEnabled(state != QMediaPlayer.PlaybackState.StoppedState)
        self.stop_btn.setEnabled(state != QMediaPlayer.PlaybackState.StoppedState)
    
    # ==================== TTS ====================
    
    def speak_selected(self):
        """Generate speech for selected line with TTS dialog."""
        item = self.list_widget.currentItem()
        if not item:
            QMessageBox.information(self, "No Selection", "Please select a line.")
            return
        
        segment = item.data(Qt.ItemDataRole.UserRole)
        if not segment:
            return
        
        # Get language from current session or selection
        if self.current_session:
            lang = self.current_session.language
        else:
            lang = self.lang_combo.currentData()
            if lang == "auto":
                lang = "en"
        
        # Pass 'self' as parent to TTSDialog
        dialog = TTSDialog(self, segment.text, lang)  # 'self' is MainWindow
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings = dialog.get_settings()
            
            self.status_label.setText("🔊 Generating speech...")
            
            self.tts_thread = TTSThread(
                settings['text'],
                settings['language'],
                settings['voice'],
                settings['speed'],
                settings['volume']
            )
            self.tts_thread.audio_ready.connect(self.play_tts)
            self.tts_thread.progress.connect(self.status_label.setText)
            self.tts_thread.error.connect(self.on_tts_error)
            self.tts_thread.start()

    def quick_tts(self):
        """Open TTS dialog with empty text for quick typing."""
        dialog = TTSDialog(self, "", "en")
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings = dialog.get_settings()
            
            self.status_label.setText("🔊 Generating speech...")
            
            self.tts_thread = TTSThread(
                settings['text'],
                settings['language'],
                settings['voice'],
                settings['speed'],
                settings['volume']
            )
            self.tts_thread.audio_ready.connect(self.play_tts)
            self.tts_thread.progress.connect(self.status_label.setText)
            self.tts_thread.error.connect(self.on_tts_error)
            self.tts_thread.start()
    
    def play_tts(self, audio_path: str):
        """Play TTS audio."""
        self.temp_files.append(audio_path)
        
        if not self.player:
            self.player = QMediaPlayer()
            self.audio_output = QAudioOutput()
            self.player.setAudioOutput(self.audio_output)
        
        self.player.setSource(QUrl.fromLocalFile(audio_path))
        self.player.play()
        self.status_label.setText("▶️ Playing TTS...")
        
        def cleanup(status):
            if status == QMediaPlayer.MediaStatus.EndOfMedia:
                try:
                    os.remove(audio_path)
                    if audio_path in self.temp_files:
                        self.temp_files.remove(audio_path)
                except:
                    pass
        
        self.player.mediaStatusChanged.connect(cleanup)
    
    def on_tts_error(self, error_msg: str):
        """Handle TTS error."""
        self.status_label.setText("❌ TTS failed")
        QMessageBox.critical(self, "TTS Error", error_msg)
    
    # ==================== COPY FUNCTIONS ====================
    
    def copy_selected(self):
        """Copy selected text to clipboard."""
        item = self.list_widget.currentItem()
        if item:
            clipboard = QApplication.clipboard()
            clipboard.setText(item.text())
            self.status_label.setText("📋 Copied to clipboard")
    
    def copy_all(self):
        """Copy all transcription to clipboard."""
        if not self.segments:
            QMessageBox.information(self, "No Data", "No transcription to copy.")
            return
        
        all_text = "\n".join([item.text() for item in self.list_widget.findItems("*", Qt.MatchFlag.MatchWildcard)])
        clipboard = QApplication.clipboard()
        clipboard.setText(all_text)
        self.status_label.setText("📋 Copied all to clipboard")
    
    def setup_copy_shortcuts(self):
        """Setup keyboard shortcuts for copying."""
        copy_action = QAction(self)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.triggered.connect(self.copy_selected)
        self.addAction(copy_action)
        
        copy_all_action = QAction(self)
        copy_all_action.setShortcut("Ctrl+Shift+C")
        copy_all_action.triggered.connect(self.copy_all)
        self.addAction(copy_all_action)
    
    # ==================== EXPORT ====================
    
    def export_menu(self):
        """Show export menu."""
        if not self.segments:
            QMessageBox.information(self, "No Data", "No transcription to export.")
            return
        
        menu = QMenu(self)
        
        txt_action = menu.addAction("📄 TXT File")
        txt_action.triggered.connect(lambda: self.export_format("txt"))
        
        srt_action = menu.addAction("🎬 SRT File (Subtitles)")
        srt_action.triggered.connect(lambda: self.export_format("srt"))
        
        vtt_action = menu.addAction("🌐 VTT File (Web Subtitles)")
        vtt_action.triggered.connect(lambda: self.export_format("vtt"))
        
        json_action = menu.addAction("📊 JSON File (Data)")
        json_action.triggered.connect(lambda: self.export_format("json"))
        
        menu.exec(QCursor.pos())
    
    def export_format(self, format_type: str):
        """Export in specified format."""
        if not self.segments:
            QMessageBox.warning(self, "No Data", "No transcription to export.")
            return
        
        # Get file path
        filters = {
            "txt": "Text Files (*.txt)",
            "srt": "Subtitle Files (*.srt)",
            "vtt": "WebVTT Files (*.vtt)",
            "json": "JSON Files (*.json)"
        }
        
        # Suggest a filename based on the audio file
        default_name = ""
        if hasattr(self, 'audio_file') and self.audio_file:
            base = os.path.splitext(os.path.basename(self.audio_file))[0]
            default_name = f"{base}_transcript.{format_type}"
        else:
            default_name = f"transcript.{format_type}"
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export as {format_type.upper()}",
            default_name,
            filters.get(format_type, "All Files (*.*)")
        )
        
        if not path:
            return
        
        try:
            if format_type == "txt":
                self.export_txt(path)
            elif format_type == "srt":
                self.export_srt(path)
            elif format_type == "vtt":
                self.export_vtt(path)
            elif format_type == "json":
                self.export_json(path)
            
            self.status_label.setText(f"✅ Exported to {os.path.basename(path)}")
            QMessageBox.information(self, "Export Successful", f"File saved to:\n{path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save file:\n{str(e)}")
            traceback.print_exc()
    
    def export_txt(self, path: str):
        """Export as plain text with timestamps."""
        with open(path, 'w', encoding='utf-8') as f:
            for segment in self.segments:
                time_str = self.format_time(segment.start)
                f.write(f"[{time_str}] {segment.text}\n")
    
    def export_srt(self, path: str):
        """Export as SRT subtitles."""
        with open(path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(self.segments, 1):
                start = self.format_srt_time(segment.start)
                end = self.format_srt_time(segment.end)
                f.write(f"{i}\n{start} --> {end}\n{segment.text}\n\n")
    
    def export_vtt(self, path: str):
        """Export as WebVTT."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for i, segment in enumerate(self.segments, 1):
                start = self.format_vtt_time(segment.start)
                end = self.format_vtt_time(segment.end)
                f.write(f"{i}\n{start} --> {end}\n{segment.text}\n\n")
    
    def export_json(self, path: str):
        """Export as JSON."""
        data = {
            'file': self.audio_file if hasattr(self, 'audio_file') else None,
            'segments': [s.to_dict() for s in self.segments],
            'stats': {
                'total_segments': len(self.segments),
                'duration': self.segments[-1].end if self.segments else 0,
                'language': self.detected_lang if hasattr(self, 'detected_lang') else None
            }
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    # ==================== SESSION MANAGEMENT ====================
    
    def save_session(self):
        """Save current session."""
        if not self.current_session:
            return
        
        default_name = f"{self.current_session.file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Session",
            default_name,
            "JSON Files (*.json)"
        )
        
        if path:
            self.session_manager.save_session(self.current_session, Path(path).stem)
            self.status_label.setText(f"💾 Session saved")
    
    def load_session(self):
        """Load saved session."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Session",
            str(self.session_manager.sessions_dir),
            "JSON Files (*.json)"
        )
        
        if path:
            session = self.session_manager.load_session(path)
            if session:
                self.current_session = session
                self.segments = session.segments
                
                self.list_widget.clear()
                self.text_view.clear()
                
                for segment in session.segments:
                    self.add_segment(segment)
                
                self.audio_file = session.file_path
                self.file_label.setText(session.file_name)
                self.lang_status.setText(f"Language: {session.language}")
                
                self.status_label.setText(f"📂 Loaded session: {session.file_name}")
    
    # ==================== UTILITIES ====================
    
    @staticmethod
    def format_time(seconds: float) -> str:
        if seconds < 0:
            seconds = 0
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    @staticmethod
    def format_srt_time(seconds: float) -> str:
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    @staticmethod
    def format_vtt_time(seconds: float) -> str:
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    def update_engine_ui(self):
        """Update UI based on selected engine."""
        use_google = self.google_radio.isChecked()
        use_vosk = self.vosk_radio.isChecked()
        
        self.api_key_input.setEnabled(use_google)
        self.enhanced_check.setEnabled(use_google)
        
        if use_google:
            self.model_status.setText("Model: Google Cloud API")
        elif use_vosk:
            self.model_status.setText("Model: Vosk (Offline)")
        else:
            if self.model:
                self.model_status.setText("Model: Ready (faster-whisper tiny)")
            else:
                self.model_status.setText("Model: Loading...")
    
    def select_all(self):
        """Select all items."""
        self.list_widget.selectAll()
    
    def clear_results(self):
        """Clear all results."""
        self.list_widget.clear()
        self.text_view.clear()
        self.segments.clear()
        self.current_session = None
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.speak_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)
        self.copy_all_btn.setEnabled(False)
        self.segment_status.setText("Segments: 0")
        self.waveform.set_segments([])
    
    # ==================== THEME & SETTINGS ====================
    
    def apply_theme(self, dark: bool = True):
        """Apply dark or light theme."""
        if dark:
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
            dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
            dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
            dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
            dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(109, 93, 252))
            dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
            QApplication.setPalette(dark_palette)
        else:
            QApplication.setPalette(self.style().standardPalette())
    
    def toggle_theme(self):
        """Toggle between dark and light theme."""
        current = self.settings.value('theme', 'dark')
        new_theme = 'light' if current == 'dark' else 'dark'
        self.apply_theme(new_theme == 'dark')
        self.settings.setValue('theme', new_theme)
    
    def load_settings(self):
        """Load saved settings."""
        self.api_key_input.setText(self.settings.value('google_api_key', ''))
        
        last_lang = self.settings.value('last_language', 'auto')
        index = self.lang_combo.findData(last_lang)
        if index >= 0:
            self.lang_combo.setCurrentIndex(index)
        
        engine = self.settings.value('default_engine', 'faster-whisper')
        if engine == 'google':
            self.google_radio.setChecked(True)
        elif engine == 'vosk':
            self.vosk_radio.setChecked(True)
        
        vad = self.settings.value('vad_filter', False, type=bool)
        self.vad_check.setChecked(vad)
        
        enhanced = self.settings.value('enhanced_model', False, type=bool)
        self.enhanced_check.setChecked(enhanced)
        
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
    
    def save_settings(self):
        """Save current settings."""
        self.settings.setValue('google_api_key', self.api_key_input.text())
        self.settings.setValue('last_language', self.lang_combo.currentData())
        
        if self.google_radio.isChecked():
            self.settings.setValue('default_engine', 'google')
        elif self.vosk_radio.isChecked():
            self.settings.setValue('default_engine', 'vosk')
        else:
            self.settings.setValue('default_engine', 'faster-whisper')
        
        self.settings.setValue('vad_filter', self.vad_check.isChecked())
        self.settings.setValue('enhanced_model', self.enhanced_check.isChecked())
        self.settings.setValue('geometry', self.saveGeometry())
    
    def show_settings(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings = dialog.get_settings()
            
            self.settings.setValue('theme', settings['theme'])
            self.settings.setValue('auto_save', settings['auto_save'])
            self.settings.setValue('save_format', settings['save_format'])
            self.settings.setValue('model_size', settings['model_size'])
            self.settings.setValue('vad_filter', settings['vad_filter'])
            self.settings.setValue('tts_rate', settings['tts_rate'])
            self.settings.setValue('tts_volume', settings['tts_volume'])
            
            self.apply_theme(settings['theme'] == 'Dark')
            
            # Update speed indicator
            if settings['speed_mode']:
                self.speed_indicator.setText("⚡ Speed Mode: Enabled")
                self.speed_indicator.setStyleSheet("color: #4CAF50; font-weight: bold;")
            else:
                self.speed_indicator.setText("🐢 Quality Mode: Enabled")
                self.speed_indicator.setStyleSheet("color: #FF9800; font-weight: bold;")
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About AI Speech-to-Text Pro",
            f"<h2>AI Speech-to-Text Pro</h2>"
            f"<p>Version 3.0.0 (Speed Optimized)</p>"
            f"<p>A powerful speech-to-text application with:</p>"
            f"<ul>"
            f"<li>🎯 faster-whisper (Local) - Now 2-5x faster with tiny model</li>"
            f"<li>🗣️ Vosk (Offline, Free) - Optimized chunking</li>"
            f"<li>☁️ Google Cloud API - Configurable enhanced models</li>"
            f"<li>⚡ Speed Mode - Process 1-hour audio in 1-5 minutes</li>"
            f"<li>Multi-language support including Khmer</li>"
            f"<li>Advanced TTS capabilities with voice customization</li>"
            f"<li>Copy functionality for selected text and entire transcript</li>"
            f"<li>Multiple export formats</li>"
            f"</ul>"
            f"<p>Built with PyQt6 and ❤️</p>"
        )
    
    def showEvent(self, event):
        print(f"🪟 SHOW EVENT - Window visible: {self.isVisible()}")
        super().showEvent(event)

    def hideEvent(self, event):
        print(f"🪟 HIDE EVENT - Window visible: {self.isVisible()}")
        super().hideEvent(event)
    
    def closeEvent(self, event):
        """Handle application close."""
        print("🔥🔥🔥 MainWindow.closeEvent called 🔥🔥🔥")
        self.save_settings()
        
        if self.current_thread and self.current_thread.isRunning():
            self.current_thread.stop()
            self.current_thread.wait(2000)
        
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        
        event.accept()

# ==================== MAIN ====================

def main():
    """Application entry point."""
    try:
        print("DEBUG: Creating QApplication")
        app = QApplication(sys.argv)
        app.setApplicationName("AI Speech-to-Text Pro")
        app.setOrganizationName("AITranscriber")
        
        print("DEBUG: Setting application style")
        app.setStyle(QStyleFactory.create('Fusion'))
        
        print("DEBUG: Creating MainWindow")
        window = MainWindow()
        
        print("DEBUG: Showing window")
        window.show()
        window.raise_()
        window.activateWindow()
        
        print("DEBUG: Window should be visible now")
        print("DEBUG: Entering event loop...")
        
        app.window = window
        
        exit_code = app.exec()
        print(f"DEBUG: Event loop exited with code: {exit_code}")
        
        import time
        time.sleep(0.5)
        sys.exit(exit_code)

    except Exception as e:
        print("=" * 60)
        print(f"FATAL ERROR IN MAIN: {e}")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()