"""
Khmer TTS Studio – High Quality Edition
- Optimized for reliability and audio quality
- Better error handling and recovery
- Improved parallel processing
- Professional audio mastering
- Internet speed aware export with progress tracking
"""
import random
import sys
import os
import asyncio
import tempfile
import subprocess
import re
import logging
import shutil
import time
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from enum import Enum
import traceback

# PyQt6
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QFileDialog,
    QMessageBox, QProgressBar, QComboBox, QSpinBox, QDoubleSpinBox,
    QLineEdit, QSplitter, QGroupBox, QCheckBox, QFrame, QStatusBar,
    QSlider, QTextEdit, QTabWidget, QToolBar, QMenu, QMenuBar, QDialog, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QTimer, QMutex, QWaitCondition, QMutexLocker, QSize
from PyQt6.QtGui import QFont, QColor, QIcon, QPixmap, QAction, QKeySequence
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

# TTS
import edge_tts

# Audio processing with fallbacks
try:
    from pydub import AudioSegment
    from pydub.effects import normalize
    from pydub.effects import compress_dynamic_range
    from pydub.silence import detect_nonsilent
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not installed. Audio processing limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('khmer_tts.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== CONSTANTS & CONFIG ====================

class VoiceGender(Enum):
    FEMALE = "female"
    MALE = "male"
    ALTERNATING = "alternating"
    
class Language(Enum):
    KHMER = "khmer"
    ENGLISH = "english"
    CHINESE = "chinese"
    
VOICE_MAP = {
    # Khmer voices
    (Language.KHMER, VoiceGender.FEMALE): "km-KH-SreymomNeural",
    (Language.KHMER, VoiceGender.MALE): "km-KH-ThearithNeural",
    
    # English voices (US)
    (Language.ENGLISH, VoiceGender.FEMALE): "en-US-JennyNeural",
    (Language.ENGLISH, VoiceGender.MALE): "en-US-GuyNeural",
    
    # Chinese voices (Mandarin)
    (Language.CHINESE, VoiceGender.FEMALE): "zh-CN-XiaoxiaoNeural",
    (Language.CHINESE, VoiceGender.MALE): "zh-CN-YunxiNeural",
}

# High-quality audio settings
SAMPLE_RATE = 48000
CHANNELS = 2  # Stereo
BIT_DEPTH = 16
DEFAULT_BITRATE = "320k"  # Maximum MP3 quality
MIN_CROSSFADE = 0
MAX_CROSSFADE = 500
MIN_WORKERS = 1
MAX_WORKERS = 8  # Conservative max for stability
MIN_SPEED = 50
MAX_SPEED = 200
MIN_PADDING = 0.0
MAX_PADDING = 5.0
CACHE_DIR = Path.home() / ".khmer_tts_cache"
MAX_CACHE_SIZE_GB = 2

# Create cache directory
CACHE_DIR.mkdir(exist_ok=True)


# ==================== CACHE MANAGER ====================

class CacheManager:
    """Manages TTS cache for faster regeneration."""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
        self._cleanup_old_files()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _get_cache_key(self, text: str, voice: str, speed: float) -> str:
        """Generate cache key from text and voice."""
        content = f"{text}_{voice}_{speed}".encode('utf-8')
        return hashlib.sha256(content).hexdigest()
    
    def get(self, text: str, voice: str, speed: float) -> Optional[Path]:
        """Get cached audio file if exists."""
        key = self._get_cache_key(text, voice, speed)
        if key in self.metadata:
            cache_file = self.cache_dir / f"{key}.mp3"
            if cache_file.exists():
                # Update access time
                self.metadata[key]['last_accessed'] = time.time()
                self._save_metadata()
                return cache_file
            else:
                # Remove stale metadata
                del self.metadata[key]
                self._save_metadata()
        return None
    
    def put(self, text: str, voice: str, speed: float, audio_file: Path) -> Path:
        """Cache audio file."""
        key = self._get_cache_key(text, voice, speed)
        cache_file = self.cache_dir / f"{key}.mp3"
        
        # Copy file to cache
        shutil.copy2(audio_file, cache_file)
        
        # Update metadata
        self.metadata[key] = {
            'text': text[:100],
            'voice': voice,
            'speed': speed,
            'created': time.time(),
            'last_accessed': time.time(),
            'size': cache_file.stat().st_size
        }
        
        self._save_metadata()
        self._cleanup_old_files()
        return cache_file
    
    def _cleanup_old_files(self):
        """Remove old cache files to free space."""
        try:
            # Calculate total cache size
            total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.mp3"))
            max_size = MAX_CACHE_SIZE_GB * 1024 * 1024 * 1024
            
            if total_size > max_size:
                # Sort by last accessed
                files = []
                for key, meta in self.metadata.items():
                    cache_file = self.cache_dir / f"{key}.mp3"
                    if cache_file.exists():
                        files.append((meta['last_accessed'], key, cache_file))
                
                files.sort()  # Oldest first
                
                # Remove oldest until under limit
                for _, key, file in files:
                    if total_size <= max_size:
                        break
                    
                    size = file.stat().st_size
                    file.unlink(missing_ok=True)
                    del self.metadata[key]
                    total_size -= size
                
                self._save_metadata()
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")


# ==================== DATA MODELS ====================

@dataclass
class Segment:
    """Represents a single transcript segment with timestamp."""
    timestamp: str
    text: str
    seconds: float = 0.0
    audio_path: Optional[str] = None
    duration_ms: int = 0
    is_generated: bool = False
    has_error: bool = False
    error_message: str = ""
    gender: VoiceGender = VoiceGender.FEMALE
    language: Language = Language.KHMER 
    retry_count: int = 0
    processing_time: float = 0.0
    from_cache: bool = False 
    
    def __post_init__(self):
        self.seconds = self._parse_timestamp(self.timestamp)
        self.text = self.text.strip()
    
    def _parse_timestamp(self, ts: str) -> float:
        """Parse [HH:MM:SS] or [MM:SS] to seconds."""
        ts_clean = ts.strip('[]')
        parts = ts_clean.split(':')
        
        try:
            if len(parts) == 3:
                h, m, s = map(float, parts)
                return h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m, s = map(float, parts)
                return m * 60 + s
        except (ValueError, TypeError):
            logger.warning(f"Invalid timestamp format: {ts}")
            return 0.0
        return 0.0
    
    def time_str(self) -> str:
        """Format seconds as HH:MM:SS."""
        h = int(self.seconds // 3600)
        m = int((self.seconds % 3600) // 60)
        s = int(self.seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def preview_text(self, max_length: int = 50) -> str:
        """Get truncated preview text."""
        if len(self.text) <= max_length:
            return self.text
        return self.text[:max_length] + "..."
    
    @property
    def display_icon(self) -> str:
        """Get display icon based on state."""
        if self.has_error:
            return "❌"
        if self.is_generated:
            return "✅"
        return "⏳"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'text': self.text,
            'seconds': self.seconds,
            'gender': self.gender.value,
            'is_generated': self.is_generated,
            'language': self.language.value,
            'has_error': self.has_error,
            'error_message': self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Segment':
        """Create segment from dictionary."""
        segment = cls(data['timestamp'], data['text'])
        segment.seconds = data.get('seconds', 0.0)
        segment.gender = VoiceGender(data.get('gender', VoiceGender.FEMALE.value))
        segment.language = Language(data.get('language', Language.KHMER.value))
        segment.is_generated = data.get('is_generated', False)
        segment.has_error = data.get('has_error', False)
        segment.error_message = data.get('error_message', '')
        return segment


@dataclass
class GenerationStats:
    """Statistics for generation process."""
    total: int = 0
    completed: int = 0
    failed: int = 0
    cached: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    # Add these new fields
    current_speed: float = 0.0  # segments per minute
    estimated_time_remaining: float = 0.0  # in seconds
    last_update_time: Optional[float] = None
    last_completed_count: int = 0
    
    @property
    def elapsed(self) -> Optional[float]:
        if self.start_time:
            end = self.end_time or time.time()
            return end - self.start_time
        return None
    
    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return ((self.completed + self.cached) / self.total) * 100
    
    def update_speed(self):
        """Update current processing speed."""
        current_time = time.time()
        if self.last_update_time and self.last_completed_count > 0:
            time_diff = current_time - self.last_update_time
            if time_diff > 0:
                completed_diff = (self.completed + self.cached) - self.last_completed_count
                self.current_speed = (completed_diff / time_diff) * 60  # segments per minute
                
                # Calculate ETA
                remaining = self.total - (self.completed + self.cached)
                if self.current_speed > 0:
                    self.estimated_time_remaining = (remaining / self.current_speed) * 60  # in seconds
        
        self.last_update_time = current_time
        self.last_completed_count = self.completed + self.cached
    
    def reset(self):
        """Reset statistics."""
        self.total = 0
        self.completed = 0
        self.failed = 0
        self.cached = 0
        self.start_time = None
        self.end_time = None
        self.current_speed = 0.0
        self.estimated_time_remaining = 0.0
        self.last_update_time = None
        self.last_completed_count = 0

# ==================== KHMER TEXT PROCESSOR ====================

class KhmerTextProcessor:
    """Professional Khmer text preprocessing for TTS."""
    
    # Khmer numerals to words mapping
    NUMERALS = {
        '០': 'សូន្យ', '១': 'មួយ', '២': 'ពីរ', '៣': 'បី', '៤': 'បួន',
        '៥': 'ប្រាំ', '៦': 'ប្រាំមួយ', '៧': 'ប្រាំពីរ', '៨': 'ប្រាំបី', '៩': 'ប្រាំបួន'
    }
    
    # Khmer punctuation
    PUNCTUATION_MAP = {
        '។': '។ ',  # Khmer period
        'ៗ': ' ៗ',  # Repetition marker
        '៖': '៖ ',  # Khmer colon
        '?': '? ',
        '!': '! ',
        ',': ', ',
        ':': ': ',
        ';': '; '
    }
    
    # Common abbreviations and their expansions
    ABBREVIATIONS = {
        'លោក': 'លោក',
        'អ្នក': 'អ្នក',
        'លោកស្រី': 'លោកស្រី',
        'កញ្ញា': 'កញ្ញា',
    }
    
    @classmethod
    def process(cls, text: str) -> str:
        """
        Apply all Khmer text preprocessing.
        
        Args:
            text: Raw Khmer text
            
        Returns:
            Preprocessed text ready for TTS
        """
        if not text:
            return ""
        
        original = text
        
        # Replace numerals
        for num, word in cls.NUMERALS.items():
            text = text.replace(num, word)
        
        # Add spacing after punctuation
        for punct, replacement in cls.PUNCTUATION_MAP.items():
            text = text.replace(punct, replacement)
        
        # Handle multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Trim
        text = text.strip()
        
        if original != text:
            logger.debug(f"Text processed: {original[:50]} -> {text[:50]}")
        
        return text
    
    @staticmethod
    def validate(text: str) -> Tuple[bool, Optional[str]]:
        """
        Validate text for TTS generation.
        
        Returns:
            (is_valid, error_message)
        """
        if not text or not text.strip():
            return False, "Empty text"
        
        # Check for excessively long text (Edge TTS limit ~3000 chars)
        if len(text) > 3000:
            return False, f"Text too long ({len(text)} chars, max 3000)"
        
        # Check for invalid characters
        invalid_chars = []
        for char in text:
            if ord(char) > 0xFFFF:  # Very rare
                invalid_chars.append(char)
        
        if invalid_chars:
            return False, f"Contains invalid characters: {invalid_chars[:5]}"
        
        return True, None
    
    @staticmethod
    def split_long_text(text: str, max_length: int = 2000) -> List[str]:
        """Split long text into smaller chunks."""
        if len(text) <= max_length:
            return [text]
        
        # Try to split at sentence boundaries
        sentences = []
        current = []
        current_len = 0
        
        # Split by Khmer punctuation
        parts = re.split('([។?!])', text)
        
        for i in range(0, len(parts), 2):
            sentence = parts[i]
            if i + 1 < len(parts):
                sentence += parts[i + 1]
            
            if current_len + len(sentence) <= max_length:
                current.append(sentence)
                current_len += len(sentence)
            else:
                if current:
                    sentences.append(''.join(current))
                current = [sentence]
                current_len = len(sentence)
        
        if current:
            sentences.append(''.join(current))
        
        return sentences

# ==================== ENGLISH TEXT PROCESSOR ====================

class EnglishTextProcessor:
    """Professional English text preprocessing for TTS."""
    
    # Common abbreviations and their expansions
    ABBREVIATIONS = {
        "Mr.": "Mister",
        "Mrs.": "Misses",
        "Ms.": "Ms",
        "Dr.": "Doctor",
        "Prof.": "Professor",
        "Rev.": "Reverend",
        "Hon.": "Honorable",
        "St.": "Saint",
        "Ave.": "Avenue",
        "Blvd.": "Boulevard",
        "Rd.": "Road",
        "Dr": "Drive",
        "Ln": "Lane",
        "Ste": "Suite",
        "Apt": "Apartment",
        "e.g.": "for example",
        "i.e.": "that is",
        "etc.": "etcetera",
        "vs.": "versus",
        "Inc.": "Incorporated",
        "Co.": "Company",
        "Corp.": "Corporation",
        "Ltd.": "Limited",
    }
    
    @classmethod
    def process(cls, text: str) -> str:
        """
        Apply English text preprocessing for better TTS results.
        
        Args:
            text: Raw English text
            
        Returns:
            Preprocessed text ready for TTS
        """
        if not text:
            return ""
        
        original = text
        
        # Handle common abbreviations
        for abbr, expansion in cls.ABBREVIATIONS.items():
            # Word boundary check to avoid partial matches
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text)
        
        # Handle URLs and emails (simplify for TTS)
        text = re.sub(r'https?://\S+', 'website link', text)
        text = re.sub(r'www\.\S+', 'website link', text)
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', 'email address', text)
        
        # Handle phone numbers (simplify pattern)
        text = re.sub(r'\d{3}[-.]?\d{3}[-.]?\d{4}', 'phone number', text)
        
        # Ensure proper spacing after punctuation
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Add pauses for commas, semicolons, colons
        text = re.sub(r',', ', ', text)
        text = re.sub(r';', '; ', text)
        text = re.sub(r':', ': ', text)
        
        # Handle multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Trim
        text = text.strip()
        
        if original != text:
            logger.debug(f"English text processed: {original[:50]} -> {text[:50]}")
        
        return text
    
    @staticmethod
    def validate(text: str) -> Tuple[bool, Optional[str]]:
        """
        Validate English text for TTS generation.
        
        Returns:
            (is_valid, error_message)
        """
        if not text or not text.strip():
            return False, "Empty text"
        
        if len(text) > 3000:
            return False, f"Text too long ({len(text)} chars, max 3000)"
        
        # Check for invalid control characters
        invalid_chars = []
        for char in text:
            if ord(char) < 32 and char not in '\n\r\t':
                invalid_chars.append(char)
        
        if invalid_chars:
            return False, f"Contains invalid characters"
        
        return True, None
    
    @staticmethod
    def split_long_text(text: str, max_length: int = 2000) -> List[str]:
        """Split long English text into smaller chunks at sentence boundaries."""
        if len(text) <= max_length:
            return [text]
        
        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) <= max_length:
                current_chunk.append(sentence)
                current_length += len(sentence)
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


# ==================== CHINESE TEXT PROCESSOR ====================

class ChineseTextProcessor:
    """Professional Chinese text preprocessing for TTS."""
    
    # Chinese numerals to words (if needed)
    NUMERALS = {
        '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
        '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
    }
    
    # Common Chinese punctuation that needs special handling
    PUNCTUATION_MAP = {
        '。': '。 ',  # Chinese period
        '？': '？ ',  # Chinese question mark
        '！': '！ ',  # Chinese exclamation mark
        '，': '， ',  # Chinese comma
        '；': '； ',  # Chinese semicolon
        '：': '： ',  # Chinese colon
        '、': '、 ',  # Chinese enumeration comma
    }
    
    @classmethod
    def process(cls, text: str) -> str:
        """
        Apply Chinese text preprocessing for better TTS results.
        
        Args:
            text: Raw Chinese text
            
        Returns:
            Preprocessed text ready for TTS
        """
        if not text:
            return ""
        
        original = text
        
        # Add spacing after Chinese punctuation for better prosody
        for punct, replacement in cls.PUNCTUATION_MAP.items():
            text = text.replace(punct, replacement)
        
        # Handle English words mixed in Chinese text
        # Add spaces around English words for better TTS separation
        text = re.sub(r'([a-zA-Z]+)', r' \1 ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Trim
        text = text.strip()
        
        if original != text:
            logger.debug(f"Chinese text processed: {original[:50]} -> {text[:50]}")
        
        return text
    
    @staticmethod
    def validate(text: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Chinese text for TTS generation.
        
        Returns:
            (is_valid, error_message)
        """
        if not text or not text.strip():
            return False, "Empty text"
        
        if len(text) > 3000:
            return False, f"Text too long ({len(text)} chars, max 3000)"
        
        # Basic validation - allow CJK characters, common punctuation, and ASCII
        invalid_chars = []
        for char in text:
            # CJK Unified Ideographs range
            if not (
                '\u4e00' <= char <= '\u9fff' or  # CJK characters
                char in ' ，。？！；：‘’“”…—～·、（）《》【】' or  # Chinese punctuation
                char.isascii() or  # ASCII (English, numbers, basic punctuation)
                char.isspace()
            ):
                invalid_chars.append(char)
        
        if invalid_chars:
            return False, f"Contains non-Chinese characters: {invalid_chars[:5]}"
        
        return True, None
    
    @staticmethod
    def split_long_text(text: str, max_length: int = 2000) -> List[str]:
        """Split long Chinese text into smaller chunks at sentence boundaries."""
        if len(text) <= max_length:
            return [text]
        
        # Split by Chinese sentence endings
        sentences = re.split(r'([。？！])', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        i = 0
        while i < len(sentences):
            # Pair sentence with its punctuation if available
            if i + 1 < len(sentences) and sentences[i+1] in ['。', '？', '！']:
                sentence = sentences[i] + sentences[i+1]
                i += 2
            else:
                sentence = sentences[i]
                i += 1
            
            if current_length + len(sentence) <= max_length:
                current_chunk.append(sentence)
                current_length += len(sentence)
            else:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
        
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        return chunks


# ==================== MULTI-LANGUAGE PROCESSOR WRAPPER ====================

class TextProcessor:
    """Wrapper class to route text processing to the appropriate language processor."""
    
    @staticmethod
    def process(text: str, language: Language) -> str:
        """
        Process text based on the specified language.
        
        Args:
            text: Raw text
            language: Language enum value
            
        Returns:
            Preprocessed text ready for TTS
        """
        if language == Language.KHMER:
            return KhmerTextProcessor.process(text)
        elif language == Language.ENGLISH:
            return EnglishTextProcessor.process(text)
        elif language == Language.CHINESE:
            return ChineseTextProcessor.process(text)
        else:
            # Default fallback
            return text.strip()
    
    @staticmethod
    def validate(text: str, language: Language) -> Tuple[bool, Optional[str]]:
        """
        Validate text based on the specified language.
        
        Args:
            text: Raw text
            language: Language enum value
            
        Returns:
            (is_valid, error_message)
        """
        if language == Language.KHMER:
            return KhmerTextProcessor.validate(text)
        elif language == Language.ENGLISH:
            return EnglishTextProcessor.validate(text)
        elif language == Language.CHINESE:
            return ChineseTextProcessor.validate(text)
        else:
            # Default validation
            if not text or not text.strip():
                return False, "Empty text"
            if len(text) > 3000:
                return False, f"Text too long ({len(text)} chars, max 3000)"
            return True, None
    
    @staticmethod
    def split_long_text(text: str, language: Language, max_length: int = 2000) -> List[str]:
        """
        Split long text based on the specified language.
        
        Args:
            text: Raw text
            language: Language enum value
            max_length: Maximum chunk length
            
        Returns:
            List of text chunks
        """
        if language == Language.KHMER:
            return KhmerTextProcessor.split_long_text(text, max_length)
        elif language == Language.ENGLISH:
            return EnglishTextProcessor.split_long_text(text, max_length)
        elif language == Language.CHINESE:
            return ChineseTextProcessor.split_long_text(text, max_length)
        else:
            # Default: return as single chunk
            return [text]
        
# ==================== GENERATION WORKER ====================

class GenerationWorker:
    """Handles individual segment generation with error handling and retries."""
   
    def __init__(self, temp_dir: str, cache_manager: CacheManager, max_retries: int = 2):
        self.temp_dir = temp_dir
        self.cache = cache_manager
        self.max_retries = max_retries
    
    def generate(self, idx: int, segment: Segment, speed: float = 1.0) -> Tuple[int, bool, Optional[str], bool]:
        """
        Generate audio for a single segment.
        
        Returns:
            (index, success, error_message, from_cache)
        """
        start_time = time.time()
        
        try:
            # Validate text based on language - UPDATED
            is_valid, error = TextProcessor.validate(segment.text, segment.language)
            if not is_valid:
                return idx, False, error, False
            
            # Preprocess text based on language - UPDATED
            processed_text = TextProcessor.process(segment.text, segment.language)
            
            # Select voice
            voice = VOICE_MAP.get((segment.language, segment.gender))
            if not voice:
                # Fallback to Khmer if language not found
                voice = VOICE_MAP.get((Language.KHMER, segment.gender))
            
            # Check cache first
            cached_file = self.cache.get(processed_text, voice, speed)
            if cached_file:
                # Use cached file
                out_path = os.path.join(self.temp_dir, f"seg_{idx:06d}.mp3")
                shutil.copy2(cached_file, out_path)
                
                segment.audio_path = out_path
                segment.is_generated = True
                segment.processing_time = time.time() - start_time
                segment.from_cache = True  # Set cache flag
                
                # Get duration
                if PYDUB_AVAILABLE:
                    try:
                        audio = AudioSegment.from_mp3(out_path)
                        segment.duration_ms = len(audio)
                    except:
                        pass
                
                logger.debug(f"Cache hit for segment {idx}")
                return idx, True, None, True
            
            # Generate with retries
            for attempt in range(self.max_retries):
                try:
                    # Generate output path
                    out_path = os.path.join(self.temp_dir, f"seg_{idx:06d}.mp3")
                    
                    # Run Edge TTS
                    async def _generate():
                        communicate = edge_tts.Communicate(processed_text, voice)
                        await communicate.save(out_path)
                    
                    asyncio.run(_generate())
                    
                    # Verify generation
                    if not os.path.exists(out_path):
                        if attempt < self.max_retries - 1:
                            time.sleep(1 * (attempt + 1))  # Exponential backoff
                            continue
                        return idx, False, "Output file not created", False
                    
                    file_size = os.path.getsize(out_path)
                    if file_size == 0:
                        if attempt < self.max_retries - 1:
                            time.sleep(1 * (attempt + 1))
                            continue
                        return idx, False, "Generated file is empty", False
                    
                    # Cache the result
                    self.cache.put(processed_text, voice, speed, Path(out_path))
                    
                    segment.audio_path = out_path
                    segment.is_generated = True
                    segment.processing_time = time.time() - start_time
                    segment.from_cache = False  # Not from cache
                    
                    # Get duration if pydub available
                    if PYDUB_AVAILABLE:
                        try:
                            audio = AudioSegment.from_mp3(out_path)
                            segment.duration_ms = len(audio)
                        except Exception as e:
                            logger.warning(f"Could not get duration for segment {idx}: {e}")
                    
                    logger.debug(f"Generated segment {idx}: {out_path} ({file_size} bytes)")
                    return idx, True, None, False
                    
                except edge_tts.exceptions.NoAudioReceived:
                    if attempt < self.max_retries - 1:
                        # Exponential backoff with jitter
                        sleep_time = (2 ** attempt) + random.uniform(0, 1)
                        time.sleep(sleep_time)
                        continue
                    return idx, False, "No audio received from TTS service", False
                    
                except edge_tts.exceptions.UnknownResponse:
                    if attempt < self.max_retries - 1:
                        sleep_time = (2 ** attempt) + random.uniform(0, 1)
                        time.sleep(sleep_time)
                        continue
                    return idx, False, "Unknown response from TTS service", False
                    
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        sleep_time = (2 ** attempt) + random.uniform(0, 1)
                        time.sleep(sleep_time)
                        continue
                    logger.exception(f"Unexpected error generating segment {idx}")
                    return idx, False, str(e), False
            
            return idx, False, "Max retries exceeded", False
            
        except Exception as e:
            logger.exception(f"Fatal error in worker for segment {idx}")
            return idx, False, str(e), False

# ==================== GENERATOR THREAD ====================

class GeneratorThread(QThread):
    """Manages parallel segment generation with proper cancellation."""
    
    progress = pyqtSignal(int, int)  # current, total
    segment_completed = pyqtSignal(int)
    segment_failed = pyqtSignal(int, str)
    generation_finished = pyqtSignal()
    status_update = pyqtSignal(str)
    stats_updated = pyqtSignal(dict)
    
    def __init__(self, segments: List[Segment], workers: int = 4, speed: float = 1.0):
        super().__init__()
        self.segments = segments
        self.workers = min(max(workers, MIN_WORKERS), MAX_WORKERS)
        self.speed = speed
        self._is_running = True
        self._mutex = QMutex()
        self._temp_dir: Optional[str] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures = []
        self.stats = GenerationStats(total=len(segments))
        self.cache = CacheManager()
    
    def stop(self):
        """Safely stop generation."""
        with QMutexLocker(self._mutex):
            self._is_running = False
            if self._executor:
                self._executor.shutdown(wait=False, cancel_futures=True)
    
    def _should_continue(self) -> bool:
        """Check if operation should continue."""
        with QMutexLocker(self._mutex):
            return self._is_running
    
    def run(self):
        """Execute parallel generation."""
        self.stats.start_time = time.time()
        
        try:
            # Create temp directory
            self._temp_dir = tempfile.mkdtemp(prefix="khmer_tts_")
            self.status_update.emit(f"Created temp directory")
            
            # Create worker
            worker = GenerationWorker(self._temp_dir, self.cache)
            
            # Submit all tasks
            with ThreadPoolExecutor(max_workers=self.workers) as self._executor:
                futures = {}
                for idx, segment in enumerate(self.segments):
                    if not self._should_continue():
                        break
                    future = self._executor.submit(worker.generate, idx, segment, self.speed)
                    futures[future] = idx
                
                # Process results as they complete
                completed = 0
                for future in as_completed(futures.keys()):
                    if not self._should_continue():
                        break
                    
                    idx = futures[future]
                    try:
                        result_idx, success, error, from_cache = future.result(timeout=60)
                        
                        if success:
                            if from_cache:
                                self.stats.cached += 1
                            else:
                                self.stats.completed += 1
                            self.segment_completed.emit(result_idx)
                        else:
                            self.stats.failed += 1
                            self.segment_failed.emit(result_idx, error or "Unknown error")
                        
                        completed += 1
                        self.progress.emit(completed, len(self.segments))
                        
                        # Emit stats periodically
                        if completed % 5 == 0:
                            self.stats_updated.emit({
                                'completed': self.stats.completed,
                                'cached': self.stats.cached,
                                'failed': self.stats.failed,
                                'total': len(self.segments),
                                'elapsed': self.stats.elapsed or 0
                            })
                            
                    except Exception as e:
                        logger.exception(f"Error processing future for segment {idx}")
                        self.stats.failed += 1
                        self.segment_failed.emit(idx, str(e))
                        completed += 1
                        self.progress.emit(completed, len(self.segments))
            
            if self._should_continue():
                self.stats.end_time = time.time()
                elapsed = self.stats.elapsed or 0
                self.status_update.emit(
                    f"Generation complete: {self.stats.completed + self.stats.cached} succeeded "
                    f"({self.stats.cached} cached), {self.stats.failed} failed in {elapsed:.1f}s"
                )
                self.generation_finished.emit()
            
        except Exception as e:
            logger.exception("Fatal error in generator thread")
            self.status_update.emit(f"Fatal error: {str(e)}")
        finally:
            # Cleanup temp files after a delay
            QTimer.singleShot(60000, self.cleanup)  # Cleanup after 1 minute
    
    def cleanup(self):
        """Clean up temporary files."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up temp directory")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")


# ==================== PROFESSIONAL AUDIO ASSEMBLER ====================

class AudioAssembler(QThread):
    """Professional audio assembler with advanced features and time tracking."""
    
    progress = pyqtSignal(int)
    status_update = pyqtSignal(str)
    assembly_finished = pyqtSignal(str)
    assembly_failed = pyqtSignal(str)
    
    def __init__(self, segments: List[Segment], output_path: str,
                 crossfade_ms: int = 50,
                 fade_in_ms: int = 10,
                 fade_out_ms: int = 10,
                 normalize_master: bool = True,
                 apply_compression: bool = True,
                 padding_sec: float = 2.0,
                 normalize_per_segment: bool = True,
                 remove_silence: bool = False,
                 target_loudness: float = -16.0):  # LUFS
        super().__init__()
        self.segments = [s for s in segments if s.is_generated and s.audio_path]
        self.output_path = output_path
        self.crossfade_ms = min(max(crossfade_ms, MIN_CROSSFADE), MAX_CROSSFADE)
        self.fade_in_ms = fade_in_ms
        self.fade_out_ms = fade_out_ms
        self.normalize_master = normalize_master
        self.apply_compression = apply_compression
        self.normalize_per_segment = normalize_per_segment
        self.remove_silence = remove_silence
        self.target_loudness = target_loudness
        self.padding_sec = padding_sec
        self._is_running = True
        self._mutex = QMutex()
        self.assembly_time = 0.0
        
        # Validate pydub availability
        if not PYDUB_AVAILABLE:
            logger.error("pydub not available for audio assembly")
    
    def stop(self):
        """Safely stop assembly."""
        with QMutexLocker(self._mutex):
            self._is_running = False
    
    def _should_continue(self) -> bool:
        """Check if operation should continue."""
        with QMutexLocker(self._mutex):
            return self._is_running
    
    def _apply_fades(self, audio: AudioSegment) -> AudioSegment:
        """Apply fade in/out to prevent clicks."""
        if self.fade_in_ms > 0:
            fade_in = min(self.fade_in_ms, len(audio) // 2)
            audio = audio.fade_in(fade_in)
        
        if self.fade_out_ms > 0:
            fade_out = min(self.fade_out_ms, len(audio) // 2)
            audio = audio.fade_out(fade_out)
        
        return audio
    
    def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        """Normalize audio to target loudness."""
        try:
            # Simple peak normalization as fallback
            return normalize(audio)
        except Exception as e:
            logger.warning(f"Normalization failed: {e}")
            return audio
    
    def _remove_silence_edges(self, audio: AudioSegment, silence_thresh: int = -50) -> AudioSegment:
        """Remove silence from beginning and end."""
        try:
            nonsilent = detect_nonsilent(audio, min_silence_len=100, silence_thresh=silence_thresh)
            if nonsilent:
                start = nonsilent[0][0]
                end = nonsilent[-1][1]
                return audio[start:end]
        except Exception as e:
            logger.warning(f"Silence removal failed: {e}")
        return audio
    
    def _calculate_overlap(self, seg1: Segment, seg2: Segment) -> int:
        """
        Calculate overlap between two segments in milliseconds.
        Returns positive overlap or 0.
        """
        pos1_ms = int(seg1.seconds * 1000)
        pos2_ms = int(seg2.seconds * 1000)
        end1_ms = pos1_ms + seg1.duration_ms
        
        if pos2_ms < end1_ms:
            return end1_ms - pos2_ms
        return 0
    
    def run(self):
        """Execute audio assembly with time tracking."""
        import time
        start_time = time.time()
        
        if not PYDUB_AVAILABLE:
            self.assembly_failed.emit("pydub not available. Install with: pip install pydub")
            return
        
        if not self.segments:
            self.assembly_failed.emit("No valid segments to assemble")
            return
        
        try:
            self.status_update.emit("Starting audio assembly...")
            
            # Sort segments by timestamp
            sorted_segments = sorted(self.segments, key=lambda s: s.seconds)
            
            # Calculate total duration
            last_segment = max(sorted_segments, key=lambda s: s.seconds + s.duration_ms/1000)
            total_ms = int((last_segment.seconds + 
                           (last_segment.duration_ms / 1000) + 
                           self.padding_sec) * 1000)
            
            self.status_update.emit(f"Creating {total_ms/1000:.1f}s timeline...")
            
            # Create silent base track with high quality
            combined = AudioSegment.silent(
                duration=total_ms,
                frame_rate=SAMPLE_RATE
            )
            
            # Convert to stereo if mono
            if combined.channels == 1:
                combined = combined.set_channels(2)
            
            # Place each segment
            total_segments = len(sorted_segments)
            overlaps_found = 0
            segments_processed = 0
            
            for i, segment in enumerate(sorted_segments):
                if not self._should_continue():
                    return
                
                try:
                    # Load audio
                    audio = AudioSegment.from_mp3(segment.audio_path)
                    
                    # Convert to high quality
                    if audio.frame_rate != SAMPLE_RATE:
                        audio = audio.set_frame_rate(SAMPLE_RATE)
                    
                    if audio.channels != CHANNELS:
                        audio = audio.set_channels(CHANNELS)
                    
                    # Apply per-segment processing
                    if self.normalize_per_segment:
                        audio = self._normalize_audio(audio)
                    
                    if self.remove_silence:
                        audio = self._remove_silence_edges(audio)
                    
                    # Apply fades
                    audio = self._apply_fades(audio)
                    
                    # Calculate position
                    position_ms = int(segment.seconds * 1000)
                    
                    # Check for overlap with previous segment
                    if i > 0:
                        prev_segment = sorted_segments[i-1]
                        overlap = self._calculate_overlap(prev_segment, segment)
                        
                        if overlap > 0:
                            overlaps_found += 1
                            
                            # Apply crossfade if configured
                            if self.crossfade_ms > 0 and overlap > 0:
                                crossfade = min(self.crossfade_ms, overlap)
                                if crossfade > 0:
                                    # Get the previous segment's audio from combined
                                    prev_end = int(prev_segment.seconds * 1000) + prev_segment.duration_ms
                                    crossfade_start = max(position_ms - crossfade, prev_end - overlap)
                                    
                                    # Extract overlapping portion for crossfade
                                    overlap_duration = min(crossfade, overlap)
                                    if overlap_duration > 10:  # Minimum for crossfade
                                        # For professional crossfade, we'd need to implement properly
                                        pass
                    
                    # Overlay audio at exact position
                    combined = combined.overlay(audio, position=position_ms)
                    segments_processed += 1
                    
                    # Update progress based on segments processed (first 40%)
                    progress = int((i + 1) / total_segments * 40)
                    self.progress.emit(progress)
                    
                except Exception as e:
                    logger.exception(f"Error processing segment {i}")
                    self.status_update.emit(f"⚠️ Error processing segment {i+1}: {str(e)}")
            
            if not self._should_continue():
                return
            
            self.status_update.emit(f"Processed {segments_processed} segments, {overlaps_found} overlaps")
            self.progress.emit(50)
            
            # Apply master processing
            if self.normalize_master:
                self.status_update.emit("Applying master normalization...")
                combined = self._normalize_audio(combined)
                self.progress.emit(60)
            
            if self.apply_compression and len(combined) > 0:
                self.status_update.emit("Applying professional compression...")
                try:
                    # Gentle compression for natural sound
                    combined = compress_dynamic_range(
                        combined,
                        threshold=-20.0,
                        ratio=2.0,
                        attack=5.0,
                        release=50.0
                    )
                except Exception as e:
                    logger.warning(f"Compression failed: {e}")
                self.progress.emit(70)
            
            if not self._should_continue():
                return
            
            # Final loudness adjustment
            self.status_update.emit("Applying final loudness adjustment...")
            try:
                # Simple gain adjustment
                peak = combined.max_dBFS
                if peak < self.target_loudness:
                    gain = self.target_loudness - peak
                    combined = combined.apply_gain(gain)
            except:
                pass
            
            self.progress.emit(80)
            
            # Export final audio with high quality
            self.status_update.emit("Exporting high-quality MP3...")
            
            # Export with highest quality settings
            combined.export(
                self.output_path,
                format="mp3",
                bitrate=DEFAULT_BITRATE,
                parameters=[
                    "-ar", str(SAMPLE_RATE),
                    "-ac", str(CHANNELS),
                    "-q:a", "0"  # Highest MP3 quality
                ]
            )
            
            self.progress.emit(100)
            
            # Store assembly time
            self.assembly_time = time.time() - start_time
            
            # Verify export
            if os.path.exists(self.output_path):
                file_size = os.path.getsize(self.output_path) / (1024 * 1024)
                duration = len(combined) / 1000
                self.status_update.emit(
                    f"✅ Audio saved: {os.path.basename(self.output_path)} "
                    f"({duration:.1f}s, {file_size:.1f} MB, {SAMPLE_RATE/1000:.0f}kHz)"
                )
                self.assembly_finished.emit(self.output_path)
            else:
                self.assembly_failed.emit("Export failed: output file not created")
            
        except Exception as e:
            logger.exception("Fatal error in audio assembly")
            self.assembly_failed.emit(f"Assembly error: {str(e)}")


# ==================== PLAYBACK MANAGER ====================

class PlaybackManager:
    """Manages audio playback with proper state handling."""
    
    def __init__(self, player: QMediaPlayer):
        self.player = player
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.current_file: Optional[str] = None
        self.duration: int = 0
        
    def load(self, file_path: str) -> bool:
        """Load audio file for playback."""
        if not os.path.exists(file_path):
            return False
        
        self.current_file = file_path
        self.player.setSource(QUrl.fromLocalFile(file_path))
        return True
    
    def play(self) -> bool:
        """Start playback."""
        if not self.current_file:
            return False
        self.player.play()
        return True
    
    def pause(self):
        """Pause playback."""
        self.player.pause()
    
    def stop(self):
        """Stop playback and reset position."""
        self.player.stop()
    
    def set_volume(self, volume: int):
        """Set volume (0-100)."""
        self.audio_output.setVolume(volume / 100.0)
    
    def seek(self, position_ms: int):
        """Seek to position in milliseconds."""
        self.player.setPosition(position_ms)
    
    @property
    def is_playing(self) -> bool:
        return self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
    
    @property
    def is_paused(self) -> bool:
        return self.player.playbackState() == QMediaPlayer.PlaybackState.PausedState
    
    @property
    def is_stopped(self) -> bool:
        return self.player.playbackState() == QMediaPlayer.PlaybackState.StoppedState

# ==================== MAIN WINDOW ====================

class KhmerTTSStudio(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.segments: List[Segment] = []
        self.generator: Optional[GeneratorThread] = None
        self.assembler: Optional[AudioAssembler] = None
        self.playback: Optional[PlaybackManager] = None
        self.output_dir: str = str(Path.home() / "Desktop")
        self.current_file: Optional[str] = None
        self._temp_dirs: List[str] = []
        self.current_project: Optional[str] = None
        self.cache = CacheManager()
        
        self.setWindowTitle("Khmer TTS Studio – High Quality Edition")
        self.setMinimumSize(1200, 750)
        
        self._init_ui()
        self._setup_player()
        self._check_dependencies()
        self._load_settings()
        self._setup_segment_editor()
    
    def _init_ui(self):
        """Initialize user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Menu bar
        self._create_menu_bar()
        
        # === Top toolbar ===
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        
        self.load_btn = QPushButton("📂 Load Transcript")
        self.load_btn.clicked.connect(self._on_load_transcript)
        self.load_btn.setMinimumHeight(40)
        self.load_btn.setMinimumWidth(150)
        toolbar_layout.addWidget(self.load_btn)
        
        self.save_btn = QPushButton("💾 Save Project")
        self.save_btn.clicked.connect(self._on_save_project)
        self.save_btn.setEnabled(False)
        self.save_btn.setMinimumHeight(40)
        toolbar_layout.addWidget(self.save_btn)
        
        self.file_label = QLabel("No file loaded")
        self.file_label.setMinimumHeight(30)
        self.file_label.setStyleSheet("padding: 5px; background-color: #2d2d2d; border-radius: 4px;")
        toolbar_layout.addWidget(self.file_label, 1)
        
        self.segment_count_label = QLabel("0 segments")
        self.segment_count_label.setMinimumWidth(120)
        self.segment_count_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        toolbar_layout.addWidget(self.segment_count_label)
        
        layout.addWidget(toolbar)
        
        # === Main splitter ===
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        
        # Left panel - Segment list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        list_header = QHBoxLayout()
        list_header.addWidget(QLabel("📋 Segments"))
        list_header.addStretch()
        
        self.clear_errors_btn = QPushButton("Clear Errors")
        self.clear_errors_btn.clicked.connect(self._on_clear_errors)
        self.clear_errors_btn.setEnabled(False)
        list_header.addWidget(self.clear_errors_btn)
        
        left_layout.addLayout(list_header)
        
        self.segment_list = QListWidget()
        self.segment_list.setAlternatingRowColors(True)
        self.segment_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        left_layout.addWidget(self.segment_list)
        
        splitter.addWidget(left_panel)
        
        # Right panel - Controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)
        
        # Tabs for organization
        tabs = QTabWidget()
        
        # ==================== VOICE SETTINGS TAB ====================
        voice_tab = QWidget()
        voice_layout = QVBoxLayout(voice_tab)

        voice_group = QGroupBox("Voice Settings")
        voice_group_layout = QVBoxLayout(voice_group)

        # Language selection
        language_layout = QHBoxLayout()
        language_layout.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItem("ខ្មែរ (Khmer)", Language.KHMER)
        self.language_combo.addItem("English", Language.ENGLISH)
        self.language_combo.addItem("中文 (Chinese)", Language.CHINESE)
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        language_layout.addWidget(self.language_combo)
        language_layout.addStretch()
        voice_group_layout.addLayout(language_layout)
        
        preview_layout = QHBoxLayout()
        self.preview_voice_btn = QPushButton("🔊 Preview Voice")
        self.preview_voice_btn.clicked.connect(self._on_preview_voice)
        self.preview_voice_btn.setMaximumWidth(120)
        preview_layout.addWidget(self.preview_voice_btn)
        preview_layout.addStretch()
        voice_group_layout.addLayout(preview_layout)
        
        # Voice mode selection
        gender_layout = QHBoxLayout()
        gender_layout.addWidget(QLabel("Voice Mode:"))
        self.gender_combo = QComboBox()
        self.gender_combo.addItem("Alternating (Female/Male)", VoiceGender.ALTERNATING)
        self.gender_combo.addItem("Female Only", VoiceGender.FEMALE)
        self.gender_combo.addItem("Male Only", VoiceGender.MALE)
        gender_layout.addWidget(self.gender_combo)
        gender_layout.addStretch()
        voice_group_layout.addLayout(gender_layout)
        
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speech Speed:"))
        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(MIN_SPEED, MAX_SPEED)
        self.speed_spin.setValue(100)
        self.speed_spin.setSuffix("%")
        speed_layout.addWidget(self.speed_spin)
        speed_layout.addStretch()
        voice_group_layout.addLayout(speed_layout)
        
        workers_layout = QHBoxLayout()
        workers_layout.addWidget(QLabel("Parallel Workers:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(MIN_WORKERS, MAX_WORKERS)
        self.workers_spin.setValue(4)
        workers_layout.addWidget(self.workers_spin)
        workers_layout.addStretch()
        voice_group_layout.addLayout(workers_layout)
        
        voice_layout.addWidget(voice_group)
        
        # Cache info
        cache_layout = QHBoxLayout()
        cache_layout.addWidget(QLabel("Cache:"))
        self.cache_label = QLabel("Calculating...")
        cache_layout.addWidget(self.cache_label)
        cache_layout.addStretch()
        
        self.clear_cache_btn = QPushButton("Clear Cache")
        self.clear_cache_btn.clicked.connect(self._on_clear_cache)
        cache_layout.addWidget(self.clear_cache_btn)
        
        voice_layout.addLayout(cache_layout)
        voice_layout.addStretch()
        
        tabs.addTab(voice_tab, "Voice")
        
        # ==================== AUDIO PROCESSING TAB ====================
        audio_tab = QWidget()
        audio_layout = QVBoxLayout(audio_tab)
        
        audio_group = QGroupBox("Audio Processing")
        audio_group_layout = QVBoxLayout(audio_group)
        
        crossfade_layout = QHBoxLayout()
        crossfade_layout.addWidget(QLabel("Crossfade:"))
        self.crossfade_spin = QSpinBox()
        self.crossfade_spin.setRange(MIN_CROSSFADE, MAX_CROSSFADE)
        self.crossfade_spin.setValue(50)
        self.crossfade_spin.setSuffix(" ms")
        crossfade_layout.addWidget(self.crossfade_spin)
        crossfade_layout.addStretch()
        audio_group_layout.addLayout(crossfade_layout)
        
        self.normalize_per_seg_check = QCheckBox("Normalize each segment")
        self.normalize_per_seg_check.setChecked(True)
        audio_group_layout.addWidget(self.normalize_per_seg_check)
        
        self.normalize_master_check = QCheckBox("Normalize overall audio")
        self.normalize_master_check.setChecked(True)
        audio_group_layout.addWidget(self.normalize_master_check)
        
        self.compress_check = QCheckBox("Apply dynamic compression")
        self.compress_check.setChecked(True)
        audio_group_layout.addWidget(self.compress_check)
        
        self.remove_silence_check = QCheckBox("Remove silence from segments")
        self.remove_silence_check.setChecked(False)
        audio_group_layout.addWidget(self.remove_silence_check)
        
        padding_layout = QHBoxLayout()
        padding_layout.addWidget(QLabel("End padding:"))
        self.padding_spin = QDoubleSpinBox()
        self.padding_spin.setRange(MIN_PADDING, MAX_PADDING)
        self.padding_spin.setValue(2.0)
        self.padding_spin.setSuffix(" s")
        self.padding_spin.setSingleStep(0.5)
        padding_layout.addWidget(self.padding_spin)
        padding_layout.addStretch()
        audio_group_layout.addLayout(padding_layout)
        
        audio_layout.addWidget(audio_group)
        audio_layout.addStretch()
        
        tabs.addTab(audio_tab, "Audio")
        
        # ==================== OUTPUT TAB ====================
        output_tab = QWidget()
        output_layout = QVBoxLayout(output_tab)
        
        output_group = QGroupBox("Output Settings")
        output_group_layout = QVBoxLayout(output_group)
        
        # Filename template
        template_layout = QHBoxLayout()
        template_layout.addWidget(QLabel("Template:"))
        self.template_edit = QLineEdit()
        self.template_edit.setText("khmer_tts_{date}_{time}")
        self.template_edit.setToolTip("Use {date}, {time}, {duration}")
        template_layout.addWidget(self.template_edit)
        output_group_layout.addLayout(template_layout)
        
        # Output directory
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Output dir:"))
        self.dir_label = QLabel(self.output_dir)
        self.dir_label.setWordWrap(True)
        dir_layout.addWidget(self.dir_label, 1)
        
        self.browse_dir_btn = QPushButton("Browse...")
        self.browse_dir_btn.clicked.connect(self._on_browse_output_dir)
        dir_layout.addWidget(self.browse_dir_btn)
        output_group_layout.addLayout(dir_layout)
        
        # Quality info
        quality_label = QLabel(f"Quality: {SAMPLE_RATE/1000:.0f}kHz / {CHANNELS}ch / {DEFAULT_BITRATE}")
        quality_label.setStyleSheet("color: #4CAF50; padding: 5px;")
        output_group_layout.addWidget(quality_label)
        
        output_layout.addWidget(output_group)
        output_layout.addStretch()
        
        tabs.addTab(output_tab, "Output")
        
        right_layout.addWidget(tabs)
        
        # ==================== ACTION BUTTONS ====================
        self.generate_btn = QPushButton("🎙️ Generate Speech")
        self.generate_btn.clicked.connect(self._on_generate)
        self.generate_btn.setEnabled(False)
        self.generate_btn.setMinimumHeight(45)
        right_layout.addWidget(self.generate_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)
        
        # ==================== STATISTICS DASHBOARD ====================
        self.stats_group = QGroupBox("📊 Statistics")
        self.stats_group.setVisible(False)  # Hidden by default
        stats_layout = QGridLayout()

        # Row 0
        stats_layout.addWidget(QLabel("Speed:"), 0, 0)
        self.speed_label = QLabel("-- seg/min")
        self.speed_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        stats_layout.addWidget(self.speed_label, 0, 1)

        stats_layout.addWidget(QLabel("ETA:"), 0, 2)
        self.eta_label = QLabel("--:--")
        self.eta_label.setStyleSheet("color: #FFA500; font-weight: bold;")
        stats_layout.addWidget(self.eta_label, 0, 3)

        # Row 1
        stats_layout.addWidget(QLabel("Success:"), 1, 0)
        self.success_rate_label = QLabel("--%")
        self.success_rate_label.setStyleSheet("color: #4CAF50;")
        stats_layout.addWidget(self.success_rate_label, 1, 1)

        stats_layout.addWidget(QLabel("Cache hits:"), 1, 2)
        self.cache_hits_label = QLabel("0")
        self.cache_hits_label.setStyleSheet("color: #2196F3;")
        stats_layout.addWidget(self.cache_hits_label, 1, 3)

        # Row 2
        stats_layout.addWidget(QLabel("Completed:"), 2, 0)
        self.completed_label = QLabel("0/0")
        stats_layout.addWidget(self.completed_label, 2, 1)

        stats_layout.addWidget(QLabel("Failed:"), 2, 2)
        self.failed_label = QLabel("0")
        self.failed_label.setStyleSheet("color: #f44336;")
        stats_layout.addWidget(self.failed_label, 2, 3)

        self.stats_group.setLayout(stats_layout)
        right_layout.addWidget(self.stats_group)
        
        # ==================== EXPORT BUTTON WITH MENU ====================
        self.export_btn = QPushButton("🔊 Export")
        self.export_btn.setEnabled(False)
        self.export_btn.setMinimumHeight(45)
        
        # Create export menu
        export_menu = QMenu(self)
        
        # Format actions
        mp3_action = QAction("MP3 (320kbps)", self)
        mp3_action.triggered.connect(lambda: self._export_with_options("mp3", "320", False, ""))
        export_menu.addAction(mp3_action)
        
        wav_action = QAction("WAV (Lossless)", self)
        wav_action.triggered.connect(lambda: self._export_with_options("wav", "lossless", False, ""))
        export_menu.addAction(wav_action)
        
        flac_action = QAction("FLAC (Lossless)", self)
        flac_action.triggered.connect(lambda: self._export_with_options("flac", "lossless", False, ""))
        export_menu.addAction(flac_action)
        
        m4a_action = QAction("M4A (AAC)", self)
        m4a_action.triggered.connect(lambda: self._export_with_options("m4a", "320", False, ""))
        export_menu.addAction(m4a_action)
        
        ogg_action = QAction("OGG (Vorbis)", self)
        ogg_action.triggered.connect(lambda: self._export_with_options("ogg", "320", False, ""))
        export_menu.addAction(ogg_action)
        
        export_menu.addSeparator()
        
        # Split options
        split_action = QAction("Split into separate files...", self)
        split_action.triggered.connect(self._on_export_with_options)
        export_menu.addAction(split_action)
        
        advanced_action = QAction("Advanced export options...", self)
        advanced_action.triggered.connect(self._on_export_with_options)
        export_menu.addAction(advanced_action)
        
        self.export_btn.setMenu(export_menu)
        right_layout.addWidget(self.export_btn)
        
        # ==================== PLAYBACK CONTROLS ====================
        playback_group = QGroupBox("Playback")
        playback_layout = QVBoxLayout(playback_group)
        
        # Position slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Position:"))
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setEnabled(False)
        self.position_slider.setRange(0, 0)
        slider_layout.addWidget(self.position_slider)
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setFixedWidth(120)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        slider_layout.addWidget(self.time_label)
        playback_layout.addLayout(slider_layout)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶")
        self.play_btn.clicked.connect(self._on_play)
        self.play_btn.setEnabled(False)
        self.play_btn.setFixedWidth(40)
        control_layout.addWidget(self.play_btn)
        
        self.pause_btn = QPushButton("⏸")
        self.pause_btn.clicked.connect(self._on_pause)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setFixedWidth(40)
        control_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("⏹")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setFixedWidth(40)
        control_layout.addWidget(self.stop_btn)
        
        control_layout.addStretch()
        control_layout.addWidget(QLabel("Volume"))
        
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setFixedWidth(100)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        control_layout.addWidget(self.volume_slider)
        
        playback_layout.addLayout(control_layout)
        
        right_layout.addWidget(playback_group)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 400])
        
        layout.addWidget(splitter)
        
        # Status bar
        status_bar = self.statusBar()
        self.status_label = QLabel("Ready")
        status_bar.addWidget(self.status_label)
        
        self.stats_label = QLabel("")
        status_bar.addPermanentWidget(self.stats_label)
        
        # Update cache info
        self._update_cache_info()
    
    def _on_language_changed(self, index):
        """Handle language change."""
        language = self.language_combo.currentData()
        logger.info(f"Language changed to: {language.value}")

    def _on_preview_voice(self):
        """Preview the selected voice."""
        language = self.language_combo.currentData()
        gender_mode = self.gender_combo.currentData()
        
        # Determine which voice to preview
        if gender_mode == VoiceGender.ALTERNATING:
            gender = VoiceGender.FEMALE  # Preview female for alternating
        else:
            gender = gender_mode
        
        voice = VOICE_MAP.get((language, gender))
        if not voice:
            QMessageBox.warning(self, "Preview Error", f"No voice available for {language.value} {gender.value}")
            return
        
        # Preview text based on language
        preview_texts = {
            Language.KHMER: "សួស្តី! នេះជាការសាកល្បងសំឡេង",
            Language.ENGLISH: "Hello! This is a voice preview.",
            Language.CHINESE: "你好！这是语音预览。"
        }
        
        text = preview_texts.get(language, "Hello! This is a voice preview.")
        
        # Disable button during preview
        self.preview_voice_btn.setEnabled(False)
        self.preview_voice_btn.setText("⏳ Previewing...")
        
        # Run preview in thread to avoid UI freeze
        class PreviewThread(QThread):
            finished = pyqtSignal(bool, str)
            
            def __init__(self, text, voice):
                super().__init__()
                self.text = text
                self.voice = voice
            
            def run(self):
                try:
                    temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                    temp_path = temp_file.name
                    temp_file.close()
                    
                    async def _generate():
                        communicate = edge_tts.Communicate(self.text, self.voice)
                        await communicate.save(temp_path)
                    
                    asyncio.run(_generate())
                    
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        self.finished.emit(True, temp_path)
                    else:
                        self.finished.emit(False, "Failed to generate preview")
                except Exception as e:
                    self.finished.emit(False, str(e))
        
        self.preview_thread = PreviewThread(text, voice)
        self.preview_thread.finished.connect(self._on_preview_finished)
        self.preview_thread.start()

    def _on_preview_finished(self, success: bool, result: str):
        """Handle preview completion."""
        self.preview_voice_btn.setEnabled(True)
        self.preview_voice_btn.setText("🔊 Preview Voice")
        
        if success:
            # Play the preview
            self.playback.stop()
            self.playback.load(result)
            self.playback.play()
            
            # Clean up temp file after playback
            QTimer.singleShot(5000, lambda: self._cleanup_preview_file(result))
        else:
            QMessageBox.warning(self, "Preview Failed", f"Could not generate preview:\n{result}")

    def _cleanup_preview_file(self, file_path: str):
        """Clean up temporary preview file."""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass
    
    def _setup_segment_editor(self):
        """Setup double-click editing for segments."""
        self.segment_list.itemDoubleClicked.connect(self._edit_segment)

    def _edit_segment(self, item):
        """Edit segment text on double-click."""
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is None:
            return
        
        segment = self.segments[index]
        
        # Create edit dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Segment - {segment.time_str()}")
        dialog.setMinimumWidth(500)
        
        layout = QVBoxLayout(dialog)
        
        # Original text
        orig_label = QLabel("Original text:")
        orig_text = QTextEdit()
        orig_text.setPlainText(segment.text)
        orig_text.setReadOnly(True)
        orig_text.setMaximumHeight(100)
        layout.addWidget(orig_label)
        layout.addWidget(orig_text)
        
        # Edit text
        edit_label = QLabel("Edit text:")
        self.edit_text = QTextEdit()
        self.edit_text.setPlainText(segment.text)
        self.edit_text.setMaximumHeight(150)
        layout.addWidget(edit_label)
        layout.addWidget(self.edit_text)
        
        # Language selector
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("Language:"))
        lang_combo = QComboBox()
        lang_combo.addItem("🇰🇭 Khmer", Language.KHMER)
        lang_combo.addItem("🇺🇸 English", Language.ENGLISH)
        lang_combo.addItem("🇨🇳 Chinese", Language.CHINESE)
        
        # Set current language
        index_lang = 0
        if segment.language == Language.ENGLISH:
            index_lang = 1
        elif segment.language == Language.CHINESE: 
            index_lang = 2
        lang_combo.setCurrentIndex(index_lang)
        
        lang_layout.addWidget(lang_combo)
        layout.addLayout(lang_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_text = self.edit_text.toPlainText().strip()
            new_language = lang_combo.currentData()
            
            if new_text and new_text != segment.text:
                # Reset generation state
                segment.text = new_text
                segment.language = new_language
                segment.is_generated = False
                segment.has_error = False
                segment.audio_path = None
                segment.duration_ms = 0
                if hasattr(segment, 'from_cache'):
                    delattr(segment, 'from_cache')
                
                self._update_segment_list_item(index)
                self._update_ui_state()
                self.status_label.setText(f"✏️ Segment {index} updated")
    
    def _create_menu_bar(self):
        """Create application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Project", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._on_new_project)
        file_menu.addAction(new_action)
        
        open_action = QAction("Open Project...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("Save Project", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save Project As...", self)
        save_as_action.triggered.connect(self._on_save_project_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        import_action = QAction("Import Transcript...", self)
        import_action.triggered.connect(self._on_load_transcript)
        file_menu.addAction(import_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        clear_errors_action = QAction("Clear Errors", self)
        clear_errors_action.triggered.connect(self._on_clear_errors)
        edit_menu.addAction(clear_errors_action)
        
        clear_all_action = QAction("Clear All", self)
        clear_all_action.triggered.connect(self._on_clear_all)
        edit_menu.addAction(clear_all_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        cache_action = QAction("Clear Cache", self)
        cache_action.triggered.connect(self._on_clear_cache)
        tools_menu.addAction(cache_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)
    
    def _setup_player(self):
        """Initialize media player."""
        player = QMediaPlayer()
        self.playback = PlaybackManager(player)
        
        # Connect signals
        player.positionChanged.connect(self._on_position_changed)
        player.durationChanged.connect(self._on_duration_changed)
        player.playbackStateChanged.connect(self._on_playback_state_changed)
    
    def _check_dependencies(self):
        """Check for required dependencies."""
        if not PYDUB_AVAILABLE:
            QTimer.singleShot(1000, lambda: 
                QMessageBox.warning(
                    self, "Missing Dependency",
                    "pydub not installed. Audio processing will be limited.\n"
                    "Install with: pip install pydub"
                )
            )
    
    def _on_export_with_options(self):
        """Export with advanced options."""
        ready_segments = [s for s in self.segments if s.is_generated]
        if not ready_segments:
            QMessageBox.warning(self, "No Audio", "No generated segments available")
            return
        
        # Create export options dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Options")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        
        # Format selection
        format_group = QGroupBox("Format")
        format_layout = QVBoxLayout()
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["MP3", "WAV", "FLAC", "M4A", "OGG"])
        format_layout.addWidget(self.export_format_combo)
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)
        
        # Quality selection
        quality_group = QGroupBox("Quality")
        quality_layout = QVBoxLayout()
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["128 kbps", "192 kbps", "256 kbps", "320 kbps", "Lossless"])
        quality_layout.addWidget(self.quality_combo)
        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)
        
        # Split options
        split_group = QGroupBox("Split Options")
        split_layout = QVBoxLayout()
        self.split_check = QCheckBox("Split into separate files")
        self.split_check.toggled.connect(self._toggle_split_options)
        split_layout.addWidget(self.split_check)
        
        self.naming_combo = QComboBox()
        self.naming_combo.addItems(["segment_{number}", "{timestamp}_{text}", "{project}_part{number}"])
        self.naming_combo.setEnabled(False)
        split_layout.addWidget(self.naming_combo)
        
        split_group.setLayout(split_layout)
        layout.addWidget(split_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        export_btn = QPushButton("🔊 Export")
        export_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(export_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._export_with_options(
                self.export_format_combo.currentText().lower(),
                self.quality_combo.currentText().split()[0],
                self.split_check.isChecked(),
                self.naming_combo.currentText()
            )

    def _toggle_split_options(self, checked):
        """Enable/disable naming options based on split checkbox."""
        self.naming_combo.setEnabled(checked)

    def _export_with_options(self, format_type: str, quality: str, split: bool, naming: str):
        """Export with specified options."""
        if split:
            self._export_split_files(format_type, quality, naming)
        else:
            self._export_single_file(format_type, quality)

    def _export_single_file(self, format_type: str, quality: str):
        """Export as single file with custom format."""
        # Generate filename
        template = self.template_edit.text().strip()
        if not template:
            template = "khmer_tts_{date}_{time}"
        
        now = datetime.now()
        filename = template.replace("{date}", now.strftime("%Y%m%d"))
        filename = filename.replace("{time}", now.strftime("%H%M%S"))
        
        # Calculate duration
        if self.segments:
            total_sec = max(s.seconds + (s.duration_ms/1000) for s in self.segments if s.is_generated)
            h = int(total_sec // 3600)
            m = int((total_sec % 3600) // 60)
            s = int(total_sec % 60)
            filename = filename.replace("{duration}", f"{h:02d}{m:02d}{s:02d}")
        
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        if not filename.endswith(f'.{format_type}'):
            filename += f'.{format_type}'
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Check if file exists
        if os.path.exists(output_path):
            reply = QMessageBox.question(
                self,
                "File Exists",
                f"{filename} already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        # Start assembly with custom format
        self._start_assembly_with_progress(output_path, format_type, quality)

    def _export_split_files(self, format_type: str, quality: str, naming: str):
        """Export each segment as separate file."""
        # Ask for output directory
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory for Split Files",
            self.output_dir
        )
        
        if not dir_path:
            return
        
        # Get base project name
        if self.current_project:
            base_name = Path(self.current_project).stem
        else:
            base_name = "tts_export"
        
        # Export each segment
        exported = 0
        for i, segment in enumerate(self.segments):
            if not segment.is_generated or not segment.audio_path:
                continue
            
            # Generate filename based on naming pattern
            if naming == "segment_{number}":
                filename = f"{base_name}_segment_{i:04d}.{format_type}"
            elif naming == "{timestamp}_{text}":
                preview = segment.preview_text(20).replace(" ", "_")
                filename = f"{base_name}_{segment.time_str()}_{preview}.{format_type}"
            else:  # {project}_part{number}
                filename = f"{base_name}_part_{i:04d}.{format_type}"
            
            # Sanitize filename
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            output_path = os.path.join(dir_path, filename)
            
            # Copy or convert file
            try:
                if format_type == "mp3" and quality == "320":
                    # Direct copy if same format
                    shutil.copy2(segment.audio_path, output_path)
                else:
                    # Need to convert using pydub
                    if PYDUB_AVAILABLE:
                        audio = AudioSegment.from_mp3(segment.audio_path)
                        
                        # Set quality
                        bitrate = f"{quality}k" if quality.isdigit() else "320k"
                        
                        # Export with specified format
                        audio.export(
                            output_path,
                            format=format_type,
                            bitrate=bitrate
                        )
                    else:
                        # Fallback to copy
                        shutil.copy2(segment.audio_path, output_path)
                
                exported += 1
            except Exception as e:
                logger.error(f"Failed to export segment {i}: {e}")
        
        self.status_label.setText(f"✅ Exported {exported} files to {dir_path}")
        QMessageBox.information(self, "Export Complete", f"Exported {exported} files to:\n{dir_path}")

    def _load_settings(self):
        """Load saved settings."""
        settings_file = Path.home() / ".khmer_tts_settings.json"
        if settings_file.exists():
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    self.output_dir = settings.get('output_dir', self.output_dir)
                    self.dir_label.setText(self.output_dir)
            except:
                pass
    
    def _save_settings(self):
        """Save settings."""
        settings_file = Path.home() / ".khmer_tts_settings.json"
        try:
            with open(settings_file, 'w') as f:
                json.dump({
                    'output_dir': self.output_dir
                }, f)
        except:
            pass
    
    def _update_cache_info(self):
        """Update cache size information."""
        try:
            total_size = sum(f.stat().st_size for f in CACHE_DIR.glob("*.mp3")) / (1024 * 1024)
            cache_count = len(list(CACHE_DIR.glob("*.mp3")))
            self.cache_label.setText(f"{cache_count} files ({total_size:.1f} MB)")
        except:
            self.cache_label.setText("Unknown")
    
    def _update_ui_state(self):
        """Update UI based on current state."""
        has_segments = len(self.segments) > 0
        has_generated = any(s.is_generated for s in self.segments)
        has_errors = any(s.has_error for s in self.segments)
        
        self.generate_btn.setEnabled(has_segments)
        self.export_btn.setEnabled(has_generated)
        self.save_btn.setEnabled(has_segments)
        self.clear_errors_btn.setEnabled(has_errors)
        
        # Update segment count
        total = len(self.segments)
        generated = sum(1 for s in self.segments if s.is_generated)
        failed = sum(1 for s in self.segments if s.has_error)
        cached = sum(1 for s in self.segments if s.is_generated and hasattr(s, 'from_cache') and s.from_cache)
        
        self.segment_count_label.setText(
            f"{total} segs | ✅ {generated} | 💾 {cached} | ❌ {failed}"
        )
    
    def _update_segment_list_item(self, index: int):
        """Update a single list item with current state."""
        if index < 0 or index >= len(self.segments):
            return
        
        segment = self.segments[index]
        item = self.segment_list.item(index)
        if not item:
            return
        
        # Language icons mapping
        language_icons = {
            Language.KHMER: "🇰🇭",
            Language.ENGLISH: "🇺🇸",
            Language.CHINESE: "🇨🇳"
        }
        
        # Build display text with language icon
        lang_icon = language_icons.get(segment.language, "🌐")
        gender_icon = "♀" if segment.gender == VoiceGender.FEMALE else "♂"
        status_icon = segment.display_icon
        time_str = segment.time_str()
        preview = segment.preview_text(35)
        
        # Add cache indicator if from cache
        if hasattr(segment, 'from_cache') and segment.from_cache:
            cache_icon = "💾"
        else:
            cache_icon = ""
        
        display = f"{status_icon}{cache_icon} {lang_icon}{gender_icon} {time_str}  {preview}"
        item.setText(display)
        
        # Set color based on state
        if segment.has_error:
            color = "#f44336"  # Red
        elif segment.is_generated:
            color = "#4CAF50"  # Green
        else:
            color = "#FFA500"  # Orange for pending
        
        item.setForeground(QColor(color))
        
        # Store segment data
        item.setData(Qt.ItemDataRole.UserRole, index)
    
    def _refresh_segment_list(self):
        """Refresh all list items."""
        self.segment_list.clear()
        for i, segment in enumerate(self.segments):
            # Language icons mapping
            language_icons = {
                Language.KHMER: "🇰🇭",
                Language.ENGLISH: "🇺🇸",
                Language.CHINESE: "🇨🇳"
            }
            
            lang_icon = language_icons.get(segment.language, "🌐")
            gender_icon = "♀" if segment.gender == VoiceGender.FEMALE else "♂"
            status_icon = segment.display_icon
            time_str = segment.time_str()
            preview = segment.preview_text(35)
            
            display = f"{status_icon} {lang_icon}{gender_icon} {time_str}  {preview}"
            item = QListWidgetItem(display)
            
            # Set color based on state
            if segment.has_error:
                color = "#f44336"
            elif segment.is_generated:
                color = "#4CAF50"
            else:
                color = "#FFA500"
            
            item.setForeground(QColor(color))
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.segment_list.addItem(item)

    def _on_new_project(self):
        """Create new project."""
        if self.segments:
            reply = QMessageBox.question(
                self, "New Project",
                "Current project not saved. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        self.segments = []
        self.current_project = None
        self.file_label.setText("No project")
        self._refresh_segment_list()
        self._update_ui_state()
        self.status_label.setText("New project created")
    
    def _on_open_project(self):
        """Open saved project."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "Khmer TTS Project (*.ktts);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            segments = []
            for seg_data in data.get('segments', []):
                segment = Segment.from_dict(seg_data)
                segments.append(segment)
            
            self.segments = segments
            self.current_project = file_path
            self.file_label.setText(os.path.basename(file_path))
            self._refresh_segment_list()
            self._update_ui_state()
            self.status_label.setText(f"Loaded project: {os.path.basename(file_path)}")
            
        except Exception as e:
            logger.exception("Error opening project")
            QMessageBox.critical(self, "Error", f"Failed to open project:\n{str(e)}")
    
    def _on_save_project(self):
        """Save current project."""
        if not self.segments:
            return
        
        if self.current_project:
            self._save_project_to_file(self.current_project)
        else:
            self._on_save_project_as()
    
    def _on_save_project_as(self):
        """Save project with new name."""
        if not self.segments:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            "",
            "Khmer TTS Project (*.ktts);;All Files (*.*)"
        )
        
        if file_path:
            if not file_path.endswith('.ktts'):
                file_path += '.ktts'
            self._save_project_to_file(file_path)
            self.current_project = file_path
            self.file_label.setText(os.path.basename(file_path))
    
    def _save_project_to_file(self, file_path: str):
        """Save project to specific file."""
        try:
            data = {
                'version': '1.0',
                'created': datetime.now().isoformat(),
                'segments': [s.to_dict() for s in self.segments]
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.status_label.setText(f"Project saved: {os.path.basename(file_path)}")
            
        except Exception as e:
            logger.exception("Error saving project")
            QMessageBox.critical(self, "Error", f"Failed to save project:\n{str(e)}")
    
    def _on_load_transcript(self):
        """Load and parse transcript file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Transcript",
            "",
            "Text Files (*.txt);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:  # Handle BOM
                content = f.read()
            
            # Parse lines
            lines = content.split('\n')
            new_segments = []
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Parse timestamp and text
                timestamp_match = re.match(r'^\[([^\]]+)\](.*)$', line)
                if timestamp_match:
                    timestamp = f"[{timestamp_match.group(1)}]"
                    text = timestamp_match.group(2).strip()
                    
                    if text:
                        segment = Segment(timestamp, text)
                        new_segments.append(segment)
                    else:
                        logger.warning(f"Line {line_num}: Empty text after timestamp")
                else:
                    logger.warning(f"Line {line_num}: Invalid format (no timestamp)")
            
            if new_segments:
                self.segments = new_segments
                self.current_project = None
                self.file_label.setText(os.path.basename(file_path))
                self._refresh_segment_list()
                self._update_ui_state()
                self.status_label.setText(f"✅ Loaded {len(self.segments)} segments")
                logger.info(f"Loaded {len(self.segments)} segments from {file_path}")
            else:
                QMessageBox.warning(self, "Warning", "No valid segments found in file")
            
        except Exception as e:
            logger.exception("Error loading transcript")
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")
    
    def _on_generate(self):
        """Start speech generation."""
        if not self.segments:
            return
        
        # Check cache for existing files
        cached_count = 0
        for segment in self.segments:
            if segment.is_generated:
                cached_count += 1
        
        if cached_count > 0:
            reply = QMessageBox.question(
                self, "Regenerate",
                f"{cached_count} segments already generated. Regenerate all?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        # Reset segment states
        for segment in self.segments:
            segment.is_generated = False
            segment.has_error = False
            segment.error_message = ""
            segment.audio_path = None
            if hasattr(segment, 'from_cache'):
                delattr(segment, 'from_cache')
        
        self._refresh_segment_list()
        
        # Get generation parameters
        gender_mode = self.gender_combo.currentData()
        speed = self.speed_spin.value() / 100.0
        workers = self.workers_spin.value()
        current_language = self.language_combo.currentData()  # Get current language
        
        # Assign genders AND language based on mode
        for i, segment in enumerate(self.segments):
            # Set the language from the combo box
            segment.language = current_language
            
            # Set gender based on mode
            if gender_mode == VoiceGender.FEMALE:
                segment.gender = VoiceGender.FEMALE
            elif gender_mode == VoiceGender.MALE:
                segment.gender = VoiceGender.MALE
            else:  # ALTERNATING
                segment.gender = VoiceGender.FEMALE if i % 2 == 0 else VoiceGender.MALE
        
        self._refresh_segment_list()
        
        # Update UI
        self.generate_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(self.segments))
        self.progress_bar.setValue(0)
        
        # Show statistics group
        if hasattr(self, 'stats_group'):
            self.stats_group.setVisible(True)
            self.speed_label.setText("-- seg/min")
            self.eta_label.setText("--:--")
            self.success_rate_label.setText("--%")
            self.cache_hits_label.setText("0")
            self.completed_label.setText(f"0/{len(self.segments)}")
            self.failed_label.setText("0")
        
        self.status_label.setText(f"Generating {len(self.segments)} segments with {workers} workers...")
        
        # Start generator thread
        self.generator = GeneratorThread(self.segments, workers, speed)
        self.generator.progress.connect(self._on_generation_progress)
        self.generator.segment_completed.connect(self._on_segment_completed)
        self.generator.segment_failed.connect(self._on_segment_failed)
        self.generator.status_update.connect(self.status_label.setText)
        self.generator.generation_finished.connect(self._on_generation_finished)
        self.generator.stats_updated.connect(self._on_stats_updated_detailed)
        self.generator.start()
    
    def _on_stats_updated_detailed(self, stats: dict):
        """Update detailed statistics display."""
        # Update the simple stats label
        self.stats_label.setText(
            f"✅ {stats['completed']} | 💾 {stats['cached']} | ❌ {stats['failed']}"
        )
        
        # Update detailed stats if available
        if hasattr(self, 'stats_group') and self.stats_group.isVisible():
            total = stats['total']
            completed = stats['completed'] + stats['cached']
            failed = stats['failed']
            
            # Update basic counts
            self.completed_label.setText(f"{completed}/{total}")
            self.failed_label.setText(str(failed))
            self.cache_hits_label.setText(str(stats['cached']))
            
            # Calculate and update speed and ETA
            if 'elapsed' in stats and stats['elapsed'] > 0:
                speed = (completed / stats['elapsed']) * 60  # segments per minute
                self.speed_label.setText(f"{speed:.1f} seg/min")
                
                remaining = total - completed
                if speed > 0:
                    eta_seconds = (remaining / speed) * 60  # in seconds
                    if eta_seconds < 60:
                        eta_str = f"{int(eta_seconds)}s"
                    elif eta_seconds < 3600:
                        minutes = int(eta_seconds / 60)
                        seconds = int(eta_seconds % 60)
                        eta_str = f"{minutes}m {seconds}s"
                    else:
                        hours = int(eta_seconds / 3600)
                        minutes = int((eta_seconds % 3600) / 60)
                        eta_str = f"{hours}h {minutes}m"
                    self.eta_label.setText(eta_str)
            
            # Calculate success rate
            if total > 0:
                success_rate = (completed / total) * 100
                self.success_rate_label.setText(f"{success_rate:.1f}%")
    
    def _on_generation_progress(self, current: int, total: int):
        """Handle generation progress update."""
        self.progress_bar.setValue(current)
    
    def _on_generation_finished(self):
        """Handle generation completion."""
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        
        # Hide statistics group
        if hasattr(self, 'stats_group'):
            self.stats_group.setVisible(False)
        
        stats = self.generator.stats if self.generator else None
        if stats:
            self.status_label.setText(
                f"Generation complete: {stats.completed + stats.cached} succeeded "
                f"({stats.cached} cached), {stats.failed} failed"
            )
        
        self._update_ui_state()
        self._update_cache_info()
        
        # Clean up generator
        if self.generator:
            self.generator = None

    def _on_segment_completed(self, index: int):
        """Handle successful segment generation."""
        self._update_segment_list_item(index)
        self._update_ui_state()
    
    def _on_segment_failed(self, index: int, error: str):
        """Handle failed segment generation."""
        if index < len(self.segments):
            self.segments[index].has_error = True
            self.segments[index].error_message = error
        self._update_segment_list_item(index)
        self._update_ui_state()
        logger.error(f"Segment {index} failed: {error}")
    
    def _on_export(self):
        """Export assembled audio with internet speed consideration."""
        ready_segments = [s for s in self.segments if s.is_generated]
        
        if not ready_segments:
            QMessageBox.warning(self, "No Audio", "No generated segments available")
            return
        
        # Check if all segments are already generated
        all_generated = all(s.is_generated for s in self.segments)
        
        if not all_generated:
            # Show warning about internet speed
            segments_to_generate = len([s for s in self.segments if not s.is_generated])
            reply = QMessageBox.question(
                self,
                "Internet Speed Notice",
                f"Some segments need to be generated first ({segments_to_generate} segments).\n\n"
                "This requires an internet connection and speed depends on:\n"
                "• Your internet connection speed\n"
                "• Edge TTS server response time\n"
                "• Number of segments to generate\n\n"
                f"Estimated time: ~{segments_to_generate * 2} seconds (may vary)\n\n"
                "Continue with export?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
            
            # Start generation first
            self.status_label.setText("Generating missing segments before export...")
            self._on_generate()
            
            # Connect to generation finished to start export
            self.generator.generation_finished.connect(self._start_export_after_generation)
            return
        
        # If all segments are generated, proceed with assembly
        self._start_assembly_with_progress()

    def _start_export_after_generation(self):
        """Start export after generation is complete."""
        self.status_label.setText("Generation complete. Starting export...")
        self._start_assembly_with_progress()

    def _start_assembly_with_progress(self, output_path: str = None, format_type: str = "mp3", quality: str = "320"):
        """Start audio assembly with progress tracking."""
        if output_path is None:
            # Generate filename from template
            template = self.template_edit.text().strip()
            if not template:
                template = "khmer_tts_{date}_{time}"
            
            now = datetime.now()
            filename = template.replace("{date}", now.strftime("%Y%m%d"))
            filename = filename.replace("{time}", now.strftime("%H%M%S"))
            
            # Calculate total duration for template
            if self.segments:
                total_sec = max(s.seconds + (s.duration_ms/1000) for s in self.segments if s.is_generated)
                h = int(total_sec // 3600)
                m = int((total_sec % 3600) // 60)
                s = int(total_sec % 60)
                filename = filename.replace("{duration}", f"{h:02d}{m:02d}{s:02d}")
            
            # Sanitize filename
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            if not filename.endswith('.mp3'):
                filename += '.mp3'
            
            output_path = os.path.join(self.output_dir, filename)
            format_type = "mp3"
            quality = "320"
        
        # Check if file exists
        if os.path.exists(output_path):
            reply = QMessageBox.question(
                self,
                "File Exists",
                f"{os.path.basename(output_path)} already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        # Update UI
        self.export_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.status_label.setText(f"Assembling audio: {os.path.basename(output_path)}")
        
        # Store format for assembler
        self._current_export_format = format_type
        self._current_export_quality = quality
        
        # Start assembler thread with progress tracking
        self.assembler = AudioAssembler(
            self.segments,
            output_path,
            crossfade_ms=self.crossfade_spin.value(),
            normalize_master=self.normalize_master_check.isChecked(),
            apply_compression=self.compress_check.isChecked(),
            normalize_per_segment=self.normalize_per_seg_check.isChecked(),
            remove_silence=self.remove_silence_check.isChecked(),
            padding_sec=self.padding_spin.value()
        )
        self.assembler.progress.connect(self._on_assembly_progress)
        self.assembler.status_update.connect(self._on_assembly_status)
        self.assembler.assembly_finished.connect(self._on_assembly_finished)
        self.assembler.assembly_failed.connect(self._on_assembly_failed)
        self.assembler.start()

    def _on_assembly_progress(self, value: int):
        """Handle assembly progress updates."""
        self.progress_bar.setValue(value)
        
        # Show detailed progress based on value
        if value < 40:
            self.status_label.setText(f"Loading segments... {value}%")
        elif value < 50:
            self.status_label.setText("Processing overlaps...")
        elif value < 60:
            self.status_label.setText("Normalizing audio...")
        elif value < 70:
            self.status_label.setText("Applying compression...")
        elif value < 80:
            self.status_label.setText("Adjusting loudness...")
        elif value < 100:
            self.status_label.setText("Exporting final audio...")
        else:
            self.status_label.setText("Finalizing...")

    def _on_assembly_status(self, message: str):
        """Handle assembly status updates."""
        self.status_label.setText(message)
    
    def _on_assembly_finished(self, output_path: str):
        """Handle successful assembly."""
        self.progress_bar.setVisible(False)
        self.export_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        
        self.current_file = output_path
        self.playback.load(output_path)
        
        # Enable playback controls
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.position_slider.setEnabled(True)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        
        # Calculate assembly time if available
        if hasattr(self.assembler, 'assembly_time') and self.assembler.assembly_time:
            assembly_time = self.assembler.assembly_time
            time_msg = f" in {assembly_time:.1f}s"
        else:
            time_msg = ""
        
        self.status_label.setText(f"✅ Saved: {os.path.basename(output_path)} ({file_size:.1f} MB){time_msg}")
        
        # Ask to open folder
        reply = QMessageBox.question(
            self,
            "Complete",
            f"Audio saved successfully{time_msg}.\n\nOpen containing folder?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._open_folder(self.output_dir)
    
    def _on_assembly_failed(self, error: str):
        """Handle assembly failure."""
        self.progress_bar.setVisible(False)
        self.export_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        
        self.status_label.setText(f"❌ Assembly failed")
        QMessageBox.critical(self, "Assembly Failed", error)
        logger.error(f"Assembly failed: {error}")
    
    def _on_clear_errors(self):
        """Clear error states from segments."""
        for segment in self.segments:
            segment.has_error = False
            segment.error_message = ""
        self._refresh_segment_list()
        self._update_ui_state()
        self.status_label.setText("Errors cleared")
    
    def _on_clear_all(self):
        """Clear all generated data."""
        reply = QMessageBox.question(
            self, "Clear All",
            "Clear all generated audio data?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            for segment in self.segments:
                segment.is_generated = False
                segment.has_error = False
                segment.error_message = ""
                segment.audio_path = None
            self._refresh_segment_list()
            self._update_ui_state()
            self.status_label.setText("All data cleared")
    
    def _on_clear_cache(self):
        """Clear the TTS cache."""
        reply = QMessageBox.question(
            self, "Clear Cache",
            "Clear all cached TTS files?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                for f in CACHE_DIR.glob("*.mp3"):
                    f.unlink()
                for f in CACHE_DIR.glob("*.json"):
                    f.unlink()
                self._update_cache_info()
                self.status_label.setText("Cache cleared")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to clear cache:\n{str(e)}")
    
    def _on_browse_output_dir(self):
        """Browse for output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_dir
        )
        if dir_path:
            self.output_dir = dir_path
            self.dir_label.setText(dir_path)
            self._save_settings()
    
    def _on_play(self):
        """Start playback."""
        if self.playback and self.playback.play():
            self.play_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.status_label.setText("▶️ Playing")
    
    def _on_pause(self):
        """Pause playback."""
        if self.playback:
            self.playback.pause()
            self.play_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setText("⏸️ Paused")
    
    def _on_stop(self):
        """Stop playback."""
        if self.playback:
            self.playback.stop()
            self.play_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.position_slider.setValue(0)
            self.time_label.setText("00:00 / 00:00")
            self.status_label.setText("⏹️ Stopped")
    
    def _on_volume_changed(self, volume: int):
        """Handle volume change."""
        if self.playback:
            self.playback.set_volume(volume)
    
    def _on_position_changed(self, position: int):
        """Update position slider and time display."""
        if not self.position_slider.isSliderDown():
            self.position_slider.setValue(position)
        
        if self.playback and self.playback.duration > 0:
            current = self._format_time(position // 1000)
            total = self._format_time(self.playback.duration // 1000)
            self.time_label.setText(f"{current} / {total}")
    
    def _on_duration_changed(self, duration: int):
        """Update duration when media loads."""
        self.position_slider.setRange(0, duration)
        if self.playback:
            self.playback.duration = duration
    
    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState):
        """Handle playback state changes."""
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self._on_stop()
    
    def _on_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Khmer TTS Studio",
            f"""<h2>Khmer TTS Studio</h2>
            <p>High Quality Edition</p>
            <p>Professional text-to-speech application</p>
            <p><b>Features:</b><br>
            • High-quality Edge TTS voices<br>
            • Multiple languages: Khmer, English, Chinese<br>
            • Parallel generation with caching<br>
            • Professional audio mastering<br>
            • Khmer text preprocessing<br>
            • Project save/load<br>
            • Internet speed aware export with progress tracking</p>
            <p><b>Audio Quality:</b><br>
            • {SAMPLE_RATE/1000:.0f}kHz / {CHANNELS}ch / {DEFAULT_BITRATE}</p>
            <p><b>Cache:</b><br>
            • Location: {CACHE_DIR}<br>
            • Max size: {MAX_CACHE_SIZE_GB} GB</p>"""
        )

    def _format_time(self, seconds: int) -> str:
        """Format seconds as HH:MM:SS."""
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def _open_folder(self, path: str):
        """Open folder in system file explorer."""
        try:
            if sys.platform == 'win32':
                os.startfile(path)  # ← Windows
            elif sys.platform == 'darwin':
                subprocess.run(['open', path])  # ← macOS
            else:
                subprocess.run(['xdg-open', path])  # ← Linux
        except Exception as e:
            logger.warning(f"Could not open folder: {e}")
    
    def closeEvent(self, event):
        """Clean up on application close."""
        logger.info("Shutting down Khmer TTS Studio")
        
        # Stop generation if running
        if self.generator and self.generator.isRunning():
            self.generator.stop()
            self.generator.wait(3000)
        
        # Stop assembly if running
        if self.assembler and self.assembler.isRunning():
            self.assembler.stop()
            self.assembler.wait(3000)
        
        # Stop playback
        if self.playback:
            self.playback.stop()
        
        # Save settings
        self._save_settings()
        
        event.accept()


# ==================== MAIN ENTRY POINT ====================

def main():
    """Application entry point."""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("Khmer TTS Studio")
    app.setApplicationDisplayName("Khmer TTS Studio – High Quality Edition")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = KhmerTTSStudio()
    window.show()
    
    logger.info("Khmer TTS Studio started")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()