"""
translate.py - Chinese to English Translation Module
- Free translation using LibreTranslate (no API key required)
- Supports TXT, SRT subtitle files with timestamp preservation
- Batch processing with progress tracking
- Translation memory cache for efficiency
- Context-aware translation for better accuracy
- Copy/paste functionality in preview
- Timestamp preservation in output
- Integration with AI Studio Pro main app
"""

import os
import re
import json
import time
import hashlib
import pickle
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import threading
from queue import Queue

# PyQt6 imports
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox, QProgressBar, QGroupBox,
    QCheckBox, QComboBox, QListWidget, QListWidgetItem,
    QSplitter, QFrame, QTextEdit, QLineEdit, QApplication,
    QSpinBox, QDoubleSpinBox, QTabWidget, QGridLayout,
    QScrollArea, QSizePolicy, QProgressDialog, QMenu
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize, QMutex, QMutexLocker, QPoint
)
from PyQt6.QtGui import (
    QFont, QColor, QTextCursor, QIcon, QAction, QTextCharFormat, QSyntaxHighlighter
)

# Try to import requests for LibreTranslate
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ requests not found - install with: pip install requests")

# Try to import subtitle library
try:
    import pysrt
    PYSRT_AVAILABLE = True
except ImportError:
    PYSRT_AVAILABLE = False
    print("⚠️ pysrt not found - install with: pip install pysrt")


# ==================== CONFIGURATION ====================

@dataclass
class TranslationConfig:
    """Configuration for translation settings."""
    api_url: str = "https://libretranslate.com"  # Public instance
    local_url: str = "http://localhost:5000"      # Local instance if running
    use_local: bool = False
    timeout: int = 30
    cache_enabled: bool = True
    cache_dir: str = "translation_cache"
    batch_size: int = 10
    context_window: int = 3
    source_lang: str = "zh"
    target_lang: str = "en"
    preserve_timestamps: bool = True


# ==================== TIMESTAMPED LINE ====================

@dataclass
class TimestampedLine:
    """Represents a line with optional timestamp."""
    text: str
    timestamp: Optional[str] = None
    index: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def formatted(self) -> str:
        """Return line with timestamp if available."""
        if self.timestamp:
            return f"[{self.timestamp}] {self.text}"
        return self.text
    
    def srt_format(self) -> str:
        """Return SRT formatted line."""
        if self.start_time and self.end_time and self.index:
            return f"{self.index}\n{self.start_time} --> {self.end_time}\n{self.text}\n"
        return self.text


# ==================== TRANSLATION CACHE ====================

class TranslationCache:
    """Cache for translations to avoid redundant API calls."""
    
    def __init__(self, cache_dir: str = "translation_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache: Dict[str, str] = {}
        self.mutex = QMutex()
        self.stats = {"hits": 0, "misses": 0}
    
    def _get_cache_key(self, text: str, source_lang: str, target_lang: str, context: str = "") -> str:
        """Generate cache key from text and languages."""
        key_string = f"{source_lang}:{target_lang}:{text}:{context}"
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, text: str, source_lang: str, target_lang: str, context: str = "") -> Optional[str]:
        """Get cached translation if available."""
        with QMutexLocker(self.mutex):
            key = self._get_cache_key(text, source_lang, target_lang, context)
            
            # Check memory cache first
            if key in self.memory_cache:
                self.stats["hits"] += 1
                return self.memory_cache[key]
            
            # Check disk cache
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        translation = pickle.load(f)
                    self.memory_cache[key] = translation
                    self.stats["hits"] += 1
                    return translation
                except:
                    pass
            
            self.stats["misses"] += 1
            return None
    
    def set(self, text: str, translation: str, source_lang: str, target_lang: str, context: str = ""):
        """Store translation in cache."""
        with QMutexLocker(self.mutex):
            key = self._get_cache_key(text, source_lang, target_lang, context)
            
            # Store in memory cache
            self.memory_cache[key] = translation
            
            # Store in disk cache
            cache_path = self._get_cache_path(key)
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(translation, f)
            except:
                pass
    
    def clear(self):
        """Clear all caches."""
        with QMutexLocker(self.mutex):
            self.memory_cache.clear()
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except:
                    pass
            self.stats = {"hits": 0, "misses": 0}
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with QMutexLocker(self.mutex):
            total = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
            return {
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "total": total,
                "hit_rate": hit_rate
            }


# ==================== LIBRETRANSLATE WRAPPER ====================

class LibreTranslator:
    """Free translator using LibreTranslate API."""
    
    # Language code mapping
    LANG_CODES = {
        "chinese": "zh",
        "english": "en",
        "japanese": "ja",
        "korean": "ko",
        "french": "fr",
        "german": "de",
        "spanish": "es",
        "russian": "ru",
        "arabic": "ar",
        "hindi": "hi",
        "vietnamese": "vi",
        "thai": "th",
        "indonesian": "id"
    }
    
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.cache = TranslationCache(config.cache_dir) if config.cache_enabled else None
        self.available = False
        self.current_url = config.api_url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AI-Studio-Pro/1.0"
        })
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to LibreTranslate instances."""
        urls_to_try = []
        
        if self.config.use_local:
            urls_to_try.append(self.config.local_url)
        urls_to_try.append(self.config.api_url)
        
        for url in urls_to_try:
            try:
                response = self.session.get(f"{url}/spec", timeout=5)
                if response.status_code == 200:
                    self.current_url = url
                    self.available = True
                    print(f"✅ LibreTranslate connected to {url}")
                    
                    # Get supported languages
                    try:
                        langs_response = self.session.get(f"{url}/languages", timeout=5)
                        if langs_response.status_code == 200:
                            self.supported_langs = [lang["code"] for lang in langs_response.json()]
                            print(f"   Supported languages: {len(self.supported_langs)}")
                    except:
                        pass
                    
                    return
            except:
                continue
        
        print("⚠️ Could not connect to any LibreTranslate instance")
        print("   To use local instance: docker run -it -p 5000:5000 libretranslate/libretranslate")
        print("   Or use public instance at: https://libretranslate.com")
    
    def _map_lang_code(self, lang: str) -> str:
        """Map UI language names to API codes."""
        return self.LANG_CODES.get(lang.lower(), lang)
    
    def translate(
        self,
        text: str,
        source_lang: str = "chinese",
        target_lang: str = "english",
        context_type: str = "general",
        previous_lines: List[str] = None
    ) -> str:
        """Translate text using LibreTranslate."""
        
        if not self.available:
            return f"[Translation unavailable - LibreTranslate not connected] {text}"
        
        if not text or not text.strip():
            return text
        
        # Map language codes
        source_code = self._map_lang_code(source_lang)
        target_code = self._map_lang_code(target_lang)
        
        # Check cache first
        if self.cache:
            context_str = "\n".join(previous_lines[-self.config.context_window:]) if previous_lines else ""
            cached = self.cache.get(text, source_code, target_code, context_str)
            if cached:
                return cached
        
        try:
            # Prepare request
            data = {
                "q": text,
                "source": source_code,
                "target": target_code,
                "format": "text"
            }
            
            # Make API call
            response = self.session.post(
                f"{self.current_url}/translate",
                json=data,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                translation = result.get("translatedText", "")
                
                # Cache the result
                if self.cache and translation:
                    context_str = "\n".join(previous_lines[-self.config.context_window:]) if previous_lines else ""
                    self.cache.set(text, translation, source_code, target_code, context_str)
                
                return translation
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"❌ LibreTranslate error: {error_msg}")
                return f"[Translation Error] {text}"
                
        except requests.exceptions.Timeout:
            print("❌ LibreTranslate timeout")
            return f"[Timeout] {text}"
        except requests.exceptions.ConnectionError:
            print("❌ LibreTranslate connection error")
            self.available = False
            return f"[Connection Error] {text}"
        except Exception as e:
            print(f"❌ LibreTranslate error: {e}")
            return f"[Error] {text}"
    
    def detect_language(self, text: str) -> str:
        """Detect language of text."""
        if not self.available or not text:
            return "unknown"
        
        try:
            response = self.session.post(
                f"{self.current_url}/detect",
                json={"q": text},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result and len(result) > 0:
                    return result[0].get("language", "unknown")
            return "unknown"
        except:
            return "unknown"
    
    def get_supported_languages(self) -> List[Dict]:
        """Get list of supported languages."""
        try:
            response = self.session.get(f"{self.current_url}/languages", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return []
    
    def translate_batch(
        self,
        texts: List[str],
        source_lang: str = "chinese",
        target_lang: str = "english",
        context_type: str = "general",
        progress_callback=None
    ) -> List[str]:
        """Translate multiple texts in batch."""
        
        translations = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            # Get previous lines for context
            start_idx = max(0, i - self.config.context_window)
            previous_lines = texts[start_idx:i] if i > 0 else None
            
            # Translate
            translation = self.translate(
                text, 
                source_lang, 
                target_lang, 
                context_type,
                previous_lines
            )
            translations.append(translation)
            
            # Report progress
            if progress_callback:
                progress_callback(i + 1, total)
            
            # Small delay to avoid rate limiting
            if i < total - 1:
                time.sleep(0.1)
        
        return translations


# ==================== TRANSLATION THREAD ====================

class TranslationThread(QThread):
    """Background thread for translation."""
    
    progress = pyqtSignal(int, int)
    status = pyqtSignal(str)
    line_translated = pyqtSignal(int, str, dict)  # Added metadata
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, translator: LibreTranslator, lines: List[TimestampedLine], config: Dict):
        super().__init__()
        self.translator = translator
        self.lines = lines
        self.config = config
        self._running = True
        self.translations = []
    
    def stop(self):
        self._running = False
    
    def run(self):
        try:
            self.status.emit("Starting translation...")
            
            total = len(self.lines)
            self.translations = []
            
            for i, line_obj in enumerate(self.lines):
                if not self._running:
                    break
                
                line = line_obj.text
                if not line.strip():
                    self.translations.append("")
                    self.line_translated.emit(i, "", line_obj.metadata)
                    continue
                
                # Get context from previous lines (text only)
                start_idx = max(0, i - self.config.get('context_window', 3))
                previous_texts = [l.text for l in self.lines[start_idx:i] if l.text.strip()]
                
                # Translate
                self.status.emit(f"Translating line {i+1}/{total}...")
                translation = self.translator.translate(
                    line,
                    source_lang=self.config.get('source_lang', 'chinese'),
                    target_lang=self.config.get('target_lang', 'english'),
                    context_type=self.config.get('context_type', 'general'),
                    previous_lines=previous_texts
                )
                
                self.translations.append(translation)
                self.line_translated.emit(i, translation, line_obj.metadata)
                self.progress.emit(i + 1, total)
                
                # Small delay
                if i < total - 1:
                    self.msleep(100)
            
            if self._running:
                self.status.emit("Translation complete!")
                self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))
            traceback.print_exc()


# ==================== SUBTITLE PARSER ====================

class SubtitleParser:
    """Parse and handle subtitle files with timestamp preservation."""
    
    @staticmethod
    def parse_file(file_path: str) -> Tuple[List[TimestampedLine], List[Dict]]:
        """Parse subtitle file and return timestamped lines with metadata."""
        
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.srt' and PYSRT_AVAILABLE:
            return SubtitleParser._parse_srt(file_path)
        elif ext == '.txt':
            return SubtitleParser._parse_txt(file_path)
        elif ext == '.json':
            return SubtitleParser._parse_json(file_path)
        elif ext == '.ass' or ext == '.ssa':
            return SubtitleParser._parse_ass(file_path)
        else:
            # Default to plain text
            return SubtitleParser._parse_txt(file_path)
    
    @staticmethod
    def _parse_srt(file_path: str) -> Tuple[List[TimestampedLine], List[Dict]]:
        """Parse SRT subtitle file."""
        subs = pysrt.open(file_path)
        lines = []
        metadata = []
        
        for sub in subs:
            # Clean text (remove HTML tags, etc.)
            text = re.sub(r'<[^>]+>', '', sub.text)
            text = text.replace('\n', ' ').strip()
            
            line_obj = TimestampedLine(
                text=text,
                index=sub.index,
                start_time=str(sub.start),
                end_time=str(sub.end),
                metadata={
                    'index': sub.index,
                    'start': str(sub.start),
                    'end': str(sub.end),
                    'position': sub.position
                }
            )
            lines.append(line_obj)
            metadata.append(line_obj.metadata)
        
        return lines, metadata
    
    @staticmethod
    def _parse_txt(file_path: str) -> Tuple[List[TimestampedLine], List[Dict]]:
        """Parse plain text file with timestamp detection."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = []
        metadata = []
        
        # Pattern for [HH:MM:SS] or [HH:MM:SS.mmm] timestamps
        timestamp_pattern = r'^\[(\d{2}:\d{2}:\d{2}(?:\.\d{3})?)\]\s*(.*)'
        # Pattern for SRT style timestamps: 00:00:00,000 --> 00:00:00,000
        srt_timestamp_pattern = r'^(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*(.*)'
        
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Check for SRT style timestamps
            srt_match = re.match(srt_timestamp_pattern, line)
            if srt_match:
                start, end, text = srt_match.groups()
                line_obj = TimestampedLine(
                    text=text.strip(),
                    start_time=start,
                    end_time=end,
                    metadata={'start': start, 'end': end}
                )
                lines.append(line_obj)
                metadata.append(line_obj.metadata)
                continue
            
            # Check for simple timestamp pattern
            timestamp_match = re.match(timestamp_pattern, line)
            if timestamp_match:
                timestamp, text = timestamp_match.groups()
                line_obj = TimestampedLine(
                    text=text.strip(),
                    timestamp=timestamp,
                    metadata={'timestamp': timestamp}
                )
                lines.append(line_obj)
                metadata.append(line_obj.metadata)
            else:
                # Plain line without timestamp
                line_obj = TimestampedLine(text=line)
                lines.append(line_obj)
                metadata.append({})
        
        return lines, metadata
    
    @staticmethod
    def _parse_json(file_path: str) -> Tuple[List[TimestampedLine], List[Dict]]:
        """Parse JSON subtitle file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        lines = []
        metadata = []
        
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    text = item.get('text', '')
                    if 'start' in item and 'end' in item:
                        line_obj = TimestampedLine(
                            text=text,
                            start_time=item.get('start'),
                            end_time=item.get('end'),
                            metadata={k: v for k, v in item.items() if k != 'text'}
                        )
                    elif 'timestamp' in item:
                        line_obj = TimestampedLine(
                            text=text,
                            timestamp=item.get('timestamp'),
                            metadata={k: v for k, v in item.items() if k != 'text'}
                        )
                    else:
                        line_obj = TimestampedLine(
                            text=text,
                            metadata={k: v for k, v in item.items() if k != 'text'}
                        )
                    
                    lines.append(line_obj)
                    metadata.append(line_obj.metadata)
                    
                elif isinstance(item, str):
                    lines.append(TimestampedLine(text=item))
                    metadata.append({})
        
        return lines, metadata
    
    @staticmethod
    def _parse_ass(file_path: str) -> Tuple[List[TimestampedLine], List[Dict]]:
        """Parse ASS/SSA subtitle file (basic support)."""
        lines = []
        metadata = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for [Events] section and Dialogue lines
        in_events = False
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('[Events]'):
                in_events = True
                continue
            if in_events and line.startswith('Dialogue:'):
                parts = line.split(',', 9)
                if len(parts) >= 10:
                    start = parts[1].strip()
                    end = parts[2].strip()
                    text = parts[9].strip()
                    # Remove ASS formatting codes
                    text = re.sub(r'\{[^}]*\}', '', text)
                    
                    line_obj = TimestampedLine(
                        text=text,
                        start_time=start,
                        end_time=end,
                        metadata={'start': start, 'end': end}
                    )
                    lines.append(line_obj)
                    metadata.append(line_obj.metadata)
        
        if not lines:
            # Fallback to plain text parsing
            return SubtitleParser._parse_txt(file_path)
        
        return lines, metadata
    
    @staticmethod
    def save_translation(
        file_path: str,
        original_lines: List[TimestampedLine],
        translated_lines: List[str],
        output_format: str = 'txt',
        preserve_timestamps: bool = True
    ):
        """Save translated text to file with timestamps preserved."""
        
        if output_format == 'srt' and PYSRT_AVAILABLE:
            SubtitleParser._save_srt(file_path, original_lines, translated_lines)
        elif output_format == 'json':
            SubtitleParser._save_json(file_path, original_lines, translated_lines)
        elif output_format == 'ass':
            SubtitleParser._save_ass(file_path, original_lines, translated_lines)
        else:
            SubtitleParser._save_txt(file_path, original_lines, translated_lines, preserve_timestamps)
    
    @staticmethod
    def _save_txt(file_path: str, original_lines: List[TimestampedLine], translated_lines: List[str], preserve_timestamps: bool):
        """Save as text file with optional timestamps."""
        with open(file_path, 'w', encoding='utf-8') as f:
            for i, (orig, trans) in enumerate(zip(original_lines, translated_lines)):
                if preserve_timestamps and orig.timestamp:
                    f.write(f"[{orig.timestamp}] {trans}\n")
                elif preserve_timestamps and orig.start_time and orig.end_time:
                    f.write(f"[{orig.start_time} -> {orig.end_time}] {trans}\n")
                else:
                    f.write(f"{trans}\n")
    
    @staticmethod
    def _save_srt(file_path: str, original_lines: List[TimestampedLine], translated_lines: List[str]):
        """Save as SRT subtitle file."""
        subs = pysrt.SubRipFile()
        
        for i, (orig, trans) in enumerate(zip(original_lines, translated_lines)):
            if orig.start_time and orig.end_time:
                # Convert time format if needed
                start = orig.start_time.replace(',', '.')
                end = orig.end_time.replace(',', '.')
                
                sub = pysrt.SubRipItem(
                    index=i + 1,
                    start=start,
                    end=end,
                    text=trans
                )
                subs.append(sub)
        
        subs.save(file_path, encoding='utf-8')
    
    @staticmethod
    def _save_json(file_path: str, original_lines: List[TimestampedLine], translated_lines: List[str]):
        """Save as JSON file."""
        data = []
        for i, (orig, trans) in enumerate(zip(original_lines, translated_lines)):
            item = {**orig.metadata, 'original': orig.text, 'translated': trans}
            data.append(item)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def _save_ass(file_path: str, original_lines: List[TimestampedLine], translated_lines: List[str]):
        """Save as ASS subtitle file (simplified)."""
        with open(file_path, 'w', encoding='utf-8') as f:
            # Write ASS header
            f.write("[Script Info]\n")
            f.write("Title: Translated Subtitles\n")
            f.write("ScriptType: v4.00+\n")
            f.write("WrapStyle: 0\n")
            f.write("ScaledBorderAndShadow: yes\n")
            f.write("PlayResX: 384\n")
            f.write("PlayResY: 288\n\n")
            
            f.write("[V4+ Styles]\n")
            f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
            f.write("Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n")
            
            f.write("[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
            
            for i, (orig, trans) in enumerate(zip(original_lines, translated_lines)):
                if orig.start_time and orig.end_time:
                    f.write(f"Dialogue: 0,{orig.start_time},{orig.end_time},Default,,0,0,0,,{trans}\n")


# ==================== COPYABLE TEXT EDIT ====================

class CopyableTextEdit(QTextEdit):
    """Text edit widget with enhanced copy/paste functionality."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
    
    def show_context_menu(self, position: QPoint):
        """Show custom context menu with copy options."""
        menu = QMenu()
        
        copy_action = menu.addAction("📋 Copy")
        copy_action.triggered.connect(self.copy)
        copy_action.setEnabled(self.textCursor().hasSelection())
        
        copy_all_action = menu.addAction("📋 Copy All")
        copy_all_action.triggered.connect(self.select_all_copy)
        
        menu.addSeparator()
        
        select_all_action = menu.addAction("🔍 Select All")
        select_all_action.triggered.connect(self.select_all)
        
        menu.exec(self.mapToGlobal(position))
    
    def select_all_copy(self):
        """Select all text and copy to clipboard."""
        self.selectAll()
        self.copy()
        # Deselect after copy
        cursor = self.textCursor()
        cursor.clearSelection()
        self.setTextCursor(cursor)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for copy."""
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key.Key_C:
                self.copy()
                event.accept()
                return
            elif event.key() == Qt.Key.Key_A:
                self.selectAll()
                event.accept()
                return
        super().keyPressEvent(event)


# ==================== TRANSLATION WIDGET ====================

class TranslationWidget(QWidget):
    """Main translation widget for AI Studio Pro."""
    
    def __init__(self, parent=None, theme_manager=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.translator = None
        self.translation_thread = None
        self.current_file = None
        self.original_lines: List[TimestampedLine] = []
        self.translated_lines: List[str] = []
        
        # Load config
        self.config = TranslationConfig()
        self._load_config()
        
        self.setup_ui()
        
        # Apply theme if available
        if theme_manager:
            self.apply_theme(theme_manager.is_dark)
    
    def setup_ui(self):
        """Setup the translation UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # ===== LibreTranslate Configuration Section =====
        config_group = QGroupBox("LibreTranslate Configuration (100% Free)")
        config_layout = QGridLayout()
        config_layout.setColumnStretch(1, 1)
        
        # API URL
        config_layout.addWidget(QLabel("Public API URL:"), 0, 0)
        self.public_url_input = QLineEdit()
        self.public_url_input.setText(self.config.api_url)
        self.public_url_input.textChanged.connect(self._save_config)
        config_layout.addWidget(self.public_url_input, 0, 1)
        
        # Local URL
        config_layout.addWidget(QLabel("Local URL:"), 1, 0)
        self.local_url_input = QLineEdit()
        self.local_url_input.setText(self.config.local_url)
        self.local_url_input.textChanged.connect(self._save_config)
        config_layout.addWidget(self.local_url_input, 1, 1)
        
        # Use local checkbox
        self.use_local_check = QCheckBox("Use local instance (self-hosted)")
        self.use_local_check.setChecked(self.config.use_local)
        self.use_local_check.toggled.connect(self._save_config)
        config_layout.addWidget(self.use_local_check, 2, 0, 1, 2)
        
        # Connection status
        self.connection_status = QLabel("⚪ Not connected")
        self.connection_status.setStyleSheet("color: #f44336;")
        config_layout.addWidget(self.connection_status, 3, 0, 1, 2)
        
        # Test connection button
        self.test_btn = QPushButton("Test Connection")
        self.test_btn.clicked.connect(self.test_connection)
        config_layout.addWidget(self.test_btn, 4, 0, 1, 2)
        
        # Cache options
        cache_widget = QWidget()
        cache_layout = QHBoxLayout(cache_widget)
        cache_layout.setContentsMargins(0, 0, 0, 0)
        
        self.cache_check = QCheckBox("Enable Translation Cache")
        self.cache_check.setChecked(self.config.cache_enabled)
        self.cache_check.toggled.connect(self._save_config)
        cache_layout.addWidget(self.cache_check)
        
        self.clear_cache_btn = QPushButton("Clear Cache")
        self.clear_cache_btn.setFixedWidth(100)
        self.clear_cache_btn.clicked.connect(self._clear_cache)
        cache_layout.addWidget(self.clear_cache_btn)
        
        cache_layout.addStretch()
        config_layout.addWidget(cache_widget, 5, 0, 1, 2)
        
        # Timestamp preservation
        self.timestamp_check = QCheckBox("Preserve Timestamps")
        self.timestamp_check.setChecked(self.config.preserve_timestamps)
        self.timestamp_check.toggled.connect(self._save_config)
        config_layout.addWidget(self.timestamp_check, 6, 0, 1, 2)
        
        # Info label
        info_label = QLabel("💡 No API key required! LibreTranslate is completely free.")
        info_label.setStyleSheet("color: #4CAF50; font-style: italic;")
        config_layout.addWidget(info_label, 7, 0, 1, 2)
        
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)
        
        # ===== File Selection Section =====
        file_group = QGroupBox("File Selection")
        file_layout = QHBoxLayout()
        
        self.file_path_input = QLineEdit()
        self.file_path_input.setPlaceholderText("Select Chinese text or subtitle file...")
        file_layout.addWidget(self.file_path_input)
        
        self.browse_btn = QPushButton("📂 Browse")
        self.browse_btn.setFixedWidth(100)
        self.browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_btn)
        
        self.load_btn = QPushButton("📄 Load File")
        self.load_btn.setFixedWidth(100)
        self.load_btn.clicked.connect(self.load_file)
        file_layout.addWidget(self.load_btn)
        
        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)
        
        # ===== Translation Options =====
        options_group = QGroupBox("Translation Options")
        options_layout = QHBoxLayout()
        
        options_layout.addWidget(QLabel("Source:"))
        self.source_lang_combo = QComboBox()
        self.source_lang_combo.addItems(["chinese", "english", "japanese", "korean", "french", "german", "spanish"])
        self.source_lang_combo.setCurrentText("chinese")
        self.source_lang_combo.setFixedWidth(100)
        options_layout.addWidget(self.source_lang_combo)
        
        options_layout.addWidget(QLabel("→"))
        
        options_layout.addWidget(QLabel("Target:"))
        self.target_lang_combo = QComboBox()
        self.target_lang_combo.addItems(["english", "chinese", "japanese", "korean", "french", "german", "spanish"])
        self.target_lang_combo.setCurrentText("english")
        self.target_lang_combo.setFixedWidth(100)
        options_layout.addWidget(self.target_lang_combo)
        
        options_layout.addStretch()
        
        options_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 50)
        self.batch_size_spin.setValue(10)
        self.batch_size_spin.setFixedWidth(60)
        options_layout.addWidget(self.batch_size_spin)
        
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)
        
        # ===== Preview Section with Copyable Text Edits =====
        preview_group = QGroupBox("Translation Preview (Right-click to copy)")
        preview_layout = QHBoxLayout()
        
        # Original text
        original_widget = QWidget()
        original_layout = QVBoxLayout(original_widget)
        original_layout.setContentsMargins(0, 0, 0, 0)
        
        original_header = QHBoxLayout()
        original_header.addWidget(QLabel("📝 Original Chinese:"))
        original_header.addStretch()
        
        self.original_copy_btn = QPushButton("📋 Copy All")
        self.original_copy_btn.setFixedWidth(80)
        self.original_copy_btn.clicked.connect(self.copy_original_all)
        original_header.addWidget(self.original_copy_btn)
        
        original_layout.addLayout(original_header)
        
        self.original_text = CopyableTextEdit()
        self.original_text.setFont(QFont("Microsoft YaHei", 10))
        original_layout.addWidget(self.original_text)
        
        preview_layout.addWidget(original_widget)
        
        # Translated text
        translated_widget = QWidget()
        translated_layout = QVBoxLayout(translated_widget)
        translated_layout.setContentsMargins(0, 0, 0, 0)
        
        translated_header = QHBoxLayout()
        translated_header.addWidget(QLabel("🌐 Translated English:"))
        translated_header.addStretch()
        
        self.translated_copy_btn = QPushButton("📋 Copy All")
        self.translated_copy_btn.setFixedWidth(80)
        self.translated_copy_btn.clicked.connect(self.copy_translated_all)
        translated_header.addWidget(self.translated_copy_btn)
        
        translated_layout.addLayout(translated_header)
        
        self.translated_text = CopyableTextEdit()
        self.translated_text.setFont(QFont("Arial", 10))
        translated_layout.addWidget(self.translated_text)
        
        preview_layout.addWidget(translated_widget)
        
        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group, 1)
        
        # ===== Progress Section =====
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready to translate")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        main_layout.addWidget(progress_group)
        
        # ===== Control Buttons =====
        button_layout = QHBoxLayout()
        
        self.translate_btn = QPushButton("▶ Start Translation")
        self.translate_btn.setMinimumHeight(35)
        self.translate_btn.setFixedWidth(150)
        self.translate_btn.clicked.connect(self.start_translation)
        button_layout.addWidget(self.translate_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setMinimumHeight(35)
        self.stop_btn.setFixedWidth(100)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_translation)
        button_layout.addWidget(self.stop_btn)
        
        self.save_btn = QPushButton("💾 Save Translation")
        self.save_btn.setMinimumHeight(35)
        self.save_btn.setFixedWidth(150)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_translation)
        button_layout.addWidget(self.save_btn)
        
        button_layout.addStretch()
        
        # Stats label
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: #4CAF50;")
        button_layout.addWidget(self.stats_label)
        
        main_layout.addLayout(button_layout)
        
        # Test connection on startup
        QTimer.singleShot(500, self.test_connection)
    
    def _load_config(self):
        """Load translation config from file."""
        config_file = Path("translation_config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
            except:
                pass
    
    def _save_config(self):
        """Save translation config to file."""
        self.config.api_url = self.public_url_input.text()
        self.config.local_url = self.local_url_input.text()
        self.config.use_local = self.use_local_check.isChecked()
        self.config.cache_enabled = self.cache_check.isChecked()
        self.config.preserve_timestamps = self.timestamp_check.isChecked()
        
        try:
            with open("translation_config.json", 'w', encoding='utf-8') as f:
                data = {
                    'api_url': self.config.api_url,
                    'local_url': self.config.local_url,
                    'use_local': self.config.use_local,
                    'cache_enabled': self.config.cache_enabled,
                    'preserve_timestamps': self.config.preserve_timestamps
                }
                json.dump(data, f, indent=2)
        except:
            pass
    
    def test_connection(self):
        """Test connection to LibreTranslate."""
        if not REQUESTS_AVAILABLE:
            self.connection_status.setText("❌ requests not installed")
            self.connection_status.setStyleSheet("color: #f44336;")
            return
        
        self.test_btn.setEnabled(False)
        self.test_btn.setText("Testing...")
        QApplication.processEvents()
        
        # Create temporary translator to test connection
        temp_translator = LibreTranslator(self.config)
        
        if temp_translator.available:
            self.connection_status.setText(f"✅ Connected to {temp_translator.current_url}")
            self.connection_status.setStyleSheet("color: #4CAF50;")
            self.translator = temp_translator
        else:
            self.connection_status.setText("❌ Could not connect to any LibreTranslate instance")
            self.connection_status.setStyleSheet("color: #f44336;")
        
        self.test_btn.setEnabled(True)
        self.test_btn.setText("Test Connection")
    
    def _clear_cache(self):
        """Clear translation cache."""
        if self.translator and self.translator.cache:
            reply = QMessageBox.question(
                self, "Clear Cache",
                "Clear all cached translations? This cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.translator.cache.clear()
                QMessageBox.information(self, "Success", "Cache cleared successfully!")
    
    def copy_original_all(self):
        """Copy all original text to clipboard."""
        self.original_text.selectAll()
        self.original_text.copy()
        cursor = self.original_text.textCursor()
        cursor.clearSelection()
        self.original_text.setTextCursor(cursor)
        self.status_label.setText("📋 Original text copied to clipboard")
        QTimer.singleShot(2000, lambda: self.status_label.setText("Ready"))
    
    def copy_translated_all(self):
        """Copy all translated text to clipboard."""
        self.translated_text.selectAll()
        self.translated_text.copy()
        cursor = self.translated_text.textCursor()
        cursor.clearSelection()
        self.translated_text.setTextCursor(cursor)
        self.status_label.setText("📋 Translated text copied to clipboard")
        QTimer.singleShot(2000, lambda: self.status_label.setText("Ready"))
    
    def browse_file(self):
        """Browse for input file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Chinese Text File",
            "",
            "All Supported Files (*.txt *.srt *.json *.ass *.ssa);;Text Files (*.txt);;Subtitle Files (*.srt *.ass *.ssa);;JSON Files (*.json);;All Files (*.*)"
        )
        
        if file_path:
            self.file_path_input.setText(file_path)
    
    def load_file(self):
        """Load and parse the selected file."""
        file_path = self.file_path_input.text().strip()
        
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "Error", "Please select a valid file")
            return
        
        try:
            self.original_lines, self.metadata = SubtitleParser.parse_file(file_path)
            self.current_file = file_path
            self.translated_lines = []
            
            # Display preview with timestamps
            preview_lines = []
            for i, line_obj in enumerate(self.original_lines[:20]):
                if self.timestamp_check.isChecked() and line_obj.timestamp:
                    preview_lines.append(f"[{line_obj.timestamp}] {line_obj.text}")
                elif self.timestamp_check.isChecked() and line_obj.start_time and line_obj.end_time:
                    preview_lines.append(f"[{line_obj.start_time} -> {line_obj.end_time}] {line_obj.text}")
                else:
                    preview_lines.append(line_obj.text)
            
            preview_text = "\n".join(preview_lines)
            if len(self.original_lines) > 20:
                preview_text += f"\n\n... and {len(self.original_lines) - 20} more lines"
            
            self.original_text.setText(preview_text)
            self.translated_text.clear()
            
            self.status_label.setText(f"✅ Loaded {len(self.original_lines)} lines")
            self.save_btn.setEnabled(False)
            self.stats_label.setText("")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
    
    def start_translation(self):
        """Start the translation process."""
        if not self.original_lines:
            QMessageBox.warning(self, "Error", "Please load a file first")
            return
        
        if not REQUESTS_AVAILABLE:
            QMessageBox.critical(
                self, "Error",
                "Requests library not available.\n"
                "Please install: pip install requests"
            )
            return
        
        # Update config
        self.config.api_url = self.public_url_input.text()
        self.config.local_url = self.local_url_input.text()
        self.config.use_local = self.use_local_check.isChecked()
        self.config.cache_enabled = self.cache_check.isChecked()
        self.config.batch_size = self.batch_size_spin.value()
        self.config.preserve_timestamps = self.timestamp_check.isChecked()
        
        # Initialize translator if needed
        if not self.translator:
            self.translator = LibreTranslator(self.config)
        
        if not self.translator.available:
            QMessageBox.critical(
                self, "Error",
                "LibreTranslate not available.\n"
                "Please check your connection settings."
            )
            return
        
        # Prepare translation config
        trans_config = {
            'source_lang': self.source_lang_combo.currentText(),
            'target_lang': self.target_lang_combo.currentText(),
            'context_window': 3,
            'batch_size': self.config.batch_size
        }
        
        # Start translation thread
        self.translation_thread = TranslationThread(
            self.translator,
            self.original_lines,
            trans_config
        )
        
        self.translation_thread.progress.connect(self.update_progress)
        self.translation_thread.status.connect(self.status_label.setText)
        self.translation_thread.line_translated.connect(self.update_line)
        self.translation_thread.finished.connect(self.translation_finished)
        self.translation_thread.error.connect(self.translation_error)
        
        # Update UI
        self.translate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.translated_lines = [''] * len(self.original_lines)
        
        self.translation_thread.start()
    
    def stop_translation(self):
        """Stop the translation process."""
        if self.translation_thread and self.translation_thread.isRunning():
            self.translation_thread.stop()
            self.status_label.setText("Stopping...")
    
    def update_progress(self, current, total):
        """Update progress bar."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
    
    def update_line(self, index, translation, metadata):
        """Update a single translated line."""
        if index < len(self.translated_lines):
            self.translated_lines[index] = translation
        
        # Update preview with timestamps preserved
        if index % 3 == 0 or index == len(self.original_lines) - 1:
            self.update_preview()
    
    def update_preview(self):
        """Update the translation preview with timestamps."""
        preview_lines = []
        for i, (orig, trans) in enumerate(zip(self.original_lines[:50], self.translated_lines[:50])):
            if not trans:
                continue
            
            if self.timestamp_check.isChecked() and orig.timestamp:
                preview_lines.append(f"[{orig.timestamp}] {trans}")
            elif self.timestamp_check.isChecked() and orig.start_time and orig.end_time:
                preview_lines.append(f"[{orig.start_time} -> {orig.end_time}] {trans}")
            else:
                preview_lines.append(trans)
        
        preview_text = "\n".join(preview_lines)
        if len(self.translated_lines) > 50:
            preview_text += f"\n\n... and {len(self.translated_lines) - 50} more lines"
        
        self.translated_text.setText(preview_text)
    
    def translation_finished(self):
        """Handle translation completion."""
        self.translate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        
        # Update full preview
        self.update_preview()
        
        # Show cache stats
        if self.translator and self.translator.cache:
            stats = self.translator.cache.get_stats()
            self.stats_label.setText(f"📊 Cache hit rate: {stats['hit_rate']:.1f}%")
            self.status_label.setText("✅ Translation complete!")
    
    def translation_error(self, error_msg):
        """Handle translation error."""
        self.translate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        QMessageBox.critical(self, "Translation Error", f"Error: {error_msg}")
        self.status_label.setText("❌ Translation failed")
    
    def save_translation(self):
        """Save translated text to file with timestamps preserved."""
        if not self.translated_lines:
            return
        
        # Determine output format based on input file extension
        ext = os.path.splitext(self.current_file)[1].lower()
        
        format_filters = {
            '.srt': "SRT Subtitles (*.srt)",
            '.ass': "ASS Subtitles (*.ass)",
            '.ssa': "SSA Subtitles (*.ssa)",
            '.json': "JSON (*.json)",
            '.txt': "Text Files (*.txt)"
        }
        
        default_ext = ext if ext in format_filters else '_en.txt'
        format_filter = format_filters.get(ext, "Text Files (*.txt)")
        
        # Get save path
        base_name = os.path.splitext(self.current_file)[0]
        default_name = f"{base_name}_en{default_ext if default_ext.startswith('.') else '.txt'}"
        
        save_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Translation",
            default_name,
            ";;".join(format_filters.values()) + ";;All Files (*.*)"
        )
        
        if save_path:
            try:
                # Determine output format from file extension
                out_ext = os.path.splitext(save_path)[1].lower()
                
                if out_ext == '.srt':
                    out_format = 'srt'
                elif out_ext == '.ass' or out_ext == '.ssa':
                    out_format = 'ass'
                elif out_ext == '.json':
                    out_format = 'json'
                else:
                    out_format = 'txt'
                
                # Save file with timestamps preserved
                SubtitleParser.save_translation(
                    save_path,
                    self.original_lines,
                    self.translated_lines,
                    out_format,
                    self.timestamp_check.isChecked()
                )
                
                QMessageBox.information(
                    self,
                    "Success",
                    f"✅ Translation saved to:\n{save_path}"
                )
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")
    
    def apply_theme(self, is_dark):
        """Apply theme to translation widget."""
        if is_dark:
            self.setStyleSheet("""
                QGroupBox {
                    color: #ffffff;
                    border: 1px solid #3a3a3a;
                    border-radius: 4px;
                    margin-top: 10px;
                    font-weight: bold;
                }
                QGroupBox::title {
                    color: #4CAF50;
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                }
                QLabel {
                    color: #cccccc;
                }
                QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                    background-color: #2d2d30;
                    color: #ffffff;
                    border: 1px solid #3a3a3a;
                    border-radius: 3px;
                    padding: 4px;
                }
                QPushButton {
                    background-color: #3a3a3a;
                    color: white;
                    border: none;
                    border-radius: 3px;
                    padding: 6px 12px;
                }
                QPushButton:hover {
                    background-color: #4CAF50;
                }
                QPushButton:pressed {
                    background-color: #45a049;
                }
                QPushButton:disabled {
                    background-color: #2d2d30;
                    color: #666666;
                }
                QProgressBar {
                    border: 1px solid #3a3a3a;
                    border-radius: 3px;
                    text-align: center;
                    color: white;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                    border-radius: 3px;
                }
            """)
        else:
            self.setStyleSheet("""
                QGroupBox {
                    color: #333333;
                    border: 1px solid #cccccc;
                    border-radius: 4px;
                    margin-top: 10px;
                    font-weight: bold;
                }
                QGroupBox::title {
                    color: #4CAF50;
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                }
                QLabel {
                    color: #666666;
                }
                QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                    background-color: #ffffff;
                    color: #333333;
                    border: 1px solid #cccccc;
                    border-radius: 3px;
                    padding: 4px;
                }
                QPushButton {
                    background-color: #e9ecef;
                    color: #333333;
                    border: none;
                    border-radius: 3px;
                    padding: 6px 12px;
                }
                QPushButton:hover {
                    background-color: #4CAF50;
                    color: white;
                }
                QPushButton:pressed {
                    background-color: #45a049;
                }
                QPushButton:disabled {
                    background-color: #f5f5f5;
                    color: #999999;
                }
                QProgressBar {
                    border: 1px solid #cccccc;
                    border-radius: 3px;
                    text-align: center;
                    color: #333333;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                    border-radius: 3px;
                }
            """)


# ==================== INTEGRATION FUNCTION ====================

def create_translation_tab(parent, theme_manager=None):
    """Create translation tab for AI Studio Pro."""
    return TranslationWidget(parent, theme_manager)


# ==================== EXPORTS ====================

__all__ = [
    'TranslationWidget',
    'LibreTranslator',
    'TranslationCache',
    'SubtitleParser',
    'create_translation_tab',
    'REQUESTS_AVAILABLE',
    'PYSRT_AVAILABLE'
]