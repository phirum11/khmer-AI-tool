"""
AI Studio Pro - Professional Video Editor (CapCut Clone)
- Complete video editing suite similar to CapCut
- Multi-track timeline (video, audio, text)
- Import all media types (video, audio, images)
- Background removal and addition
- Transitions, effects, and text overlays
- Keyframe animations
- Export in multiple formats
"""

import os
import cv2
import numpy as np
import json
import time
import shutil
import uuid
import subprocess
import tempfile
import traceback
import math
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import timedelta
import threading
from queue import Queue

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox, QProgressBar, QGroupBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox,
    QListWidget, QListWidgetItem, QSplitter, QFrame,
    QSlider, QDialog, QFormLayout, QLineEdit, QToolBar,
    QApplication, QGridLayout, QScrollArea, QSizePolicy,
    QStackedWidget, QTextEdit, QMenu, QMenuBar, QStatusBar,
    QColorDialog, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QGraphicsTextItem, QGraphicsItem, QRubberBand,
    QDialogButtonBox
)
from PyQt6.QtCore import (
    Qt, QTimer, QSize, pyqtSignal, QThread, QMutex, QMutexLocker,
    QRect, QPoint, QByteArray, QUrl, QPropertyAnimation,
    QEasingCurve, QPointF, QRectF
)
from PyQt6.QtGui import (
    QFont, QColor, QPainter, QPen, QBrush, QAction, QIcon,
    QPixmap, QImage, QMovie, QFontDatabase, QShortcut, QKeySequence,
    QLinearGradient, QPalette, QDesktopServices, QTransform,
    QPainterPath, QFontMetrics
)

# Try to import audio libraries with fallbacks
try:
    import pydub
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("⚠️ pydub not found - audio editing features will be limited")
    print("   Install with: pip install pydub")

# Use pygame for audio playback (easier to install on Windows)
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️ pygame not found - audio playback will be disabled")
    print("   Install with: pip install pygame")

# Try to import background removal libraries
try:
    import rembg
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("⚠️ rembg not found - background removal features will be limited")
    print("   Install with: pip install rembg")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ Pillow not found - install with: pip install Pillow")


# ==================== CONSTANTS ====================

# Modern dark theme colors
COLORS = {
    'bg_primary': '#18181b',
    'bg_secondary': '#27272a',
    'bg_tertiary': '#3f3f46',
    'bg_hover': '#52525b',
    'text_primary': '#e4e4e7',
    'text_secondary': '#a1a1aa',
    'accent': '#10b981',
    'accent_hover': '#34d399',
    'border': '#3f3f46',
    'panel_border': '#3f3f46',
    'video_track': '#10b981',
    'audio_track': '#3b82f6',
    'text_track': '#f59e0b',
    'playhead': '#ef4444'
}

THUMBNAIL_SIZE = QSize(160, 90)
PREVIEW_SIZE = QSize(854, 480)  # 16:9 - reduced from 1280x720 for performance
CONTROL_HEIGHT = 60
TIMELINE_TRACK_HEIGHT = 60
TIMELINE_RULER_HEIGHT = 30
PIXELS_PER_SECOND = 50

# Supported formats
VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
AUDIO_FORMATS = ['.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a', '.wma']
IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']


# ==================== FFMPEG CHECK ====================

FFMPEG_AVAILABLE = shutil.which('ffmpeg') is not None
FFPROBE_AVAILABLE = shutil.which('ffprobe') is not None

if not FFMPEG_AVAILABLE:
    print("⚠️ FFmpeg not found - install FFmpeg for video processing")
    print("   Download from: https://ffmpeg.org/download.html")
else:
    print("✅ FFmpeg found")


# ==================== ENUMS ====================

class MediaType(Enum):
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"


class TrackType(Enum):
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"


class TransitionType(Enum):
    NONE = "none"
    FADE = "fade"
    CROSSFADE = "crossfade"
    WIPE = "wipe"
    SLIDE = "slide"
    ZOOM = "zoom"


class EffectType(Enum):
    NONE = "none"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATION = "saturation"
    BLUR = "blur"
    SHARPEN = "sharpen"
    GRAYSCALE = "grayscale"
    SEPIA = "sepia"
    INVERT = "invert"


class BackgroundMode(Enum):
    NONE = "none"
    REMOVE = "remove"
    ADD = "add"
    REPLACE = "replace"
    GREEN_SCREEN = "green_screen"


# ==================== DATA CLASSES ====================

@dataclass
class MediaItem:
    """Represents a media item in the media pool."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    file_path: str = ""
    type: MediaType = MediaType.VIDEO
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 30.0
    file_size: int = 0
    thumbnail: Optional[QPixmap] = None
    is_loading: bool = False
    load_error: Optional[str] = None
    has_audio: bool = False
    audio_channels: int = 0
    audio_sample_rate: int = 0
    codec: str = ""
    
    @property
    def resolution_str(self) -> str:
        return f"{self.width}x{self.height}"
    
    @property
    def duration_str(self) -> str:
        hours = int(self.duration // 3600)
        minutes = int((self.duration % 3600) // 60)
        seconds = int(self.duration % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    @property
    def size_str(self) -> str:
        if self.file_size < 1024:
            return f"{self.file_size} B"
        elif self.file_size < 1024 * 1024:
            return f"{self.file_size / 1024:.1f} KB"
        elif self.file_size < 1024 * 1024 * 1024:
            return f"{self.file_size / (1024 * 1024):.1f} MB"
        else:
            return f"{self.file_size / (1024 * 1024 * 1024):.1f} GB"
    
    @property
    def is_large_file(self) -> bool:
        """Check if file is larger than 1GB."""
        return self.file_size > 1024 * 1024 * 1024
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'file_path': self.file_path,
            'type': self.type.value,
            'duration': self.duration,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'file_size': self.file_size,
            'has_audio': self.has_audio,
            'codec': self.codec
        }


@dataclass
class TimelineClip:
    """Represents a clip on the timeline."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    media_id: str = ""
    track: int = 0
    track_type: TrackType = TrackType.VIDEO
    start: float = 0.0  # Timeline start time
    end: float = 0.0    # Timeline end time
    media_start: float = 0.0  # Start time in source media
    media_end: float = 0.0    # End time in source media
    name: str = ""
    type: MediaType = MediaType.VIDEO
    speed: float = 1.0
    volume: float = 1.0
    pan: float = 0.0
    muted: bool = False
    effects: List[Dict] = field(default_factory=list)
    keyframes: List[Dict] = field(default_factory=list)
    transform: Dict = field(default_factory=lambda: {
        'x': 0, 'y': 0, 'scale': 1.0, 'rotation': 0, 'opacity': 1.0
    })
    text_content: str = ""
    text_font: str = "Arial"
    text_size: int = 24
    text_color: str = "#ffffff"
    
    @property
    def duration(self) -> float:
        return self.end - self.start
    
    @property
    def media_duration(self) -> float:
        return self.media_end - self.media_start
    
    def split(self, time: float) -> Tuple['TimelineClip', 'TimelineClip']:
        """Split clip at given time."""
        if time <= self.start or time >= self.end:
            raise ValueError("Split time must be within clip")
        
        # Calculate media position at split point
        media_pos = self.media_start + (time - self.start) * self.speed
        
        clip1 = TimelineClip(
            media_id=self.media_id,
            track=self.track,
            track_type=self.track_type,
            start=self.start,
            end=time,
            media_start=self.media_start,
            media_end=media_pos,
            name=self.name,
            type=self.type,
            speed=self.speed,
            volume=self.volume,
            pan=self.pan,
            muted=self.muted,
            effects=self.effects.copy(),
            text_content=self.text_content
        )
        
        clip2 = TimelineClip(
            media_id=self.media_id,
            track=self.track,
            track_type=self.track_type,
            start=time,
            end=self.end,
            media_start=media_pos,
            media_end=self.media_end,
            name=self.name,
            type=self.type,
            speed=self.speed,
            volume=self.volume,
            pan=self.pan,
            muted=self.muted,
            effects=self.effects.copy(),
            text_content=self.text_content
        )
        
        return clip1, clip2
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'media_id': self.media_id,
            'track': self.track,
            'track_type': self.track_type.value,
            'start': self.start,
            'end': self.end,
            'media_start': self.media_start,
            'media_end': self.media_end,
            'name': self.name,
            'type': self.type.value,
            'speed': self.speed,
            'volume': self.volume,
            'pan': self.pan,
            'muted': self.muted,
            'effects': self.effects,
            'text_content': self.text_content
        }


@dataclass
class Track:
    """Represents a timeline track."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: TrackType = TrackType.VIDEO
    index: int = 0
    height: int = TIMELINE_TRACK_HEIGHT
    visible: bool = True
    locked: bool = False
    muted: bool = False


# ==================== VIDEO LOADER THREAD ====================

class MediaLoaderThread(QThread):
    """Thread for loading media metadata without blocking UI."""
    
    metadata_loaded = pyqtSignal(str, object)  # id, media_item
    thumbnail_ready = pyqtSignal(str, QPixmap)
    error = pyqtSignal(str, str)
    progress = pyqtSignal(str, int)  # id, progress percent
    
    def __init__(self):
        super().__init__()
        self.queue = []
        self._running = True
        self._mutex = QMutex()
    
    def add_job(self, media_id: str, file_path: str, media_type: MediaType):
        """Add media loading job to queue."""
        with QMutexLocker(self._mutex):
            self.queue.append((media_id, file_path, media_type))
    
    def stop(self):
        self._running = False
    
    def run(self):
        """Main thread loop."""
        while self._running:
            job = None
            with QMutexLocker(self._mutex):
                if self.queue:
                    job = self.queue.pop(0)
            
            if job:
                media_id, file_path, media_type = job
                self._load_metadata(media_id, file_path, media_type)
            else:
                self.msleep(100)
    
    def _load_metadata(self, media_id: str, file_path: str, media_type: MediaType):
        """Load media metadata."""
        try:
            self.progress.emit(media_id, 10)
            
            if media_type == MediaType.VIDEO:
                self._load_video_metadata(media_id, file_path)
            elif media_type == MediaType.AUDIO:
                self._load_audio_metadata(media_id, file_path)
            elif media_type == MediaType.IMAGE:
                self._load_image_metadata(media_id, file_path)
                
        except Exception as e:
            self.error.emit(media_id, f"Failed to load: {str(e)}")
    
    def _load_video_metadata(self, media_id: str, file_path: str):
        """Load video metadata."""
        try:
            if FFPROBE_AVAILABLE:
                cmd = [
                    'ffprobe', '-v', 'quiet', '-print_format', 'json',
                    '-show_streams', '-show_format', file_path
                ]
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=30,
                    encoding='utf-8',
                    errors='ignore'
                )
                
                self.progress.emit(media_id, 50)
                
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    
                    # Create media item
                    media_item = MediaItem(
                        id=media_id,
                        name=os.path.basename(file_path),
                        file_path=file_path,
                        type=MediaType.VIDEO,
                        file_size=os.path.getsize(file_path)
                    )
                    
                    # Get duration
                    if 'format' in data and 'duration' in data['format']:
                        media_item.duration = float(data['format']['duration'])
                    
                    # Get stream info
                    for stream in data.get('streams', []):
                        stream_type = stream.get('codec_type')
                        
                        if stream_type == 'video':
                            media_item.width = stream.get('width', 0)
                            media_item.height = stream.get('height', 0)
                            media_item.codec = stream.get('codec_name', '')
                            
                            fps_str = stream.get('r_frame_rate', '30/1')
                            if '/' in fps_str:
                                num, den = map(int, fps_str.split('/'))
                                media_item.fps = num / den if den != 0 else 30.0
                        
                        elif stream_type == 'audio':
                            media_item.has_audio = True
                            media_item.audio_channels = stream.get('channels', 0)
                            media_item.audio_sample_rate = int(stream.get('sample_rate', 0))
                    
                    self.metadata_loaded.emit(media_id, media_item)
                    self.progress.emit(media_id, 70)
                    
                    # Generate thumbnail
                    self._generate_video_thumbnail(media_id, file_path, media_item.duration)
                else:
                    self._load_video_with_opencv(media_id, file_path)
            else:
                self._load_video_with_opencv(media_id, file_path)
                
        except Exception as e:
            self.error.emit(media_id, f"Failed to load video: {str(e)}")
    
    def _load_video_with_opencv(self, media_id: str, file_path: str):
        """Load video metadata using OpenCV."""
        try:
            cap = cv2.VideoCapture(file_path)
            self.progress.emit(media_id, 30)
            
            if cap.isOpened():
                media_item = MediaItem(
                    id=media_id,
                    name=os.path.basename(file_path),
                    file_path=file_path,
                    type=MediaType.VIDEO,
                    file_size=os.path.getsize(file_path)
                )
                
                media_item.fps = cap.get(cv2.CAP_PROP_FPS)
                if media_item.fps <= 0:
                    media_item.fps = 30.0
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                media_item.duration = total_frames / media_item.fps if media_item.fps > 0 else 0
                media_item.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                media_item.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                self.metadata_loaded.emit(media_id, media_item)
                self.progress.emit(media_id, 60)
                
                # Generate thumbnail
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                ret, frame = cap.read()
                if ret:
                    self._create_thumbnail_from_frame(media_id, frame)
                
                cap.release()
            else:
                self.error.emit(media_id, "Could not open video file")
                
        except Exception as e:
            self.error.emit(media_id, f"OpenCV error: {str(e)}")
    
    def _load_audio_metadata(self, media_id: str, file_path: str):
        """Load audio metadata."""
        try:
            media_item = MediaItem(
                id=media_id,
                name=os.path.basename(file_path),
                file_path=file_path,
                type=MediaType.AUDIO,
                file_size=os.path.getsize(file_path)
            )
            
            if FFPROBE_AVAILABLE:
                cmd = [
                    'ffprobe', '-v', 'quiet', '-print_format', 'json',
                    '-show_streams', '-show_format', file_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    
                    if 'format' in data and 'duration' in data['format']:
                        media_item.duration = float(data['format']['duration'])
                    
                    for stream in data.get('streams', []):
                        if stream.get('codec_type') == 'audio':
                            media_item.audio_channels = stream.get('channels', 0)
                            media_item.audio_sample_rate = int(stream.get('sample_rate', 0))
                            media_item.codec = stream.get('codec_name', '')
                            break
            
            # Generate audio waveform thumbnail
            self._generate_audio_thumbnail(media_id, file_path)
            
            self.metadata_loaded.emit(media_id, media_item)
            self.progress.emit(media_id, 100)
            
        except Exception as e:
            self.error.emit(media_id, f"Failed to load audio: {str(e)}")
    
    def _load_image_metadata(self, media_id: str, file_path: str):
        """Load image metadata."""
        try:
            media_item = MediaItem(
                id=media_id,
                name=os.path.basename(file_path),
                file_path=file_path,
                type=MediaType.IMAGE,
                file_size=os.path.getsize(file_path)
            )
            
            # Get image dimensions
            img = cv2.imread(file_path)
            if img is not None:
                media_item.height, media_item.width = img.shape[:2]
                media_item.duration = 5.0  # Default duration for images
            
            self.metadata_loaded.emit(media_id, media_item)
            self.progress.emit(media_id, 50)
            
            # Generate thumbnail
            self._generate_image_thumbnail(media_id, file_path)
            
        except Exception as e:
            self.error.emit(media_id, f"Failed to load image: {str(e)}")
    
    def _generate_video_thumbnail(self, media_id: str, file_path: str, duration: float):
        """Generate video thumbnail using ffmpeg."""
        try:
            time_pos = duration / 2 if duration > 0 else 0
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
            
            cmd = [
                'ffmpeg', '-ss', str(time_pos), '-i', file_path,
                '-vframes', '1', '-vf', f'scale={THUMBNAIL_SIZE.width()}:{THUMBNAIL_SIZE.height()}',
                '-q:v', '85', '-y', temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
            self.progress.emit(media_id, 90)
            
            if result.returncode == 0 and os.path.exists(temp_path):
                pixmap = QPixmap(temp_path)
                if not pixmap.isNull():
                    self.thumbnail_ready.emit(media_id, pixmap)
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            self.progress.emit(media_id, 100)
            
        except Exception as e:
            print(f"Thumbnail generation error: {e}")
    
    def _generate_audio_thumbnail(self, media_id: str, file_path: str):
        """Generate audio waveform thumbnail."""
        try:
            # Create a simple audio waveform icon
            pixmap = QPixmap(THUMBNAIL_SIZE)
            pixmap.fill(QColor(COLORS['bg_tertiary']))
            
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor(COLORS['accent']), 2))
            
            # Draw waveform
            width = THUMBNAIL_SIZE.width()
            height = THUMBNAIL_SIZE.height()
            
            for x in range(0, width, 4):
                h = int((math.sin(x * 0.1) + 1) * height / 4)
                painter.drawLine(x, height//2 - h//2, x, height//2 + h//2)
            
            painter.end()
            
            self.thumbnail_ready.emit(media_id, pixmap)
            
        except Exception as e:
            print(f"Audio thumbnail error: {e}")
    
    def _generate_image_thumbnail(self, media_id: str, file_path: str):
        """Generate image thumbnail."""
        try:
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    THUMBNAIL_SIZE,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.thumbnail_ready.emit(media_id, scaled)
            
        except Exception as e:
            print(f"Image thumbnail error: {e}")
    
    def _create_thumbnail_from_frame(self, media_id: str, frame):
        """Create thumbnail from OpenCV frame."""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            
            aspect = w / h
            new_w = THUMBNAIL_SIZE.width()
            new_h = int(new_w / aspect)
            
            if new_h > THUMBNAIL_SIZE.height():
                new_h = THUMBNAIL_SIZE.height()
                new_w = int(new_h * aspect)
            
            resized = cv2.resize(rgb_frame, (new_w, new_h))
            
            bytes_per_line = ch * new_w
            image = QImage(resized.data, new_w, new_h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            
            self.thumbnail_ready.emit(media_id, pixmap)
            self.progress.emit(media_id, 100)
            
        except Exception as e:
            print(f"Thumbnail creation error: {e}")


# ==================== AUDIO PLAYER THREAD (with pygame) ====================

class AudioPlayerThread(QThread):
    """Thread for playing audio without blocking UI using pygame."""
    
    position_changed = pyqtSignal(float)
    finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        global PYGAME_AVAILABLE
        
        self.audio_file = None
        self.audio_data = None
        self._running = True
        self._playing = False
        self._mutex = QMutex()
        self.position = 0.0
        self.duration = 0.0
        self.start_time = 0.0
        self.pygame_available = PYGAME_AVAILABLE
        
        # Initialize pygame mixer if available
        if self.pygame_available:
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                print("✅ Pygame audio initialized")
            except Exception as e:
                print(f"⚠️ Pygame init error: {e}")
                self.pygame_available = False
    
    def load_audio(self, file_path: str):
        """Load audio file."""
        try:
            if PYDUB_AVAILABLE:
                self.audio_data = AudioSegment.from_file(file_path)
                self.duration = len(self.audio_data) / 1000.0
                self.audio_file = file_path
                print(f"✅ Audio loaded: {os.path.basename(file_path)}")
            else:
                # Try to load with pygame directly
                if self.pygame_available:
                    self.audio_file = file_path
                    # Try to get duration with ffprobe
                    if FFPROBE_AVAILABLE:
                        cmd = [
                            'ffprobe', '-v', 'quiet', '-print_format', 'json',
                            '-show_format', file_path
                        ]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                        if result.returncode == 0:
                            data = json.loads(result.stdout)
                            self.duration = float(data['format'].get('duration', 0))
                    print(f"✅ Audio file set: {os.path.basename(file_path)}")
                else:
                    print("⚠️ Audio loading disabled - install pydub")
        except Exception as e:
            print(f"Audio load error: {e}")
    
    def play(self):
        """Start playback."""
        with QMutexLocker(self._mutex):
            if not self._playing and self.audio_file and self.pygame_available:
                try:
                    pygame.mixer.music.load(self.audio_file)
                    pygame.mixer.music.play()
                    self._playing = True
                    self.start_time = time.time() - self.position
                    print(f"▶️ Audio playing: {os.path.basename(self.audio_file)}")
                except Exception as e:
                    print(f"Audio play error: {e}")
    
    def pause(self):
        """Pause playback."""
        with QMutexLocker(self._mutex):
            if self.pygame_available and self._playing:
                pygame.mixer.music.pause()
                self._playing = False
                print("⏸️ Audio paused")
    
    def stop(self):
        """Stop playback."""
        with QMutexLocker(self._mutex):
            if self.pygame_available:
                pygame.mixer.music.stop()
            self._playing = False
            self.position = 0.0
            self.start_time = 0.0
            print("⏹️ Audio stopped")
    
    def seek(self, position: float):
        """Seek to position."""
        with QMutexLocker(self._mutex):
            self.position = max(0, min(position, self.duration))
            if self.pygame_available and self._playing:
                # Pygame doesn't support seeking well, so we'll restart
                try:
                    pygame.mixer.music.play(start=position)
                    self.start_time = time.time() - position
                    print(f"⏩ Audio seeking to {position:.2f}s")
                except Exception as e:
                    print(f"Audio seek error: {e}")
    
    def set_volume(self, volume: float):
        """Set volume (0.0 to 1.0)."""
        with QMutexLocker(self._mutex):
            if self.pygame_available:
                pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))
    
    def is_playing(self) -> bool:
        """Check if audio is playing."""
        with QMutexLocker(self._mutex):
            return self._playing
    
    def get_position(self) -> float:
        """Get current position."""
        with QMutexLocker(self._mutex):
            return self.position
    
    def get_duration(self) -> float:
        """Get total duration."""
        with QMutexLocker(self._mutex):
            return self.duration
    
    def run(self):
        """Main thread loop."""
        print("🎵 Audio player thread started")
        
        while self._running:
            if self._playing and self.pygame_available:
                try:
                    # Check if music is still playing
                    if pygame.mixer.music.get_busy():
                        # Calculate position from start time
                        self.position = time.time() - self.start_time
                        if self.position > self.duration:
                            self.position = self.duration
                        
                        self.position_changed.emit(self.position)
                    else:
                        # Music finished
                        if self._playing:  # Only if we were playing
                            self._playing = False
                            self.position = 0.0
                            self.start_time = 0.0
                            self.finished.emit()
                            print("⏹️ Audio playback finished")
                except Exception as e:
                    print(f"Audio playback error: {e}")
                    self._playing = False
            
            self.msleep(50)  # Update every 50ms
        
        print("🎵 Audio player thread stopped")
    
    def stop_thread(self):
        """Stop the thread."""
        print("🛑 Stopping audio player thread...")
        self._running = False
        self.stop()
        if self.pygame_available:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except:
                pass
        self.wait()
        print("✅ Audio player thread stopped")


# ==================== VIDEO PLAYER THREAD (FIXED) ====================

class VideoPlayerThread(QThread):
    """Thread for playing video without blocking UI. FIXED version with proper buffer handling."""
    
    frame_ready = pyqtSignal(QImage)
    position_changed = pyqtSignal(float)
    duration_ready = pyqtSignal(float)
    error = pyqtSignal(str)
    load_started = pyqtSignal()
    load_finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.cap = None
        self.fps = 30.0
        self.total_frames = 0
        self.current_frame = 0
        self.playing = False
        self._running = True
        self._mutex = QMutex()
        self.position = 0.0
        self.duration = 0.0
        self.is_loading = False
        self.backend_used = None
        self.target_size = (PREVIEW_SIZE.width(), PREVIEW_SIZE.height())
        self.target_dimensions = self.target_size
        self.test_mode = False
        
        # Effects
        self.effects = []
        self.bg_mode = BackgroundMode.NONE
        self.bg_color = (0, 255, 0)
        self.bg_image = None
    
    def load_video(self, path: str):
        """Load video file asynchronously with backend fallbacks."""
        print(f"\n[VideoThread] ===== LOADING VIDEO =====")
        print(f"[VideoThread] File: {os.path.basename(path)}")
        self.load_started.emit()
        self.is_loading = True
        QTimer.singleShot(10, lambda: self._load_video_thread(path))
    
    def _load_video_thread(self, path: str):
        """Actual video loading in background with backend fallbacks."""
        try:
            # Try different backends in order of reliability
            backends = [
                cv2.CAP_FFMPEG,      # Best for most videos
                cv2.CAP_ANY,          # OpenCV default
                cv2.CAP_DSHOW,        # Windows DirectShow
            ]
            
            cap = None
            used_backend = None
            
            for backend in backends:
                try:
                    print(f"[VideoThread] Trying backend: {backend}")
                    cap = cv2.VideoCapture(path, backend)
                    if cap.isOpened():
                        # Test read one frame
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            used_backend = backend
                            print(f"[VideoThread] ✓ Backend {backend} works")
                            break
                        else:
                            cap.release()
                            cap = None
                    else:
                        if cap:
                            cap.release()
                        cap = None
                except Exception as e:
                    print(f"[VideoThread] Backend {backend} failed: {e}")
                    if cap:
                        cap.release()
                    cap = None
            
            if cap is None:
                # Final attempt with default
                print("[VideoThread] Trying default backend")
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    self.error.emit(f"Cannot open video with any backend")
                    return
            
            with QMutexLocker(self._mutex):
                if self.cap:
                    self.cap.release()
                self.cap = cap
                self.backend_used = used_backend
                
                # Get properties
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.fps <= 0 or math.isnan(self.fps):
                    self.fps = 30.0
                    print(f"[VideoThread] Invalid FPS, using default: {self.fps}")
                
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if self.total_frames <= 0:
                    # Estimate by seeking to end (slow but works)
                    print("[VideoThread] Estimating frame count...")
                    pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 10**9)  # Seek far
                    self.total_frames = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    print(f"[VideoThread] Estimated frames: {self.total_frames}")
                
                self.duration = self.total_frames / self.fps if self.fps > 0 else 0
                self.current_frame = 0
                self.position = 0.0
                
                # Get video dimensions
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if width <= 0 or height <= 0:
                    width, height = 640, 480
                
                # Calculate target size
                aspect = width / height
                target_width, target_height = self.target_size
                
                if width > target_width or height > target_height:
                    if aspect > 1:
                        new_width = min(width, target_width)
                        new_height = int(new_width / aspect)
                    else:
                        new_height = min(height, target_height)
                        new_width = int(new_height * aspect)
                else:
                    new_width, new_height = width, height
                
                self.target_dimensions = (new_width, new_height)
                
                # Pre-roll one frame to verify
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    self.error.emit("Video opens but cannot read first frame")
                    return
                
                # Get codec info
                fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
                codec_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]) if fourcc else "unknown"
                
                print(f"\n[VideoThread] ✓ SUCCESSFULLY LOADED:")
                print(f"           File: {os.path.basename(path)}")
                print(f"           Backend: {used_backend}")
                print(f"           Resolution: {width}x{height}")
                print(f"           FPS: {self.fps:.2f}")
                print(f"           Frames: {self.total_frames}")
                print(f"           Duration: {self.duration:.1f}s")
                print(f"           Codec: {codec_str}")
                print(f"           Target size: {new_width}x{new_height}\n")
                
                self.duration_ready.emit(self.duration)
                
        except Exception as e:
            print(f"[VideoThread] ❌ Load error: {e}")
            traceback.print_exc()
            self.error.emit(f"Load failed: {str(e)}")
        finally:
            self.is_loading = False
            self.load_finished.emit()
    
    def set_effects(self, effects: List[Dict]):
        """Set video effects."""
        with QMutexLocker(self._mutex):
            self.effects = effects
    
    def set_background_mode(self, mode: BackgroundMode, color: Tuple[int, int, int] = (0, 255, 0), image_path: str = None):
        """Set background mode."""
        with QMutexLocker(self._mutex):
            self.bg_mode = mode
            self.bg_color = color
            if image_path and os.path.exists(image_path):
                self.bg_image = cv2.imread(image_path)
    
    def _apply_effects(self, frame):
        """Apply effects to frame."""
        if not self.effects:
            return frame
        
        result = frame.copy()
        
        for effect in self.effects:
            effect_type = effect.get('type', EffectType.NONE)
            intensity = effect.get('intensity', 1.0)
            
            if effect_type == EffectType.BRIGHTNESS:
                result = cv2.convertScaleAbs(result, alpha=1.0, beta=intensity * 50)
            
            elif effect_type == EffectType.CONTRAST:
                result = cv2.convertScaleAbs(result, alpha=1.0 + intensity, beta=0)
            
            elif effect_type == EffectType.GRAYSCALE:
                if len(result.shape) == 3:
                    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            elif effect_type == EffectType.BLUR:
                k = int(intensity * 10) + 1
                if k % 2 == 0:
                    k += 1
                result = cv2.GaussianBlur(result, (k, k), 0)
            
            elif effect_type == EffectType.SHARPEN:
                kernel = np.array([[-1,-1,-1],
                                   [-1, 9,-1],
                                   [-1,-1,-1]]) * intensity
                result = cv2.filter2D(result, -1, kernel)
        
        return result
    
    def _apply_background(self, frame):
        """Apply background processing."""
        if self.bg_mode == BackgroundMode.NONE:
            return frame
        
        try:
            if self.bg_mode == BackgroundMode.REMOVE and REMBG_AVAILABLE:
                import rembg
                from PIL import Image
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                output = rembg.remove(pil_image)
                result = cv2.cvtColor(np.array(output), cv2.COLOR_RGBA2BGR)
                return result
            
            elif self.bg_mode == BackgroundMode.GREEN_SCREEN:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Green color range
                lower_green = np.array([40, 40, 40])
                upper_green = np.array([80, 255, 255])
                
                mask = cv2.inRange(hsv, lower_green, upper_green)
                mask_inv = cv2.bitwise_not(mask)
                
                result = cv2.bitwise_and(frame, frame, mask=mask_inv)
                return result
            
            elif self.bg_mode == BackgroundMode.REPLACE and self.bg_image is not None:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_green = np.array([40, 40, 40])
                upper_green = np.array([80, 255, 255])
                
                mask = cv2.inRange(hsv, lower_green, upper_green)
                mask_inv = cv2.bitwise_not(mask)
                
                bg_resized = cv2.resize(self.bg_image, (frame.shape[1], frame.shape[0]))
                
                fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                bg = cv2.bitwise_and(bg_resized, bg_resized, mask=mask)
                result = cv2.add(fg, bg)
                return result
            
        except Exception as e:
            print(f"Background processing error: {e}")
        
        return frame
    
    def play(self):
        """Start playback."""
        with QMutexLocker(self._mutex):
            self.playing = True
            print(f"[VideoThread] ▶️ Play")
    
    def pause(self):
        """Pause playback."""
        with QMutexLocker(self._mutex):
            self.playing = False
            print(f"[VideoThread] ⏸️ Pause at {self.position:.1f}s")
    
    def stop(self):
        """Stop playback."""
        with QMutexLocker(self._mutex):
            self.playing = False
            self.current_frame = 0
            self.position = 0.0
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print(f"[VideoThread] ⏹️ Stop")
    
    def seek(self, position: float):
        """Seek to position in seconds."""
        with QMutexLocker(self._mutex):
            self.position = max(0, min(position, self.duration))
            self.current_frame = int(self.position * self.fps)
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            self.position_changed.emit(self.position)
            print(f"[VideoThread] ⏩ Seek to {position:.1f}s (frame {self.current_frame})")
    
    def set_frame(self, frame: int):
        """Set to specific frame."""
        with QMutexLocker(self._mutex):
            self.current_frame = max(0, min(frame, self.total_frames - 1))
            self.position = self.current_frame / self.fps if self.fps > 0 else 0
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            self.position_changed.emit(self.position)
    
    def prev_frame(self):
        """Go to previous frame."""
        self.set_frame(self.current_frame - 1)
    
    def next_frame(self):
        """Go to next frame."""
        self.set_frame(self.current_frame + 1)
    
    def run(self):
        """Main thread loop with proper buffer handling and error recovery."""
        last_frame_time = 0
        frame_interval = 0
        frame_count = 0
        error_count = 0
        thread_id = int(QThread.currentThreadId())
        
        print(f"[VideoThread] ▶️ Thread started (ID: {thread_id})")
        
        while self._running:
            try:
                with QMutexLocker(self._mutex):
                    playing = self.playing
                    if not playing or not self.cap or not self.cap.isOpened():
                        self.msleep(10)
                        continue
                    
                    if frame_interval == 0 and self.fps > 0:
                        frame_interval = 1.0 / min(self.fps, 60.0)  # Cap at 60fps
                        print(f"[VideoThread] Target interval: {frame_interval*1000:.1f}ms")
                    
                    now = time.time()
                    if now - last_frame_time < frame_interval:
                        self.msleep(1)
                        continue
                
                # Read frame outside mutex to avoid blocking
                ret, cv_frame = self.cap.read()
                
                if not ret:
                    # End of video - loop
                    print(f"[VideoThread] 🔄 End of video, looping (frame {self.current_frame})")
                    with QMutexLocker(self._mutex):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.current_frame = 0
                        self.position = 0.0
                    self.msleep(10)
                    continue
                
                # Process frame
                try:
                    # Apply effects if any
                    if self.effects or self.bg_mode != BackgroundMode.NONE:
                        cv_frame = self._apply_effects(cv_frame)
                        cv_frame = self._apply_background(cv_frame)
                    
                    # Resize if needed
                    if hasattr(self, 'target_dimensions'):
                        new_width, new_height = self.target_dimensions
                        if cv_frame.shape[1] > new_width or cv_frame.shape[0] > new_height:
                            cv_frame = cv2.resize(cv_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                    
                    # Convert colorspace
                    rgb = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    
                    # CRITICAL: Make a deep copy of the buffer
                    rgb_copy = rgb.copy()  # This ensures the data persists
                    
                    # Create QImage from the copied buffer
                    bytes_per_line = ch * w
                    qimg = QImage(rgb_copy.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    
                    # Detach the buffer - make a deep copy in Qt
                    qimg_copy = qimg.copy()  # This ensures Qt owns the memory
                    
                except Exception as e:
                    print(f"[VideoThread] ⚠️ Frame processing error: {e}")
                    error_count += 1
                    if error_count > 10:
                        self.error.emit("Too many frame processing errors")
                        break
                    self.msleep(10)
                    continue
                
                # Update position and emit
                with QMutexLocker(self._mutex):
                    self.current_frame += 1
                    self.position = self.current_frame / self.fps if self.fps > 0 else 0
                
                self.frame_ready.emit(qimg_copy)
                self.position_changed.emit(self.position)
                
                frame_count += 1
                if frame_count % 300 == 0:  # Every ~10 seconds at 30fps
                    print(f"[VideoThread] Playing: {frame_count} frames, pos: {self.position:.1f}s")
                
                last_frame_time = now
                error_count = 0  # Reset error count on success
                
                # Calculate sleep to maintain frame rate
                elapsed = time.time() - now
                sleep_ms = max(1, int((frame_interval - elapsed) * 1000))
                self.msleep(min(33, sleep_ms))
                
            except Exception as e:
                print(f"[VideoThread] ❌ CRITICAL ERROR: {e}")
                traceback.print_exc()
                self.error.emit(f"Playback error: {str(e)}")
                self.msleep(100)  # Avoid tight error loop
        
        print(f"[VideoThread] ⏹️ Thread stopped after {frame_count} frames, {error_count} errors (ID: {thread_id})")
    
    def stop_thread(self):
        """Stop the thread."""
        print("[VideoThread] Stopping thread...")
        self._running = False
        self.wait()
        if self.cap:
            self.cap.release()
        print("[VideoThread] Thread stopped")


# ==================== TIMELINE WIDGET ====================

class TimelineWidget(QWidget):
    """Professional multi-track timeline widget."""
    
    position_changed = pyqtSignal(float)
    clip_selected = pyqtSignal(object)
    clip_moved = pyqtSignal(object, float)
    clip_split = pyqtSignal(object, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_primary']};
                border-top: 1px solid {COLORS['border']};
            }}
            QPushButton {{
                background-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_primary']};
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_hover']};
            }}
        """)
        
        self.tracks = self._create_default_tracks()
        self.clips = []  # List of timeline clips
        self.media_items = {}  # media_id -> media_item
        self.duration = 60.0
        self.current_time = 0.0
        self.zoom_level = 1.0
        self.scroll_x = 0
        self.selected_clip = None
        self.dragging = False
        self.drag_clip = None
        self.drag_offset = 0
        self.snapping_enabled = True
        self.snap_threshold = 10
        
        self.setup_ui()
        self.setup_shortcuts()
    
    def _create_default_tracks(self) -> List[Track]:
        """Create default tracks."""
        return [
            Track(name="Video 1", type=TrackType.VIDEO, index=0),
            Track(name="Video 2", type=TrackType.VIDEO, index=1),
            Track(name="Audio 1", type=TrackType.AUDIO, index=2),
            Track(name="Audio 2", type=TrackType.AUDIO, index=3),
            Track(name="Text 1", type=TrackType.TEXT, index=4, height=40),
        ]
    
    def setup_ui(self):
        """Setup timeline UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)
        
        # Timeline ruler
        self.ruler = TimelineRuler(self)
        self.ruler.position_changed.connect(self.set_position)
        layout.addWidget(self.ruler)
        
        # Tracks area with scroll
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        tracks_container = QWidget()
        tracks_layout = QHBoxLayout(tracks_container)
        tracks_layout.setContentsMargins(0, 0, 0, 0)
        tracks_layout.setSpacing(0)
        
        # Track headers
        self.track_headers = TrackHeaders(self)
        self.track_headers.set_tracks(self.tracks)
        self.track_headers.track_muted.connect(self.on_track_muted)
        self.track_headers.track_locked.connect(self.on_track_locked)
        tracks_layout.addWidget(self.track_headers)
        
        # Track view
        self.track_view = TrackView(self)
        self.track_view.set_tracks(self.tracks)
        self.track_view.set_clips(self.clips)
        self.track_view.clip_selected.connect(self.on_clip_selected)
        self.track_view.clip_moved.connect(self.on_clip_moved)
        self.track_view.position_changed.connect(self.position_changed)
        tracks_layout.addWidget(self.track_view)
        
        scroll_area.setWidget(tracks_container)
        layout.addWidget(scroll_area, 1)
    
    def _create_toolbar(self) -> QWidget:
        """Create timeline toolbar."""
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 2, 5, 2)
        
        # Zoom controls
        toolbar_layout.addWidget(QLabel("Zoom:"))
        
        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.setFixedSize(24, 24)
        self.zoom_out_btn.clicked.connect(lambda: self.set_zoom(self.zoom_level * 0.8))
        toolbar_layout.addWidget(self.zoom_out_btn)
        
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 500)
        self.zoom_slider.setValue(int(self.zoom_level * 100))
        self.zoom_slider.setFixedWidth(100)
        self.zoom_slider.valueChanged.connect(lambda v: self.set_zoom(v / 100))
        toolbar_layout.addWidget(self.zoom_slider)
        
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedSize(24, 24)
        self.zoom_in_btn.clicked.connect(lambda: self.set_zoom(self.zoom_level * 1.2))
        toolbar_layout.addWidget(self.zoom_in_btn)
        
        toolbar_layout.addSpacing(10)
        
        # Snap toggle
        self.snap_btn = QPushButton("🔗 Snap")
        self.snap_btn.setCheckable(True)
        self.snap_btn.setChecked(True)
        self.snap_btn.toggled.connect(lambda checked: setattr(self, 'snapping_enabled', checked))
        toolbar_layout.addWidget(self.snap_btn)
        
        toolbar_layout.addStretch()
        
        # Split button
        self.split_btn = QPushButton("✂️ Split")
        self.split_btn.clicked.connect(self.split_current_clip)
        toolbar_layout.addWidget(self.split_btn)
        
        # Delete button
        self.delete_btn = QPushButton("🗑️ Delete")
        self.delete_btn.clicked.connect(self.delete_selected_clip)
        toolbar_layout.addWidget(self.delete_btn)
        
        # Position display
        self.position_label = QLabel("00:00:00:00")
        self.position_label.setFixedWidth(100)
        self.position_label.setStyleSheet(f"color: {COLORS['accent']}; font-weight: bold;")
        toolbar_layout.addWidget(self.position_label)
        
        return toolbar
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        QShortcut(QKeySequence("Ctrl+="), self, activated=lambda: self.set_zoom(self.zoom_level * 1.2))
        QShortcut(QKeySequence("Ctrl+-"), self, activated=lambda: self.set_zoom(self.zoom_level * 0.8))
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=lambda: self.set_position(self.current_time - 1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=lambda: self.set_position(self.current_time + 1))
        QShortcut(QKeySequence("Ctrl+K"), self, activated=self.split_current_clip)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, activated=self.delete_selected_clip)
    
    def set_media_items(self, media_items: Dict[str, MediaItem]):
        """Set media items for reference."""
        self.media_items = media_items
    
    def add_clip(self, media_item: MediaItem, track: int = None, time: float = None):
        """Add clip to timeline."""
        if time is None:
            time = self.current_time
        
        # Find appropriate track
        if track is None:
            if media_item.type == MediaType.VIDEO:
                for t in self.tracks:
                    if t.type == TrackType.VIDEO and not t.locked:
                        track = t.index
                        break
            elif media_item.type == MediaType.AUDIO:
                for t in self.tracks:
                    if t.type == TrackType.AUDIO and not t.locked:
                        track = t.index
                        break
            elif media_item.type == MediaType.IMAGE:
                for t in self.tracks:
                    if t.type == TrackType.VIDEO and not t.locked:
                        track = t.index
                        break
            elif media_item.type == MediaType.TEXT:
                for t in self.tracks:
                    if t.type == TrackType.TEXT and not t.locked:
                        track = t.index
                        break
        
        if track is None:
            return None
        
        # Determine track type
        track_type = TrackType.VIDEO
        if media_item.type == MediaType.AUDIO:
            track_type = TrackType.AUDIO
        elif media_item.type == MediaType.TEXT:
            track_type = TrackType.TEXT
        
        clip = TimelineClip(
            media_id=media_item.id,
            name=media_item.name,
            type=media_item.type,
            track_type=track_type,
            track=track,
            start=time,
            end=time + (media_item.duration or 5),
            media_start=0,
            media_end=media_item.duration or 5
        )
        
        self.clips.append(clip)
        self.track_view.set_clips(self.clips)
        self._update_duration()
        self.update()
        
        return clip
    
    def add_text_clip(self, text: str, duration: float = 5.0, track: int = None):
        """Add text clip to timeline."""
        if track is None:
            for t in self.tracks:
                if t.type == TrackType.TEXT and not t.locked:
                    track = t.index
                    break
        
        if track is None:
            return None
        
        # Create a temporary media item for text
        text_id = str(uuid.uuid4())
        
        clip = TimelineClip(
            id=text_id,
            media_id=text_id,
            name="Text: " + text[:20],
            type=MediaType.TEXT,
            track_type=TrackType.TEXT,
            track=track,
            start=self.current_time,
            end=self.current_time + duration,
            media_start=0,
            media_end=duration,
            text_content=text
        )
        
        self.clips.append(clip)
        self.track_view.set_clips(self.clips)
        self._update_duration()
        
        return clip
    
    def _update_duration(self):
        """Update timeline duration based on clips."""
        if self.clips:
            max_end = max(clip.end for clip in self.clips)
            self.duration = max(max_end, 60.0)
            self.ruler.set_duration(self.duration)
    
    def set_position(self, time: float):
        """Set current time position."""
        self.current_time = max(0, min(time, self.duration))
        self.position_label.setText(self._format_time(self.current_time))
        self.track_view.set_position(self.current_time)
        self.ruler.set_position(self.current_time)
        self.position_changed.emit(self.current_time)
    
    def set_zoom(self, level: float):
        """Set zoom level."""
        self.zoom_level = max(0.1, min(5.0, level))
        self.zoom_slider.setValue(int(self.zoom_level * 100))
        self.track_view.set_zoom(self.zoom_level)
        self.ruler.set_zoom(self.zoom_level)
    
    def split_current_clip(self):
        """Split selected clip at current position."""
        if self.selected_clip and self.current_time > self.selected_clip.start and self.current_time < self.selected_clip.end:
            clip1, clip2 = self.selected_clip.split(self.current_time)
            
            self.clips.remove(self.selected_clip)
            self.clips.append(clip1)
            self.clips.append(clip2)
            self.clips.sort(key=lambda c: c.start)
            
            self.track_view.set_clips(self.clips)
            self.update()
            self.clip_split.emit(self.selected_clip, self.current_time)
            self.selected_clip = None
    
    def delete_selected_clip(self):
        """Delete selected clip."""
        if self.selected_clip:
            self.clips.remove(self.selected_clip)
            self.track_view.set_clips(self.clips)
            self.update()
            self.selected_clip = None
            self.clip_selected.emit(None)
    
    def on_clip_selected(self, clip):
        """Handle clip selection."""
        self.selected_clip = clip
        self.clip_selected.emit(clip)
    
    def on_clip_moved(self, clip, new_start):
        """Handle clip movement."""
        self._update_duration()
        self.clip_moved.emit(clip, new_start)
    
    def on_track_muted(self, track_index: int, muted: bool):
        """Handle track mute."""
        if track_index < len(self.tracks):
            self.tracks[track_index].muted = muted
    
    def on_track_locked(self, track_index: int, locked: bool):
        """Handle track lock."""
        if track_index < len(self.tracks):
            self.tracks[track_index].locked = locked
    
    def _format_time(self, seconds: float) -> str:
        """Format time for display."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        frames = int((seconds - int(seconds)) * 30)  # Assuming 30fps for display
        return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"


class TimelineRuler(QWidget):
    """Timeline ruler with time markers."""
    
    position_changed = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self.setStyleSheet(f"background-color: {COLORS['bg_secondary']}; border-bottom: 1px solid {COLORS['border']};")
        
        self.duration = 60.0
        self.current_time = 0.0
        self.zoom_level = 1.0
        self.dragging = False
    
    def set_duration(self, duration: float):
        self.duration = duration
        self.update()
    
    def set_position(self, time: float):
        self.current_time = time
        self.update()
    
    def set_zoom(self, level: float):
        self.zoom_level = level
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Background
        painter.fillRect(0, 0, width, height, QColor(COLORS['bg_secondary']))
        
        # Time markers
        pixels_per_second = PIXELS_PER_SECOND * self.zoom_level
        
        # Determine interval based on zoom
        if self.zoom_level < 0.3:
            interval = 30
            sub_interval = 10
        elif self.zoom_level < 0.8:
            interval = 10
            sub_interval = 5
        elif self.zoom_level < 1.5:
            interval = 5
            sub_interval = 1
        else:
            interval = 1
            sub_interval = 0.5
        
        painter.setPen(QPen(QColor(COLORS['text_secondary']), 1))
        
        time = 0
        while time <= self.duration:
            x = time * pixels_per_second
            if x > width:
                break
            
            is_major = abs(time % interval) < 0.001
            
            if is_major:
                painter.setPen(QPen(QColor(COLORS['text_primary']), 2))
                painter.drawLine(int(x), height - 10, int(x), height)
                
                time_str = f"{int(time // 60):02d}:{int(time % 60):02d}"
                painter.setPen(QPen(QColor(COLORS['text_primary']), 1))
                painter.drawText(int(x) + 5, 15, time_str)
            else:
                if abs(time % sub_interval) < 0.001:
                    painter.setPen(QPen(QColor(COLORS['text_secondary']), 1))
                    painter.drawLine(int(x), height - 5, int(x), height)
            
            time += 0.5 if self.zoom_level > 1.5 else 1
        
        # Playhead
        playhead_x = self.current_time * pixels_per_second
        painter.setPen(QPen(QColor(COLORS['playhead']), 2))
        painter.drawLine(int(playhead_x), 0, int(playhead_x), height)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.mouseMoveEvent(event)
    
    def mouseMoveEvent(self, event):
        if self.dragging:
            pixels_per_second = PIXELS_PER_SECOND * self.zoom_level
            time = event.position().x() / pixels_per_second
            time = max(0, min(time, self.duration))
            self.position_changed.emit(time)
    
    def mouseReleaseEvent(self, event):
        self.dragging = False


class TrackHeaders(QWidget):
    """Track headers for timeline."""
    
    track_muted = pyqtSignal(int, bool)
    track_locked = pyqtSignal(int, bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(120)
        self.setStyleSheet(f"""
            background-color: {COLORS['bg_secondary']};
            border-right: 1px solid {COLORS['border']};
        """)
        
        self.tracks = []
    
    def set_tracks(self, tracks):
        self.tracks = tracks
        self.setFixedHeight(len(tracks) * TIMELINE_TRACK_HEIGHT)
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        
        y = 0
        for i, track in enumerate(self.tracks):
            height = track.height
            
            # Background
            painter.fillRect(0, y, self.width(), height, QColor(COLORS['bg_secondary']))
            
            # Border
            painter.setPen(QPen(QColor(COLORS['border']), 1))
            painter.drawLine(0, y + height - 1, self.width(), y + height - 1)
            
            # Icon and name
            if track.type == TrackType.VIDEO:
                icon = "🎬"
            elif track.type == TrackType.AUDIO:
                icon = "🔊"
            else:
                icon = "📝"
            
            painter.setPen(QPen(QColor(COLORS['text_primary']), 1))
            painter.drawText(10, y + height//2 + 5, f"{icon} {track.name}")
            
            # Mute/Lock indicators
            if track.muted:
                painter.setPen(QPen(QColor(COLORS['accent']), 1))
                painter.drawText(self.width() - 40, y + height//2 + 5, "🔇")
            
            if track.locked:
                painter.setPen(QPen(QColor(COLORS['accent']), 1))
                painter.drawText(self.width() - 20, y + height//2 + 5, "🔒")
            
            y += height
    
    def mousePressEvent(self, event):
        pos = event.position()
        y = 0
        
        for i, track in enumerate(self.tracks):
            if y <= pos.y() < y + track.height:
                if pos.x() > self.width() - 40:
                    track.muted = not track.muted
                    self.track_muted.emit(i, track.muted)
                elif pos.x() > self.width() - 20:
                    track.locked = not track.locked
                    self.track_locked.emit(i, track.locked)
                self.update()
                break
            y += track.height


class TrackView(QWidget):
    """Main timeline track view."""
    
    clip_selected = pyqtSignal(object)
    position_changed = pyqtSignal(float)
    clip_moved = pyqtSignal(object, float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {COLORS['bg_primary']};")
        
        self.clips = []
        self.tracks = []
        self.media_items = {}
        self.duration = 60.0
        self.current_time = 0.0
        self.zoom_level = 1.0
        
        self.dragging = False
        self.drag_clip = None
        self.drag_offset = 0
        self.resizing = False
        self.resize_edge = None
        
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    def set_tracks(self, tracks):
        self.tracks = tracks
        self.setFixedHeight(len(tracks) * TIMELINE_TRACK_HEIGHT)
        self.update()
    
    def set_clips(self, clips):
        self.clips = clips
        self.update()
    
    def set_position(self, time: float):
        self.current_time = time
        self.update()
    
    def set_zoom(self, level: float):
        self.zoom_level = level
        self.update()
    
    def get_track_y(self, track_index: int) -> int:
        y = 0
        for i, track in enumerate(self.tracks):
            if i == track_index:
                return y
            y += track.height
        return 0
    
    def time_to_pixels(self, time: float) -> float:
        return time * PIXELS_PER_SECOND * self.zoom_level
    
    def pixels_to_time(self, x: float) -> float:
        return x / (PIXELS_PER_SECOND * self.zoom_level)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Background
        painter.fillRect(0, 0, width, height, QColor(COLORS['bg_primary']))
        
        # Time grid
        painter.setPen(QPen(QColor(COLORS['bg_tertiary']), 1, Qt.PenStyle.DotLine))
        pixels_per_second = PIXELS_PER_SECOND * self.zoom_level
        
        for second in range(0, int(self.duration) + 1):
            x = second * pixels_per_second
            if x > width:
                break
            painter.drawLine(int(x), 0, int(x), height)
        
        # Draw tracks and clips
        y = 0
        for i, track in enumerate(self.tracks):
            track_height = track.height
            
            # Track background (alternating)
            if i % 2 == 0:
                painter.fillRect(0, y, width, track_height, QColor(COLORS['bg_secondary']))
            
            # Track border
            painter.setPen(QPen(QColor(COLORS['border']), 1))
            painter.drawLine(0, y + track_height - 1, width, y + track_height - 1)
            
            # Draw clips
            for clip in self.clips:
                if clip.track == i and not track.locked:
                    self._draw_clip(painter, clip, y, track_height)
            
            y += track_height
        
        # Draw playhead
        playhead_x = self.time_to_pixels(self.current_time)
        painter.setPen(QPen(QColor(COLORS['playhead']), 2))
        painter.drawLine(int(playhead_x), 0, int(playhead_x), height)
    
    def _draw_clip(self, painter: QPainter, clip: TimelineClip, track_y: int, track_height: int):
        """Draw a single clip."""
        x = self.time_to_pixels(clip.start)
        width = self.time_to_pixels(clip.duration)
        
        if x + width < 0 or x > self.width():
            return
        
        # Clip colors based on type
        if clip.type == MediaType.VIDEO:
            color = QColor(COLORS['video_track'])
        elif clip.type == MediaType.AUDIO:
            color = QColor(COLORS['audio_track'])
        elif clip.type == MediaType.TEXT:
            color = QColor(COLORS['text_track'])
        else:
            color = QColor(COLORS['accent'])
        
        # Draw clip
        gradient = QLinearGradient(x, track_y, x, track_y + track_height - 4)
        gradient.setColorAt(0, color.lighter(120))
        gradient.setColorAt(1, color)
        
        painter.fillRect(int(x), track_y + 2, int(width), track_height - 4, gradient)
        painter.setPen(QPen(color.lighter(150), 1))
        painter.drawRect(int(x), track_y + 2, int(width), track_height - 4)
        
        # Clip name
        painter.setPen(QPen(Qt.GlobalColor.white, 1))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        
        name = clip.name
        text_width = painter.fontMetrics().horizontalAdvance(name)
        if text_width > width - 10:
            while text_width > width - 10 and len(name) > 3:
                name = name[:-1]
                text_width = painter.fontMetrics().horizontalAdvance(name + "...")
            name += "..."
        
        painter.drawText(int(x) + 5, track_y + track_height//2 + 3, name)
        
        # Duration
        duration_str = f"{clip.duration:.1f}s"
        painter.setPen(QPen(QColor(COLORS['text_secondary']), 1))
        painter.drawText(int(x) + 5, track_y + track_height - 8, duration_str)
        
        # Handles for selected clip
        if clip == self.parent().selected_clip:
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            # Left handle
            painter.drawLine(int(x), track_y + track_height//2 - 5, int(x) + 5, track_y + track_height//2)
            painter.drawLine(int(x), track_y + track_height//2 + 5, int(x) + 5, track_y + track_height//2)
            # Right handle
            painter.drawLine(int(x + width - 5), track_y + track_height//2, int(x + width), track_y + track_height//2 - 5)
            painter.drawLine(int(x + width - 5), track_y + track_height//2, int(x + width), track_y + track_height//2 + 5)
    
    def find_clip_at_pos(self, pos: QPointF) -> Tuple[Optional[TimelineClip], Optional[str]]:
        """Find clip at position."""
        for clip in reversed(self.clips):
            track_y = self.get_track_y(clip.track)
            track = self.tracks[clip.track] if clip.track < len(self.tracks) else None
            
            if track and track.locked:
                continue
            
            track_height = track.height if track else TIMELINE_TRACK_HEIGHT
            
            x = self.time_to_pixels(clip.start)
            width = self.time_to_pixels(clip.duration)
            
            clip_rect = QRectF(x, track_y + 2, width, track_height - 4)
            
            if clip_rect.contains(pos):
                if abs(pos.x() - x) < 10:
                    return clip, 'left'
                elif abs(pos.x() - (x + width)) < 10:
                    return clip, 'right'
                else:
                    return clip, None
        
        return None, None
    
    def mousePressEvent(self, event):
        pos = event.position()
        
        clip, edge = self.find_clip_at_pos(pos)
        
        if clip:
            if edge:
                self.resizing = True
                self.drag_clip = clip
                self.resize_edge = edge
                self.drag_offset = pos.x() - self.time_to_pixels(clip.start)
            else:
                self.dragging = True
                self.drag_clip = clip
                self.drag_offset = pos.x() - self.time_to_pixels(clip.start)
            
            self.clip_selected.emit(clip)
        else:
            self.clip_selected.emit(None)
    
    def mouseMoveEvent(self, event):
        pos = event.position()
        parent = self.parent()
        
        if self.dragging and self.drag_clip and parent:
            new_x = pos.x() - self.drag_offset
            new_start = self.pixels_to_time(new_x)
            new_start = max(0, new_start)
            
            # Snapping
            if parent.snapping_enabled:
                snap_positions = []
                for clip in self.clips:
                    if clip != self.drag_clip:
                        snap_positions.append(clip.start)
                        snap_positions.append(clip.end)
                snap_positions.append(self.current_time)
                
                min_dist = float('inf')
                best_pos = new_start
                
                for snap_time in snap_positions:
                    snap_x = self.time_to_pixels(snap_time)
                    dist = abs(new_x - snap_x)
                    if dist < parent.snap_threshold and dist < min_dist:
                        min_dist = dist
                        best_pos = snap_time
                
                new_start = best_pos
            
            duration = self.drag_clip.duration
            self.drag_clip.start = new_start
            self.drag_clip.end = new_start + duration
            
            self.clip_moved.emit(self.drag_clip, new_start)
            self.update()
        
        elif self.resizing and self.drag_clip:
            new_x = pos.x()
            new_time = self.pixels_to_time(new_x)
            
            if self.resize_edge == 'left':
                if new_time < self.drag_clip.end:
                    self.drag_clip.start = new_time
            else:
                if new_time > self.drag_clip.start:
                    self.drag_clip.end = new_time
            
            self.update()
        
        else:
            clip, edge = self.find_clip_at_pos(pos)
            if edge:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif clip:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.resizing = False
        self.drag_clip = None
        self.setCursor(Qt.CursorShape.ArrowCursor)


# ==================== MEDIA POOL WIDGET ====================

class MediaPoolWidget(QWidget):
    """Media pool for managing imported media."""
    
    media_selected = pyqtSignal(object)
    media_added = pyqtSignal(object)
    media_double_clicked = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.media_items = {}
        self.loader_thread = MediaLoaderThread()
        self.loader_thread.metadata_loaded.connect(self.on_metadata_loaded)
        self.loader_thread.thumbnail_ready.connect(self.on_thumbnail_ready)
        self.loader_thread.error.connect(self.on_load_error)
        self.loader_thread.progress.connect(self.on_load_progress)
        self.loader_thread.start()
        
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_secondary']};
                border-right: 1px solid {COLORS['border']};
            }}
            QPushButton {{
                background-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_primary']};
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_hover']};
            }}
            QListWidget {{
                background-color: {COLORS['bg_primary']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                outline: none;
            }}
            QListWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {COLORS['border']};
            }}
            QListWidget::item:hover {{
                background-color: {COLORS['bg_hover']};
            }}
            QListWidget::item:selected {{
                background-color: {COLORS['accent']};
                color: white;
            }}
            QProgressBar {{
                border: none;
                background-color: {COLORS['bg_tertiary']};
                border-radius: 2px;
                height: 4px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['accent']};
                border-radius: 2px;
            }}
        """)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup media pool UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header
        header = QLabel("📁 Media Pool")
        header.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['text_primary']};")
        layout.addWidget(header)
        
        # Import buttons
        btn_layout = QHBoxLayout()
        
        self.import_video_btn = QPushButton("🎬 Video")
        self.import_video_btn.clicked.connect(lambda: self.import_media('video'))
        btn_layout.addWidget(self.import_video_btn)
        
        self.import_audio_btn = QPushButton("🎵 Audio")
        self.import_audio_btn.clicked.connect(lambda: self.import_media('audio'))
        btn_layout.addWidget(self.import_audio_btn)
        
        self.import_image_btn = QPushButton("🖼️ Image")
        self.import_image_btn.clicked.connect(lambda: self.import_media('image'))
        btn_layout.addWidget(self.import_image_btn)
        
        layout.addLayout(btn_layout)
        
        # Media list
        self.media_list = QListWidget()
        self.media_list.setIconSize(THUMBNAIL_SIZE)
        self.media_list.setSpacing(2)
        self.media_list.itemClicked.connect(self.on_item_clicked)
        self.media_list.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.media_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.media_list.customContextMenuRequested.connect(self.show_context_menu)
        
        layout.addWidget(self.media_list)
    
    def import_media(self, media_type: str):
        """Import media files."""
        if media_type == 'video':
            filter_str = f"Video Files (*{' *'.join(VIDEO_FORMATS)});;All Files (*.*)"
        elif media_type == 'audio':
            filter_str = f"Audio Files (*{' *'.join(AUDIO_FORMATS)});;All Files (*.*)"
        elif media_type == 'image':
            filter_str = f"Image Files (*{' *'.join(IMAGE_FORMATS)});;All Files (*.*)"
        else:
            filter_str = "All Files (*.*)"
        
        files, _ = QFileDialog.getOpenFileNames(
            self,
            f"Import {media_type.capitalize()} Files",
            "",
            filter_str
        )
        
        if files:
            for file_path in files:
                self.add_media(file_path)
    
    def add_media(self, file_path: str) -> Optional[MediaItem]:
        """Add media to pool."""
        if not os.path.exists(file_path):
            return None
        
        file_size = os.path.getsize(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        
        # Determine media type
        if ext in VIDEO_FORMATS:
            media_type = MediaType.VIDEO
        elif ext in AUDIO_FORMATS:
            media_type = MediaType.AUDIO
        elif ext in IMAGE_FORMATS:
            media_type = MediaType.IMAGE
        else:
            return None
        
        # Warning for large files
        if file_size > 1024 * 1024 * 1024:  # > 1GB
            reply = QMessageBox.question(
                self,
                "Large File Detected",
                f"The file is {file_size / (1024*1024*1024):.1f}GB in size.\n\n"
                "Loading large files may take a moment.\n\n"
                "Do you want to continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                return None
        
        media_item = MediaItem(
            id=str(uuid.uuid4()),
            name=os.path.basename(file_path),
            file_path=file_path,
            type=media_type,
            file_size=file_size,
            is_loading=True
        )
        
        self.media_items[media_item.id] = media_item
        
        # Create list item
        item = QListWidgetItem(f"⏳ Loading: {media_item.name}")
        item.setData(Qt.ItemDataRole.UserRole, media_item.id)
        self.media_list.addItem(item)
        
        # Start loading
        self.loader_thread.add_job(media_item.id, file_path, media_type)
        
        return media_item
    
    def on_metadata_loaded(self, media_id: str, media_item: MediaItem):
        """Handle loaded metadata."""
        if media_id in self.media_items:
            self.media_items[media_id] = media_item
            self.media_added.emit(media_item)
    
    def on_thumbnail_ready(self, media_id: str, pixmap: QPixmap):
        """Handle thumbnail ready."""
        if media_id in self.media_items:
            media_item = self.media_items[media_id]
            media_item.thumbnail = pixmap
            media_item.is_loading = False
            
            # Update list item
            for i in range(self.media_list.count()):
                item = self.media_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == media_id:
                    item.setIcon(QIcon(pixmap))
                    item.setText(media_item.name)
                    break
    
    def on_load_progress(self, media_id: str, progress: int):
        """Handle load progress."""
        if media_id in self.media_items:
            for i in range(self.media_list.count()):
                item = self.media_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == media_id:
                    media_item = self.media_items[media_id]
                    if progress < 100:
                        item.setText(f"⏳ Loading: {media_item.name} ({progress}%)")
                    break
    
    def on_load_error(self, media_id: str, error: str):
        """Handle load error."""
        if media_id in self.media_items:
            media_item = self.media_items[media_id]
            media_item.is_loading = False
            media_item.load_error = error
            
            for i in range(self.media_list.count()):
                item = self.media_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == media_id:
                    item.setText(f"❌ Failed: {media_item.name}")
                    item.setToolTip(f"Error: {error}")
                    break
            
            QMessageBox.warning(self, "Load Error", f"Failed to load {media_item.name}:\n{error}")
    
    def on_item_clicked(self, item: QListWidgetItem):
        """Handle item click."""
        media_id = item.data(Qt.ItemDataRole.UserRole)
        if media_id in self.media_items:
            media_item = self.media_items[media_id]
            if not media_item.is_loading and not media_item.load_error:
                self.media_selected.emit(media_item)
    
    def on_item_double_clicked(self, item: QListWidgetItem):
        """Handle double click."""
        media_id = item.data(Qt.ItemDataRole.UserRole)
        if media_id in self.media_items:
            media_item = self.media_items[media_id]
            if not media_item.is_loading and not media_item.load_error:
                self.media_double_clicked.emit(media_item)
    
    def show_context_menu(self, position):
        """Show context menu."""
        item = self.media_list.itemAt(position)
        if not item:
            return
        
        menu = QMenu()
        
        add_to_timeline = menu.addAction("➕ Add to Timeline")
        show_properties = menu.addAction("ℹ️ Properties")
        menu.addSeparator()
        remove = menu.addAction("🗑️ Remove")
        
        action = menu.exec(self.media_list.mapToGlobal(position))
        
        if action == add_to_timeline:
            media_id = item.data(Qt.ItemDataRole.UserRole)
            if media_id in self.media_items:
                self.media_double_clicked.emit(self.media_items[media_id])
        
        elif action == show_properties:
            media_id = item.data(Qt.ItemDataRole.UserRole)
            if media_id in self.media_items:
                self.show_properties(self.media_items[media_id])
        
        elif action == remove:
            media_id = item.data(Qt.ItemDataRole.UserRole)
            if media_id in self.media_items:
                del self.media_items[media_id]
                self.media_list.takeItem(self.media_list.row(item))
    
    def show_properties(self, media_item: MediaItem):
        """Show media properties dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Media Properties")
        dialog.setMinimumWidth(400)
        
        layout = QFormLayout(dialog)
        
        layout.addRow("Name:", QLabel(media_item.name))
        layout.addRow("Type:", QLabel(media_item.type.value))
        layout.addRow("Path:", QLabel(media_item.file_path))
        layout.addRow("Size:", QLabel(media_item.size_str))
        
        if media_item.type == MediaType.VIDEO:
            layout.addRow("Duration:", QLabel(media_item.duration_str))
            layout.addRow("Resolution:", QLabel(media_item.resolution_str))
            layout.addRow("FPS:", QLabel(f"{media_item.fps:.2f}"))
            layout.addRow("Codec:", QLabel(media_item.codec))
            layout.addRow("Has Audio:", QLabel("Yes" if media_item.has_audio else "No"))
        
        elif media_item.type == MediaType.AUDIO:
            layout.addRow("Duration:", QLabel(media_item.duration_str))
            layout.addRow("Channels:", QLabel(str(media_item.audio_channels)))
            layout.addRow("Sample Rate:", QLabel(f"{media_item.audio_sample_rate} Hz"))
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addRow(buttons)
        
        dialog.exec()
    
    def get_media_items(self) -> Dict[str, MediaItem]:
        """Get all media items."""
        return self.media_items
    
    def closeEvent(self, event):
        """Clean up on close."""
        self.loader_thread.stop()
        self.loader_thread.wait()
        super().closeEvent(event)


# ==================== INSPECTOR WIDGET ====================

class InspectorWidget(QWidget):
    """Inspector panel for clip properties."""
    
    clip_updated = pyqtSignal(object)
    effect_applied = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumWidth(350)
        self.setMinimumWidth(300)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_secondary']};
                border-left: 1px solid {COLORS['border']};
            }}
            QGroupBox {{
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                margin-top: 10px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                color: {COLORS['accent']};
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
            QLabel {{
                color: {COLORS['text_secondary']};
            }}
            QSlider::groove:horizontal {{
                height: 4px;
                background: {COLORS['bg_tertiary']};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS['accent']};
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }}
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
                background-color: {COLORS['bg_primary']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 4px;
            }}
            QPushButton {{
                background-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_primary']};
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_hover']};
            }}
            QTabWidget::pane {{
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                background-color: {COLORS['bg_primary']};
            }}
            QTabBar::tab {{
                background-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_secondary']};
                padding: 6px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {COLORS['bg_primary']};
                color: {COLORS['accent']};
            }}
        """)
        
        self.current_clip = None
        self.current_media = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup inspector UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("📋 Inspector")
        title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['text_primary']};")
        layout.addWidget(title)
        
        # Tabs
        self.tabs = QTabWidget()
        
        # Video tab
        self.video_tab = self.create_video_tab()
        self.tabs.addTab(self.video_tab, "Video")
        
        # Audio tab
        self.audio_tab = self.create_audio_tab()
        self.tabs.addTab(self.audio_tab, "Audio")
        
        # Effects tab
        self.effects_tab = self.create_effects_tab()
        self.tabs.addTab(self.effects_tab, "Effects")
        
        # Text tab
        self.text_tab = self.create_text_tab()
        self.tabs.addTab(self.text_tab, "Text")
        
        layout.addWidget(self.tabs)
    
    def create_video_tab(self) -> QWidget:
        """Create video properties tab."""
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setVerticalSpacing(10)
        
        # Speed
        speed_layout = QHBoxLayout()
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 4.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(1.0)
        self.speed_spin.valueChanged.connect(self.on_speed_changed)
        speed_layout.addWidget(self.speed_spin)
        speed_layout.addWidget(QLabel("x"))
        layout.addRow("Speed:", speed_layout)
        
        # Transform
        self.pos_x = QSpinBox()
        self.pos_x.setRange(-1000, 1000)
        self.pos_x.valueChanged.connect(self.on_transform_changed)
        layout.addRow("Position X:", self.pos_x)
        
        self.pos_y = QSpinBox()
        self.pos_y.setRange(-1000, 1000)
        self.pos_y.valueChanged.connect(self.on_transform_changed)
        layout.addRow("Position Y:", self.pos_y)
        
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.1, 5.0)
        self.scale.setSingleStep(0.1)
        self.scale.setValue(1.0)
        self.scale.valueChanged.connect(self.on_transform_changed)
        layout.addRow("Scale:", self.scale)
        
        self.rotation = QSpinBox()
        self.rotation.setRange(-180, 180)
        self.rotation.valueChanged.connect(self.on_transform_changed)
        layout.addRow("Rotation:", self.rotation)
        
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(100)
        self.opacity.valueChanged.connect(self.on_opacity_changed)
        
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(self.opacity)
        self.opacity_label = QLabel("100%")
        opacity_layout.addWidget(self.opacity_label)
        layout.addRow("Opacity:", opacity_layout)
        
        return tab
    
    def create_audio_tab(self) -> QWidget:
        """Create audio properties tab."""
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setVerticalSpacing(10)
        
        # Volume
        volume_layout = QHBoxLayout()
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 200)
        self.volume_slider.setValue(100)
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        volume_layout.addWidget(self.volume_slider)
        
        self.volume_label = QLabel("100%")
        self.volume_label.setFixedWidth(40)
        volume_layout.addWidget(self.volume_label)
        layout.addRow("Volume:", volume_layout)
        
        # Pan
        pan_layout = QHBoxLayout()
        self.pan_slider = QSlider(Qt.Orientation.Horizontal)
        self.pan_slider.setRange(-100, 100)
        self.pan_slider.setValue(0)
        self.pan_slider.valueChanged.connect(self.on_pan_changed)
        pan_layout.addWidget(self.pan_slider)
        
        self.pan_label = QLabel("C")
        self.pan_label.setFixedWidth(40)
        pan_layout.addWidget(self.pan_label)
        layout.addRow("Pan:", pan_layout)
        
        # Mute
        self.mute_check = QCheckBox("Mute")
        self.mute_check.toggled.connect(self.on_mute_toggled)
        layout.addRow("", self.mute_check)
        
        return tab
    
    def create_effects_tab(self) -> QWidget:
        """Create effects tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Effect list
        self.effect_list = QListWidget()
        self.effect_list.addItems([
            "Brightness",
            "Contrast",
            "Grayscale",
            "Sepia",
            "Blur",
            "Sharpen",
            "Invert"
        ])
        self.effect_list.itemDoubleClicked.connect(self.add_effect)
        layout.addWidget(QLabel("Double-click to add effect:"))
        layout.addWidget(self.effect_list)
        
        # Applied effects
        layout.addWidget(QLabel("Applied effects:"))
        self.applied_effects = QListWidget()
        self.applied_effects.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.applied_effects.customContextMenuRequested.connect(self.remove_effect_menu)
        layout.addWidget(self.applied_effects)
        
        # Background removal
        bg_group = QGroupBox("Background")
        bg_layout = QVBoxLayout()
        
        self.bg_combo = QComboBox()
        self.bg_combo.addItems(["None", "Remove Background", "Green Screen", "Replace Background"])
        self.bg_combo.currentTextChanged.connect(self.on_bg_mode_changed)
        bg_layout.addWidget(self.bg_combo)
        
        self.bg_color_btn = QPushButton("Select Green Screen Color")
        self.bg_color_btn.clicked.connect(self.select_bg_color)
        bg_layout.addWidget(self.bg_color_btn)
        
        self.bg_image_btn = QPushButton("Select Background Image")
        self.bg_image_btn.clicked.connect(self.select_bg_image)
        bg_layout.addWidget(self.bg_image_btn)
        
        bg_group.setLayout(bg_layout)
        layout.addWidget(bg_group)
        
        layout.addStretch()
        
        return tab
    
    def create_text_tab(self) -> QWidget:
        """Create text properties tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Text content
        layout.addWidget(QLabel("Text Content:"))
        self.text_edit = QTextEdit()
        self.text_edit.setMaximumHeight(100)
        self.text_edit.textChanged.connect(self.on_text_changed)
        layout.addWidget(self.text_edit)
        
        # Text properties
        props_layout = QFormLayout()
        
        self.font_combo = QComboBox()
        self.font_combo.addItems(["Arial", "Times New Roman", "Courier New", "Helvetica"])
        self.font_combo.currentTextChanged.connect(self.on_text_property_changed)
        props_layout.addRow("Font:", self.font_combo)
        
        self.font_size = QSpinBox()
        self.font_size.setRange(8, 200)
        self.font_size.setValue(24)
        self.font_size.valueChanged.connect(self.on_text_property_changed)
        props_layout.addRow("Size:", self.font_size)
        
        self.text_color_btn = QPushButton("Select Color")
        self.text_color_btn.clicked.connect(self.select_text_color)
        props_layout.addRow("Color:", self.text_color_btn)
        
        layout.addLayout(props_layout)
        
        # Add text button
        self.add_text_btn = QPushButton("➕ Add Text to Timeline")
        self.add_text_btn.clicked.connect(self.add_text_to_timeline)
        layout.addWidget(self.add_text_btn)
        
        layout.addStretch()
        
        return tab
    
    def set_clip(self, clip: Optional[TimelineClip], media_item: Optional[MediaItem] = None):
        """Set current clip."""
        self.current_clip = clip
        self.current_media = media_item
        
        if clip:
            # Update video properties
            self.speed_spin.setValue(clip.speed)
            
            if hasattr(clip, 'transform'):
                self.pos_x.setValue(clip.transform.get('x', 0))
                self.pos_y.setValue(clip.transform.get('y', 0))
                self.scale.setValue(clip.transform.get('scale', 1.0))
                self.rotation.setValue(clip.transform.get('rotation', 0))
                opacity = clip.transform.get('opacity', 1.0)
                self.opacity.setValue(int(opacity * 100))
                self.opacity_label.setText(f"{int(opacity * 100)}%")
            
            # Update audio properties
            self.volume_slider.setValue(int(clip.volume * 100))
            self.volume_label.setText(f"{int(clip.volume * 100)}%")
            self.pan_slider.setValue(int(clip.pan * 100))
            self.pan_label.setText(self._get_pan_text(clip.pan))
            self.mute_check.setChecked(clip.muted)
            
            # Update text properties
            self.text_edit.setPlainText(clip.text_content)
            
            self.set_enabled(True)
        else:
            self.set_enabled(False)
    
    def set_enabled(self, enabled: bool):
        """Enable/disable controls."""
        self.speed_spin.setEnabled(enabled)
        self.pos_x.setEnabled(enabled)
        self.pos_y.setEnabled(enabled)
        self.scale.setEnabled(enabled)
        self.rotation.setEnabled(enabled)
        self.opacity.setEnabled(enabled)
        self.volume_slider.setEnabled(enabled)
        self.pan_slider.setEnabled(enabled)
        self.mute_check.setEnabled(enabled)
        self.text_edit.setEnabled(enabled)
        self.font_combo.setEnabled(enabled)
        self.font_size.setEnabled(enabled)
        self.text_color_btn.setEnabled(enabled)
        self.add_text_btn.setEnabled(enabled)
    
    def _get_pan_text(self, pan: float) -> str:
        """Get pan display text."""
        if pan < 0:
            return f"L {abs(int(pan * 100))}%"
        elif pan > 0:
            return f"R {int(pan * 100)}%"
        else:
            return "C"
    
    def on_speed_changed(self, value):
        """Handle speed change."""
        if self.current_clip:
            self.current_clip.speed = value
            self.clip_updated.emit(self.current_clip)
    
    def on_transform_changed(self):
        """Handle transform change."""
        if self.current_clip:
            self.current_clip.transform = {
                'x': self.pos_x.value(),
                'y': self.pos_y.value(),
                'scale': self.scale.value(),
                'rotation': self.rotation.value(),
                'opacity': self.opacity.value() / 100.0
            }
            self.clip_updated.emit(self.current_clip)
    
    def on_opacity_changed(self, value):
        """Handle opacity change."""
        self.opacity_label.setText(f"{value}%")
        self.on_transform_changed()
    
    def on_volume_changed(self, value):
        """Handle volume change."""
        if self.current_clip:
            self.current_clip.volume = value / 100.0
            self.volume_label.setText(f"{value}%")
            self.clip_updated.emit(self.current_clip)
    
    def on_pan_changed(self, value):
        """Handle pan change."""
        if self.current_clip:
            self.current_clip.pan = value / 100.0
            self.pan_label.setText(self._get_pan_text(self.current_clip.pan))
            self.clip_updated.emit(self.current_clip)
    
    def on_mute_toggled(self, checked):
        """Handle mute toggle."""
        if self.current_clip:
            self.current_clip.muted = checked
            self.clip_updated.emit(self.current_clip)
    
    def add_effect(self, item):
        """Add effect to clip."""
        effect_name = item.text()
        effect_item = QListWidgetItem(effect_name)
        self.applied_effects.addItem(effect_item)
        
        if self.current_clip:
            effect = {
                'type': effect_name.lower(),
                'intensity': 1.0
            }
            self.current_clip.effects.append(effect)
            self.effect_applied.emit(effect)
    
    def remove_effect_menu(self, position):
        """Show context menu to remove effect."""
        item = self.applied_effects.itemAt(position)
        if item:
            menu = QMenu()
            remove_action = menu.addAction("Remove Effect")
            action = menu.exec(self.applied_effects.mapToGlobal(position))
            
            if action == remove_action:
                row = self.applied_effects.row(item)
                self.applied_effects.takeItem(row)
                if self.current_clip and row < len(self.current_clip.effects):
                    del self.current_clip.effects[row]
    
    def on_bg_mode_changed(self, mode):
        """Handle background mode change."""
        if mode == "None":
            self.effect_applied.emit({'bg_mode': 'none'})
        elif mode == "Remove Background":
            self.effect_applied.emit({'bg_mode': 'remove'})
        elif mode == "Green Screen":
            self.effect_applied.emit({'bg_mode': 'green_screen', 'color': (0, 255, 0)})
        elif mode == "Replace Background":
            self.effect_applied.emit({'bg_mode': 'replace'})
    
    def select_bg_color(self):
        """Select green screen color."""
        color = QColorDialog.getColor(Qt.GlobalColor.green, self, "Select Green Screen Color")
        if color.isValid():
            self.effect_applied.emit({
                'bg_mode': 'green_screen',
                'color': (color.red(), color.green(), color.blue())
            })
    
    def select_bg_image(self):
        """Select background image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Background Image",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            self.effect_applied.emit({'bg_mode': 'replace', 'image': file_path})
    
    def on_text_changed(self):
        """Handle text content change."""
        if self.current_clip:
            self.current_clip.text_content = self.text_edit.toPlainText()
            self.clip_updated.emit(self.current_clip)
    
    def on_text_property_changed(self):
        """Handle text property change."""
        if self.current_clip:
            # Store text properties
            self.current_clip.text_font = self.font_combo.currentText()
            self.current_clip.text_size = self.font_size.value()
            self.clip_updated.emit(self.current_clip)
    
    def select_text_color(self):
        """Select text color."""
        color = QColorDialog.getColor(Qt.GlobalColor.white, self, "Select Text Color")
        if color.isValid():
            self.text_color_btn.setStyleSheet(f"background-color: {color.name()};")
            if self.current_clip:
                self.current_clip.text_color = color.name()
                self.clip_updated.emit(self.current_clip)
    
    def add_text_to_timeline(self):
        """Add text clip to timeline."""
        if hasattr(self.parent(), 'timeline') and self.text_edit.toPlainText():
            self.parent().timeline.add_text_clip(self.text_edit.toPlainText())


# ==================== EMPTY STATE WIDGET ====================

class EmptyStateWidget(QWidget):
    """Initial empty state with Import button."""
    
    import_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_primary']};
            }}
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 15px 40px;
                min-width: 200px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
            }}
            QLabel {{
                color: {COLORS['text_secondary']};
                font-size: 14px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)
        
        # Icon
        icon_label = QLabel("🎬")
        icon_label.setStyleSheet("font-size: 64px;")
        layout.addWidget(icon_label)
        
        # Title
        title = QLabel("No Video Loaded")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #e4e4e7;")
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Import a video to start editing")
        layout.addWidget(desc)
        
        # Import button
        self.import_btn = QPushButton("📁 Import Video")
        self.import_btn.clicked.connect(self.import_clicked)
        layout.addWidget(self.import_btn)


# ==================== VIDEO PREVIEW WIDGET ====================

class VideoPreviewWidget(QWidget):
    """Video preview with controls."""
    
    position_changed = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_primary']};
                border: none;
                font-size: 18px;
                min-width: 36px;
                min-height: 36px;
                border-radius: 18px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_hover']};
            }}
            QSlider::groove:horizontal {{
                height: 4px;
                background: {COLORS['bg_tertiary']};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {COLORS['accent']};
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }}
            QLabel {{
                color: {COLORS['text_secondary']};
                font-size: 12px;
            }}
            QProgressBar {{
                border: none;
                background-color: {COLORS['bg_tertiary']};
                border-radius: 4px;
                text-align: center;
                color: white;
                height: 20px;
            }}
        """)
        
        self.video_path = None
        self.duration = 0.0
        self.current_time = 0.0
        self.playing = False
        self.was_playing = False
        self.is_loading = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup preview UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Video display - remove borders
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #000000; border: none;")
        self.video_label.setMinimumHeight(250)
        layout.addWidget(self.video_label, 1)
        
        # Controls
        controls = QWidget()
        controls.setFixedHeight(CONTROL_HEIGHT)
        controls.setStyleSheet(f"background-color: {COLORS['bg_primary']}; border-top: 1px solid {COLORS['border']};")
        
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(10, 5, 10, 5)
        controls_layout.setSpacing(10)
        
        # Playback controls
        self.prev_btn = QPushButton("⏮")
        self.prev_btn.clicked.connect(self.prev_frame)
        controls_layout.addWidget(self.prev_btn)
        
        self.play_btn = QPushButton("▶")
        self.play_btn.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_btn)
        
        self.next_btn = QPushButton("⏭")
        self.next_btn.clicked.connect(self.next_frame)
        controls_layout.addWidget(self.next_btn)
        
        # Time slider
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 1000)
        self.time_slider.sliderMoved.connect(self.slider_moved)
        self.time_slider.sliderPressed.connect(self.slider_pressed)
        self.time_slider.sliderReleased.connect(self.slider_released)
        controls_layout.addWidget(self.time_slider, 1)
        
        # Time label
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setFixedWidth(120)
        controls_layout.addWidget(self.time_label)
        
        layout.addWidget(controls)
    
    def load_video(self, file_path: str):
        """Load video."""
        self.video_path = file_path
        self.video_label.setText("Loading...")
    
    def update_frame(self, image: QImage):
        """Update video frame."""
        if not image.isNull():
            pixmap = QPixmap.fromImage(image)
            scaled = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(scaled)
    
    def on_position_changed(self, position: float):
        """Handle position change."""
        self.current_time = position
        self.update_time_display()
        
        if self.duration > 0:
            slider_pos = int((position / self.duration) * 1000)
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(slider_pos)
            self.time_slider.blockSignals(False)
    
    def on_duration_ready(self, duration: float):
        """Handle duration ready."""
        self.duration = duration
        self.update_time_display()
        self.video_label.setText("")
    
    def toggle_play(self):
        """Toggle playback."""
        if not self.video_path:
            return
        
        if self.playing:
            if hasattr(self.parent(), 'video_player'):
                self.parent().video_player.pause()
            self.play_btn.setText("▶")
        else:
            if hasattr(self.parent(), 'video_player'):
                self.parent().video_player.play()
            self.play_btn.setText("⏸")
        
        self.playing = not self.playing
    
    def prev_frame(self):
        """Go to previous frame."""
        if hasattr(self.parent(), 'video_player'):
            self.parent().video_player.prev_frame()
    
    def next_frame(self):
        """Go to next frame."""
        if hasattr(self.parent(), 'video_player'):
            self.parent().video_player.next_frame()
    
    def slider_moved(self, value):
        """Handle slider movement."""
        if self.duration > 0:
            time_val = (value / 1000) * self.duration
            self.current_time = time_val
            self.update_time_display()
            self.position_changed.emit(time_val)
    
    def slider_pressed(self):
        """Handle slider press."""
        if self.playing:
            self.was_playing = True
            if hasattr(self.parent(), 'video_player'):
                self.parent().video_player.pause()
            self.play_btn.setText("▶")
            self.playing = False
    
    def slider_released(self):
        """Handle slider release."""
        if self.duration > 0:
            time_val = (self.time_slider.value() / 1000) * self.duration
            if hasattr(self.parent(), 'video_player'):
                self.parent().video_player.seek(time_val)
            
            if self.was_playing:
                if hasattr(self.parent(), 'video_player'):
                    self.parent().video_player.play()
                self.play_btn.setText("⏸")
                self.playing = True
            self.was_playing = False
    
    def update_time_display(self):
        """Update time label."""
        current = str(timedelta(seconds=int(self.current_time))).zfill(8)
        total = str(timedelta(seconds=int(self.duration))).zfill(8)
        self.time_label.setText(f"{current} / {total}")


# ==================== MAIN VIDEO EDITOR TAB ====================

class VideoEditorTab(QWidget):
    """Main video editor with CapCut-like features."""
    
    file_loaded = pyqtSignal(str)
    
    def __init__(self, parent=None, theme_manager=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.current_media = None
        self.video_player = None
        self.audio_player = None
        
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['bg_primary']};
                color: {COLORS['text_primary']};
            }}
            QSplitter::handle {{
                background-color: {COLORS['border']};
            }}
        """)
        
        self.setup_ui()
        self.setup_players()
    
    def setup_ui(self):
        """Setup the main UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Use stacked widget for state management
        self.stacked = QStackedWidget()
        main_layout.addWidget(self.stacked)
        
        # Empty state (initial)
        self.empty_state = EmptyStateWidget()
        self.empty_state.import_clicked.connect(self.import_video)
        self.stacked.addWidget(self.empty_state)
        
        # Editor state (full UI)
        self.editor_widget = self.create_editor_ui()
        self.stacked.addWidget(self.editor_widget)
        
        # Start with empty state
        self.stacked.setCurrentWidget(self.empty_state)
    
    def create_editor_ui(self) -> QWidget:
        """Create the full editor UI."""
        editor = QWidget()
        layout = QVBoxLayout(editor)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = self.create_header()
        layout.addWidget(header)
        
        # Main content splitter
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        
        # Left panel - Media Pool
        self.media_pool = MediaPoolWidget()
        self.media_pool.media_selected.connect(self.on_media_selected)
        self.media_pool.media_double_clicked.connect(self.add_to_timeline)
        self.media_pool.media_added.connect(self.on_media_added)
        self.main_splitter.addWidget(self.media_pool)
        
        # Center panel - Video Preview
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        
        self.preview = VideoPreviewWidget()
        self.preview.position_changed.connect(self.on_position_changed)
        center_layout.addWidget(self.preview, 1)
        
        self.main_splitter.addWidget(center_panel)
        
        # Right panel - Inspector
        self.inspector = InspectorWidget()
        self.inspector.clip_updated.connect(self.on_clip_updated)
        self.inspector.effect_applied.connect(self.on_effect_applied)
        self.main_splitter.addWidget(self.inspector)
        
        # Set splitter sizes
        self.main_splitter.setSizes([300, 700, 350])
        
        layout.addWidget(self.main_splitter, 1)
        
        # Bottom - Timeline
        self.timeline = TimelineWidget()
        self.timeline.position_changed.connect(self.on_timeline_position_changed)
        self.timeline.clip_selected.connect(self.on_clip_selected)
        self.timeline.clip_moved.connect(self.on_clip_moved)
        layout.addWidget(self.timeline)
        
        return editor
    
    def create_header(self) -> QWidget:
        """Create the header with project info and export button."""
        header = QWidget()
        header.setFixedHeight(50)
        header.setStyleSheet(f"""
            background-color: {COLORS['bg_secondary']};
            border-bottom: 1px solid {COLORS['border']};
        """)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(15, 0, 15, 0)
        
        # Project title
        self.project_title = QLabel("Project: Untitled")
        self.project_title.setStyleSheet(f"font-weight: bold; color: {COLORS['text_primary']};")
        layout.addWidget(self.project_title)
        
        layout.addStretch()
        
        # Export button
        self.export_btn = QPushButton("📤 Export Video")
        self.export_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_tertiary']};
                color: {COLORS['text_secondary']};
            }}
        """)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_video)
        layout.addWidget(self.export_btn)
        
        return header
    
    def setup_players(self):
        """Setup video and audio players."""
        self.video_player = VideoPlayerThread()
        self.video_player.frame_ready.connect(self.preview.update_frame)
        self.video_player.position_changed.connect(self.preview.on_position_changed)
        self.video_player.duration_ready.connect(self.preview.on_duration_ready)
        self.video_player.error.connect(self.on_player_error)
        self.video_player.start()
        
        self.audio_player = AudioPlayerThread()
        self.audio_player.position_changed.connect(self.on_audio_position_changed)
        self.audio_player.start()
    
    def import_video(self):
        """Import video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Video",
            "",
            f"Video Files (*{' *'.join(VIDEO_FORMATS)});;All Files (*.*)"
        )
        
        if file_path and os.path.exists(file_path):
            self.stacked.setCurrentWidget(self.editor_widget)
            media_item = self.media_pool.add_media(file_path)
            
            if media_item:
                self.current_media = media_item
                self.project_title.setText(f"Project: Loading...")
    
    def on_media_added(self, media_item):
        """Handle media added."""
        if media_item == self.current_media:
            self.preview.load_video(media_item.file_path)
            self.audio_player.load_audio(media_item.file_path)
            self.project_title.setText(f"Project: {media_item.name}")
            self.export_btn.setEnabled(True)
            self.timeline.set_media_items(self.media_pool.get_media_items())
    
    def on_media_selected(self, media_item):
        """Handle media selection."""
        if media_item.type == MediaType.VIDEO:
            self.preview.load_video(media_item.file_path)
            self.audio_player.load_audio(media_item.file_path)
    
    def add_to_timeline(self, media_item):
        """Add media to timeline."""
        clip = self.timeline.add_clip(media_item)
        if clip:
            self.timeline.set_media_items(self.media_pool.get_media_items())
    
    def on_position_changed(self, position: float):
        """Handle position change from preview."""
        self.timeline.set_position(position)
    
    def on_timeline_position_changed(self, position: float):
        """Handle position change from timeline."""
        if self.video_player and not self.preview.is_loading:
            self.video_player.seek(position)
            self.audio_player.seek(position)
    
    def on_audio_position_changed(self, position: float):
        """Handle audio position change."""
        if not self.preview.playing:
            self.timeline.set_position(position)
    
    def on_clip_selected(self, clip):
        """Handle clip selection."""
        media_item = None
        if clip and clip.media_id in self.media_pool.media_items:
            media_item = self.media_pool.media_items[clip.media_id]
        self.inspector.set_clip(clip, media_item)
    
    def on_clip_updated(self, clip):
        """Handle clip property update."""
        self.timeline.update()
    
    def on_effect_applied(self, effect):
        """Handle effect application."""
        if self.video_player:
            if 'bg_mode' in effect:
                mode = effect['bg_mode']
                if mode == 'none':
                    self.video_player.set_background_mode(BackgroundMode.NONE)
                elif mode == 'remove':
                    self.video_player.set_background_mode(BackgroundMode.REMOVE)
                elif mode == 'green_screen':
                    color = effect.get('color', (0, 255, 0))
                    self.video_player.set_background_mode(BackgroundMode.GREEN_SCREEN, color)
                elif mode == 'replace':
                    image = effect.get('image')
                    self.video_player.set_background_mode(BackgroundMode.REPLACE, image_path=image)
            elif self.inspector.current_clip:
                # Apply to current clip
                self.inspector.current_clip.effects.append(effect)
    
    def on_player_error(self, error_msg):
        """Handle player error."""
        QMessageBox.critical(self, "Player Error", error_msg)
    
    def on_clip_moved(self, clip, new_start):
        """Handle clip movement."""
        pass
    
    def export_video(self):
        """Export video."""
        if not self.current_media:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Video",
            "",
            "MP4 Video (*.mp4);;MOV Video (*.mov);;AVI Video (*.avi)"
        )
        
        if file_path:
            QMessageBox.information(
                self,
                "Export",
                f"Video would be exported to:\n{file_path}\n\nThis is a placeholder. Implement actual export functionality."
            )
    
    def apply_theme(self, is_dark):
        """Apply theme."""
        pass
    
    def closeEvent(self, event):
        """Clean up on close."""
        if self.video_player:
            self.video_player.stop_thread()
        if self.audio_player:
            self.audio_player.stop_thread()
        super().closeEvent(event)


# ==================== EXPORTS ====================

__all__ = [
    'VideoEditorTab',
    'MediaPoolWidget',
    'TimelineWidget',
    'InspectorWidget',
    'VideoPreviewWidget',
    'MediaItem',
    'TimelineClip',
    'FFMPEG_AVAILABLE',
    'PYDUB_AVAILABLE',
    'PYGAME_AVAILABLE',
    'REMBG_AVAILABLE'
]

def create_video_editor_tab(parent, theme_manager=None):
    """Create video editor tab."""
    return VideoEditorTab(parent, theme_manager)