"""
AI Studio Pro - Download Module
- Support for YouTube, Facebook, Twitter/X, Instagram, TikTok, and more
- Video/audio downloading with format selection
- Progress tracking with callbacks
- Automatic audio conversion for transcription
- Error handling and validation
- Integration with main.py for unified theme and UI
"""

import os
import sys
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, Dict, List, Tuple, Any

# PyQt6 imports
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QComboBox, QProgressBar, QMessageBox, 
    QDialog, QCheckBox, QGroupBox, QTextEdit, QFileDialog, 
    QRadioButton, QButtonGroup, QListWidget, QListWidgetItem,
    QGridLayout, QTabWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

# Try to import yt-dlp
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

# Try to import for audio conversion
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


# ==================== SUPPORTED PLATFORMS ====================

SUPPORTED_PLATFORMS = {
    'youtube': {
        'domains': ['youtube.com', 'youtu.be', 'm.youtube.com', 'www.youtube.com'],
        'icon': '📺',
        'name': 'YouTube',
        'color': '#FF0000'
    },
    'facebook': {
        'domains': ['facebook.com', 'fb.com', 'fb.watch', 'www.facebook.com'],
        'icon': '👤',
        'name': 'Facebook',
        'color': '#1877F2'
    },
    'twitter': {
        'domains': ['twitter.com', 'x.com', 'www.twitter.com', 'www.x.com'],
        'icon': '🐦',
        'name': 'Twitter/X',
        'color': '#1DA1F2'
    },
    'instagram': {
        'domains': ['instagram.com', 'www.instagram.com'],
        'icon': '📷',
        'name': 'Instagram',
        'color': '#E4405F'
    },
    'tiktok': {
        'domains': ['tiktok.com', 'www.tiktok.com'],
        'icon': '🎵',
        'name': 'TikTok',
        'color': '#000000'
    },
    'vimeo': {
        'domains': ['vimeo.com', 'www.vimeo.com'],
        'icon': '🎥',
        'name': 'Vimeo',
        'color': '#1AB7EA'
    },
    'dailymotion': {
        'domains': ['dailymotion.com', 'www.dailymotion.com'],
        'icon': '🎬',
        'name': 'Dailymotion',
        'color': '#0066DC'
    },
    'twitch': {
        'domains': ['twitch.tv', 'www.twitch.tv'],
        'icon': '🎮',
        'name': 'Twitch',
        'color': '#9146FF'
    },
    'soundcloud': {
        'domains': ['soundcloud.com', 'www.soundcloud.com'],
        'icon': '🎵',
        'name': 'SoundCloud',
        'color': '#FF7700'
    },
    'generic': {
        'domains': [],
        'icon': '🌐',
        'name': 'Other',
        'color': '#4CAF50'
    }
}


# ==================== UTILITY FUNCTIONS ====================

def detect_platform(url: str) -> Tuple[str, str, str, str]:
    """Detect which platform the URL belongs to."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    for platform, info in SUPPORTED_PLATFORMS.items():
        for platform_domain in info['domains']:
            if platform_domain in domain:
                return platform, info['name'], info['icon'], info['color']
    
    return 'generic', 'Other', '🌐', '#4CAF50'


def is_valid_url(url: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def format_duration(seconds: int) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        return f"{hours}h {minutes}m {remaining_seconds}s"


# ==================== DOWNLOAD THREAD ====================

class DownloadThread(QThread):
    """Background thread for downloading with yt-dlp."""
    
    progress = pyqtSignal(int, int, int)
    speed = pyqtSignal(str)
    status = pyqtSignal(str)
    info_ready = pyqtSignal(dict)
    formats_ready = pyqtSignal(list)
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)
    
    def __init__(self, url: str, output_dir: str, options: Dict[str, Any]):
        super().__init__()
        self.url = url
        self.output_dir = output_dir
        self.options = options
        self._running = True
        self.ydl = None
    
    def stop(self):
        self._running = False
        if self.ydl:
            self.ydl.params['quiet'] = True
    
    def progress_hook(self, d):
        if not self._running:
            raise Exception("Download cancelled by user")
        
        if d['status'] == 'downloading':
            if 'total_bytes' in d and d['total_bytes'] > 0:
                percent = int(d['downloaded_bytes'] * 100 / d['total_bytes'])
                self.progress.emit(percent, d['downloaded_bytes'], d['total_bytes'])
            elif 'total_bytes_estimate' in d and d['total_bytes_estimate'] > 0:
                percent = int(d['downloaded_bytes'] * 100 / d['total_bytes_estimate'])
                self.progress.emit(percent, d['downloaded_bytes'], d['total_bytes_estimate'])
            
            if 'speed' in d and d['speed']:
                speed_str = format_file_size(d['speed']) + '/s'
                self.speed.emit(speed_str)
            
            if '_percent_str' in d:
                self.status.emit(f"Downloading: {d['_percent_str']}")
            else:
                self.status.emit("Downloading...")
                
        elif d['status'] == 'finished':
            self.status.emit("Download finished, processing...")
        
        elif d['status'] == 'error':
            self.error.emit("Download error occurred")
    
    def get_available_formats(self):
        """Get available formats for the video."""
        try:
            with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
                info = ydl.extract_info(self.url, download=False)
                if 'formats' in info:
                    return info['formats']
                return []
        except:
            return []
    
    def run(self):
        if not YT_DLP_AVAILABLE:
            self.error.emit("yt-dlp is not installed. Please install with: pip install yt-dlp")
            return
        
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Base options
            ydl_opts = {
                'progress_hooks': [self.progress_hook],
                'outtmpl': os.path.join(self.output_dir, '%(title)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'ignoreerrors': True,
            }
            
            # If we're just fetching formats
            if self.options.get('get_formats_only', False):
                formats = self.get_available_formats()
                self.formats_ready.emit(formats)
                return
            
            # Configure format selection based on options
            if self.options.get('format') == 'audio':
                ydl_opts.update({
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': self.options.get('audio_format', 'mp3'),
                        'preferredquality': self.options.get('audio_quality', '192'),
                    }],
                })
            elif self.options.get('format') == 'video':
                quality = self.options.get('video_quality', 'best')
                
                # For clean video without overlays, prefer progressive streams or best video+audio
                if quality == 'best':
                    # Try to get best quality without re-encoding
                    ydl_opts['format'] = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
                elif quality == '1080p':
                    ydl_opts['format'] = 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best[height<=1080]'
                elif quality == '720p':
                    ydl_opts['format'] = 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]'
                elif quality == '480p':
                    ydl_opts['format'] = 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best[height<=480]'
                else:
                    ydl_opts['format'] = f'best[height<={quality}]'
            
            # Add subtitle options
            if self.options.get('subtitles', False):
                ydl_opts.update({
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': ['en', self.options.get('subtitle_lang', 'en')],
                    'embedsubs': False,  # Don't embed subtitles to keep video clean
                })
            
            # Add thumbnail option
            if self.options.get('thumbnail', False):
                ydl_opts['writethumbnail'] = True
            
            # Add cookies if available (helps with age-restricted content)
            cookies_file = os.path.join(os.path.dirname(__file__), 'cookies.txt')
            if os.path.exists(cookies_file):
                ydl_opts['cookiefile'] = cookies_file
            
            # For platforms that might have watermarks, try to get original format
            platform_key, _, _, _ = detect_platform(self.url)
            if platform_key in ['tiktok', 'instagram', 'facebook']:
                # These platforms often have watermarks in certain formats
                # Try to get the original uploaded format
                if 'format' not in ydl_opts or ydl_opts['format'] == 'best':
                    ydl_opts['format'] = 'best[protocol^=http][ext=mp4]/best'
            
            if self.options.get('get_info_only', False):
                with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                    info = ydl.extract_info(self.url, download=False)
                    self.info_ready.emit(info)
                    return
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                self.ydl = ydl
                info = ydl.extract_info(self.url, download=True)
                
                if info and self._running:
                    if 'entries' in info:
                        first_entry = info['entries'][0]
                        title = first_entry.get('title', 'video')
                        ext = first_entry.get('ext', 'mp4')
                    else:
                        title = info.get('title', 'video')
                        ext = info.get('ext', 'mp4')
                    
                    if self.options.get('format') == 'audio':
                        ext = self.options.get('audio_format', 'mp3')
                    
                    output_path = os.path.join(self.output_dir, f"{title}.{ext}")
                    
                    if not os.path.exists(output_path):
                        files = os.listdir(self.output_dir)
                        for file in files:
                            if title in file:
                                output_path = os.path.join(self.output_dir, file)
                                break
                    
                    self.finished.emit(output_path, title)
                
        except Exception as e:
            if self._running:
                self.error.emit(str(e))


# ==================== INFO FETCH THREAD ====================

class InfoFetchThread(QThread):
    """Background thread for fetching video information."""
    
    info_ready = pyqtSignal(dict)
    formats_ready = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self._running = True
    
    def stop(self):
        self._running = False
    
    def run(self):
        if not YT_DLP_AVAILABLE:
            self.error.emit("yt-dlp is not installed")
            return
        
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'ignoreerrors': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=False)
                
                if self._running:
                    self.info_ready.emit(info)
                    if 'formats' in info:
                        self.formats_ready.emit(info['formats'])
                
        except Exception as e:
            if self._running:
                self.error.emit(str(e))


# ==================== FORMAT SELECTOR DIALOG ====================

class FormatSelectorDialog(QDialog):
    """Dialog for selecting specific video/audio format."""
    
    def __init__(self, formats, parent=None, theme_manager=None):
        super().__init__(parent)
        self.formats = formats
        self.theme_manager = theme_manager
        self.selected_format = None
        
        self.setWindowTitle("Select Format")
        self.setMinimumSize(600, 400)
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Format list
        self.format_list = QListWidget()
        self.format_list.setAlternatingRowColors(True)
        
        # Populate formats
        for fmt in self.formats:
            format_id = fmt.get('format_id', 'N/A')
            ext = fmt.get('ext', 'N/A')
            resolution = fmt.get('resolution', 'N/A')
            filesize = fmt.get('filesize', 0) or fmt.get('filesize_approx', 0)
            vcodec = fmt.get('vcodec', 'none')
            acodec = fmt.get('acodec', 'none')
            fps = fmt.get('fps', '')
            
            # Determine if this is a clean format (likely without watermarks)
            is_clean = (
                vcodec != 'none' and 
                acodec != 'none' and
                'source' in fmt.get('format_note', '').lower() or
                'original' in fmt.get('format_note', '').lower()
            )
            
            size_str = format_file_size(filesize) if filesize else 'Unknown size'
            
            display_text = f"[{format_id}] {ext} - {resolution}"
            if fps:
                display_text += f" {fps}fps"
            display_text += f" - {size_str}"
            
            if vcodec != 'none':
                display_text += f" - Video: {vcodec[:20]}"
            if acodec != 'none':
                display_text += f" - Audio: {acodec[:20]}"
            
            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, fmt)
            
            # Highlight clean formats
            if is_clean:
                item.setBackground(Qt.GlobalColor.lightGray)
                item.setToolTip("This format might be free of watermarks/overlays")
            
            self.format_list.addItem(item)
        
        layout.addWidget(QLabel("Available formats (highlighted formats may be cleaner):"))
        layout.addWidget(self.format_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        select_btn = QPushButton("Select Format")
        select_btn.clicked.connect(self.select_format)
        button_layout.addWidget(select_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def select_format(self):
        current_item = self.format_list.currentItem()
        if current_item:
            self.selected_format = current_item.data(Qt.ItemDataRole.UserRole)
            self.accept()
        else:
            QMessageBox.warning(self, "No Selection", "Please select a format")


# ==================== DOWNLOAD DIALOG ====================

class DownloadDialog(QDialog):
    """Dialog for downloading media from URLs."""
    
    def __init__(self, parent=None, theme_manager=None):
        super().__init__(parent)
        self.parent = parent
        self.theme_manager = theme_manager
        self.download_thread = None
        self.info_thread = None
        self.current_info = None
        self.downloaded_file = None
        self.available_formats = []
        
        self.setWindowTitle("Download Media")
        self.setMinimumSize(800, 700)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint)
        
        # Apply theme if available
        if theme_manager:
            self.setStyleSheet(theme_manager.get_nav_stylesheet())
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the download dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # ========== URL SECTION ==========
        url_group = QGroupBox("Media URL")
        url_layout = QVBoxLayout()
        
        # URL input with fetch button
        url_input_layout = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter YouTube, Facebook, or other media URL...")
        self.url_input.textChanged.connect(self.on_url_changed)
        url_input_layout.addWidget(self.url_input)
        
        self.fetch_btn = QPushButton("Fetch Info")
        self.fetch_btn.clicked.connect(self.fetch_info)
        self.fetch_btn.setEnabled(False)
        self.fetch_btn.setFixedWidth(100)
        url_input_layout.addWidget(self.fetch_btn)
        
        url_layout.addLayout(url_input_layout)
        
        # Platform indicator
        self.platform_label = QLabel("Enter a URL to detect platform")
        self.platform_label.setStyleSheet("color: gray; font-style: italic;")
        url_layout.addWidget(self.platform_label)
        
        url_group.setLayout(url_layout)
        layout.addWidget(url_group)
        
        # ========== INFO SECTION ==========
        info_group = QGroupBox("Media Information")
        info_layout = QVBoxLayout()
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        self.info_text.setFont(QFont("Consolas", 10))
        info_layout.addWidget(self.info_text)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # ========== TABS FOR OPTIONS ==========
        self.tab_widget = QTabWidget()
        
        # Tab 1: Simple Options
        simple_tab = QWidget()
        simple_layout = QGridLayout(simple_tab)
        simple_layout.setVerticalSpacing(10)
        simple_layout.setHorizontalSpacing(15)
        
        # Format selection
        format_label = QLabel("Format:")
        simple_layout.addWidget(format_label, 0, 0, Qt.AlignmentFlag.AlignRight)
        
        format_widget = QWidget()
        format_layout = QHBoxLayout(format_widget)
        format_layout.setContentsMargins(0, 0, 0, 0)
        format_layout.setSpacing(15)
        
        self.format_group = QButtonGroup()
        
        self.video_radio = QRadioButton("Video")
        self.video_radio.setChecked(True)
        self.format_group.addButton(self.video_radio)
        format_layout.addWidget(self.video_radio)
        
        self.audio_radio = QRadioButton("Audio Only")
        self.format_group.addButton(self.audio_radio)
        format_layout.addWidget(self.audio_radio)
        
        format_layout.addStretch()
        simple_layout.addWidget(format_widget, 0, 1)
        
        # Video quality
        video_quality_label = QLabel("Video Quality:")
        simple_layout.addWidget(video_quality_label, 1, 0, Qt.AlignmentFlag.AlignRight)
        
        self.video_quality = QComboBox()
        self.video_quality.addItems(["Best", "1080p", "720p", "480p", "360p"])
        self.video_quality.setCurrentText("Best")
        simple_layout.addWidget(self.video_quality, 1, 1)
        
        # Clean video option
        self.clean_video_check = QCheckBox("Prefer clean video (without watermarks/logos)")
        self.clean_video_check.setChecked(True)
        self.clean_video_check.setToolTip("Try to select formats that might be free of watermarks and overlays")
        simple_layout.addWidget(self.clean_video_check, 2, 1)
        
        # Audio format
        audio_format_label = QLabel("Audio Format:")
        simple_layout.addWidget(audio_format_label, 3, 0, Qt.AlignmentFlag.AlignRight)
        
        format_quality_widget = QWidget()
        format_quality_layout = QHBoxLayout(format_quality_widget)
        format_quality_layout.setContentsMargins(0, 0, 0, 0)
        format_quality_layout.setSpacing(10)
        self.audio_format = QComboBox()
        self.audio_format.addItems(["mp3", "m4a", "wav", "flac", "opus"])
        self.audio_format.setCurrentText("mp3")
        format_quality_layout.addWidget(self.audio_format)
        
        self.audio_quality = QComboBox()
        self.audio_quality.addItems(["128 kbps", "192 kbps", "256 kbps", "320 kbps"])
        self.audio_quality.setCurrentText("192 kbps")
        format_quality_layout.addWidget(self.audio_quality)
        
        format_quality_layout.addStretch()
        simple_layout.addWidget(format_quality_widget, 3, 1)
        
        # Additional options
        extra_label = QLabel("Extras:")
        simple_layout.addWidget(extra_label, 4, 0, Qt.AlignmentFlag.AlignRight)
        
        extra_widget = QWidget()
        extra_layout = QHBoxLayout(extra_widget)
        extra_layout.setContentsMargins(0, 0, 0, 0)
        extra_layout.setSpacing(15)
        
        self.subs_check = QCheckBox("Subtitles (external)")
        extra_layout.addWidget(self.subs_check)
        
        self.thumb_check = QCheckBox("Thumbnail")
        extra_layout.addWidget(self.thumb_check)
        
        extra_layout.addStretch()
        simple_layout.addWidget(extra_widget, 4, 1)
        
        # Tab 2: Advanced Options
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        
        self.advanced_info = QLabel("Advanced format selection:")
        advanced_layout.addWidget(self.advanced_info)
        
        self.show_formats_btn = QPushButton("Show Available Formats")
        self.show_formats_btn.clicked.connect(self.show_available_formats)
        self.show_formats_btn.setEnabled(False)
        advanced_layout.addWidget(self.show_formats_btn)
        
        self.selected_format_label = QLabel("No format selected")
        self.selected_format_label.setStyleSheet("color: gray; font-style: italic;")
        advanced_layout.addWidget(self.selected_format_label)
        
        advanced_layout.addStretch()
        
        self.tab_widget.addTab(simple_tab, "Simple")
        self.tab_widget.addTab(advanced_tab, "Advanced")
        
        layout.addWidget(self.tab_widget)
        
        # Connect radio buttons
        self.video_radio.toggled.connect(self.update_options_ui)
        self.update_options_ui()
        
        # ========== OUTPUT SECTION ==========
        output_group = QGroupBox("Output Location")
        output_layout = QHBoxLayout()
        
        self.output_path = QLineEdit()
        self.output_path.setText(os.path.expanduser("~/Downloads"))
        output_layout.addWidget(self.output_path)
        
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_output)
        self.browse_btn.setFixedWidth(80)
        output_layout.addWidget(self.browse_btn)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # ========== PROGRESS SECTION ==========
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.speed_label = QLabel("")
        self.speed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.speed_label)
        
        self.status_label = QLabel("Ready to download")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # ========== BUTTONS ==========
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.download_btn = QPushButton("Download")
        self.download_btn.clicked.connect(self.start_download)
        self.download_btn.setEnabled(False)
        self.download_btn.setMinimumHeight(35)
        self.download_btn.setFixedWidth(100)
        button_layout.addWidget(self.download_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_download)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setMinimumHeight(35)
        self.cancel_btn.setFixedWidth(100)
        button_layout.addWidget(self.cancel_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        self.close_btn.setMinimumHeight(35)
        self.close_btn.setFixedWidth(100)
        button_layout.addWidget(self.close_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
    
    def update_options_ui(self):
        """Update UI based on selected format."""
        is_video = self.video_radio.isChecked()
        self.video_quality.setEnabled(is_video)
        self.clean_video_check.setEnabled(is_video)
        self.audio_format.setEnabled(not is_video)
        self.audio_quality.setEnabled(not is_video)
    
    def on_url_changed(self, text):
        """Handle URL input changes."""
        self.fetch_btn.setEnabled(len(text.strip()) > 0 and is_valid_url(text))
        
        if is_valid_url(text):
            platform_key, platform_name, icon, color = detect_platform(text)
            self.platform_label.setText(f"{icon} Detected: {platform_name}")
            self.platform_label.setStyleSheet(f"color: {color}; font-style: normal;")
            
            if self.current_info:
                self.download_btn.setEnabled(True)
                self.show_formats_btn.setEnabled(True)
        else:
            self.platform_label.setText("Enter a valid URL to detect platform")
            self.platform_label.setStyleSheet("color: gray; font-style: italic;")
    
    def browse_output(self):
        """Browse for output directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_path.text(),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if directory:
            self.output_path.setText(directory)
    
    def fetch_info(self):
        """Fetch media information."""
        url = self.url_input.text().strip()
        
        if not url:
            QMessageBox.warning(self, "No URL", "Please enter a URL")
            return
        
        if not is_valid_url(url):
            QMessageBox.warning(self, "Invalid URL", "Please enter a valid URL")
            return
        
        self.info_text.clear()
        self.info_text.append("Fetching media information...")
        self.fetch_btn.setEnabled(False)
        self.download_btn.setEnabled(False)
        self.show_formats_btn.setEnabled(False)
        
        self.info_thread = InfoFetchThread(url)
        self.info_thread.info_ready.connect(self.on_info_ready)
        self.info_thread.formats_ready.connect(self.on_formats_ready)
        self.info_thread.error.connect(self.on_info_error)
        self.info_thread.start()
    
    def on_info_ready(self, info):
        """Handle fetched media information."""
        self.current_info = info
        self.fetch_btn.setEnabled(True)
        self.download_btn.setEnabled(True)
        self.show_formats_btn.setEnabled(True)
        
        self.info_text.clear()
        
        if 'entries' in info:
            # Playlist
            self.info_text.append(f"PLAYLIST: {info.get('title', 'Untitled')}")
            self.info_text.append(f"Videos: {len(info['entries'])}")
            self.info_text.append("")
            
            for i, entry in enumerate(info['entries'][:5], 1):
                title = entry.get('title', 'Untitled')
                duration = entry.get('duration', 0)
                duration_str = format_duration(duration)
                self.info_text.append(f"{i}. {title} ({duration_str})")
            
            if len(info['entries']) > 5:
                self.info_text.append(f"... and {len(info['entries']) - 5} more")
        else:
            # Single video
            title = info.get('title', 'Untitled')
            self.info_text.append(f"Title: {title}")
            
            uploader = info.get('uploader', 'Unknown')
            self.info_text.append(f"Uploader: {uploader}")
            
            duration = info.get('duration', 0)
            duration_str = format_duration(duration)
            self.info_text.append(f"Duration: {duration_str}")
            
            view_count = info.get('view_count', 0)
            if view_count:
                self.info_text.append(f"Views: {view_count:,}")
            
            upload_date = info.get('upload_date', '')
            if upload_date:
                year = upload_date[:4]
                month = upload_date[4:6]
                day = upload_date[6:8]
                self.info_text.append(f"Uploaded: {year}-{month}-{day}")
    
    def on_formats_ready(self, formats):
        """Handle available formats."""
        self.available_formats = formats
    
    def on_info_error(self, error_msg):
        """Handle info fetch error."""
        self.info_text.clear()
        self.info_text.append(f"Error: {error_msg}")
        self.fetch_btn.setEnabled(True)
        self.download_btn.setEnabled(False)
        self.show_formats_btn.setEnabled(False)
        
        QMessageBox.critical(self, "Fetch Error", f"Failed to fetch information:\n{error_msg}")
    
    def show_available_formats(self):
        """Show dialog with available formats."""
        if not self.available_formats:
            QMessageBox.information(self, "No Formats", "No format information available")
            return
        
        dialog = FormatSelectorDialog(self.available_formats, self, self.theme_manager)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_format:
            fmt = dialog.selected_format
            format_id = fmt.get('format_id', 'unknown')
            resolution = fmt.get('resolution', 'unknown')
            self.selected_format_label.setText(f"Selected: {format_id} - {resolution}")
            self.selected_format_label.setStyleSheet("color: green; font-style: normal;")
    
    def start_download(self):
        """Start the download process."""
        url = self.url_input.text().strip()
        output_dir = self.output_path.text()
        
        if not output_dir:
            QMessageBox.warning(self, "No Output", "Please select an output directory")
            return
        
        options = {
            'format': 'audio' if self.audio_radio.isChecked() else 'video',
            'get_info_only': False,
        }
        
        if options['format'] == 'video':
            quality = self.video_quality.currentText().lower()
            if quality != 'best':
                options['video_quality'] = quality.replace('p', '')
            
            # Add clean video preference
            if self.clean_video_check.isChecked():
                # This will be handled in the download thread
                options['prefer_clean'] = True
        else:
            options['audio_format'] = self.audio_format.currentText()
            quality = self.audio_quality.currentText().split()[0]
            options['audio_quality'] = quality
        
        options['subtitles'] = self.subs_check.isChecked()
        options['thumbnail'] = self.thumb_check.isChecked()
        
        self.download_btn.setEnabled(False)
        self.fetch_btn.setEnabled(False)
        self.show_formats_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting download...")
        
        self.download_thread = DownloadThread(url, output_dir, options)
        self.download_thread.progress.connect(self.update_progress)
        self.download_thread.speed.connect(self.speed_label.setText)
        self.download_thread.status.connect(self.status_label.setText)
        self.download_thread.finished.connect(self.on_download_finished)
        self.download_thread.error.connect(self.on_download_error)
        self.download_thread.start()
    
    def update_progress(self, percent, downloaded, total):
        """Update progress bar."""
        self.progress_bar.setValue(percent)
        
        downloaded_str = format_file_size(downloaded)
        total_str = format_file_size(total)
        self.status_label.setText(f"Downloaded: {downloaded_str} / {total_str} ({percent}%)")
    
    def on_download_finished(self, output_path, title):
        """Handle successful download."""
        self.downloaded_file = output_path
        self.progress_bar.setValue(100)
        self.status_label.setText(f"Download complete: {title}")
        self.speed_label.setText("")
        self.cancel_btn.setEnabled(False)
        self.download_btn.setEnabled(True)
        self.fetch_btn.setEnabled(True)
        self.show_formats_btn.setEnabled(True)
        
        reply = QMessageBox.question(
            self,
            "Download Complete",
            f"File saved to:\n{output_path}\n\nOpen containing folder?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.open_folder(os.path.dirname(output_path))
    
    def on_download_error(self, error_msg):
        """Handle download error."""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Download failed")
        self.speed_label.setText("")
        self.cancel_btn.setEnabled(False)
        self.download_btn.setEnabled(True)
        self.fetch_btn.setEnabled(True)
        self.show_formats_btn.setEnabled(True)
        
        QMessageBox.critical(self, "Download Error", f"Failed to download:\n{error_msg}")
    
    def cancel_download(self):
        """Cancel the current download."""
        if self.download_thread and self.download_thread.isRunning():
            self.download_thread.stop()
            self.download_thread.wait(2000)
            self.status_label.setText("Download cancelled")
            self.speed_label.setText("")
            self.cancel_btn.setEnabled(False)
            self.download_btn.setEnabled(True)
            self.fetch_btn.setEnabled(True)
            self.show_formats_btn.setEnabled(True)
    
    def open_folder(self, folder_path):
        """Open folder in file explorer."""
        try:
            if sys.platform == 'win32':
                os.startfile(folder_path)
            elif sys.platform == 'darwin':
                subprocess.run(['open', folder_path])
            else:
                subprocess.run(['xdg-open', folder_path])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open folder:\n{str(e)}")
    
    def get_downloaded_file(self) -> Optional[str]:
        """Get the path of the downloaded file."""
        return self.downloaded_file


# ==================== DOWNLOAD MANAGER WIDGET ====================

class DownloadManagerWidget(QWidget):
    """Widget for managing downloads within the main application."""
    
    download_complete = pyqtSignal(str)
    
    def __init__(self, parent=None, theme_manager=None):
        super().__init__(parent)
        self.parent = parent
        self.theme_manager = theme_manager
        self.downloads = []
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the download manager UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header_layout = QHBoxLayout()
        title = QLabel("Download Manager")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        self.new_download_btn = QPushButton("New Download")
        self.new_download_btn.clicked.connect(self.open_download_dialog)
        header_layout.addWidget(self.new_download_btn)
        
        layout.addLayout(header_layout)
        
        # Download list
        self.download_list = QListWidget()
        self.download_list.setAlternatingRowColors(True)
        layout.addWidget(self.download_list)
        
        # Quick URL input
        quick_layout = QHBoxLayout()
        self.quick_url = QLineEdit()
        self.quick_url.setPlaceholderText("Paste URL and press Enter")
        self.quick_url.returnPressed.connect(self.quick_download)
        quick_layout.addWidget(self.quick_url)
        
        self.quick_download_btn = QPushButton("Quick Download")
        self.quick_download_btn.clicked.connect(self.quick_download)
        quick_layout.addWidget(self.quick_download_btn)
        
        layout.addLayout(quick_layout)
    
    def open_download_dialog(self):
        """Open the download dialog."""
        dialog = DownloadDialog(self, self.theme_manager)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            file_path = dialog.get_downloaded_file()
            if file_path:
                self.add_download_to_list(file_path)
                self.download_complete.emit(file_path)
    
    def quick_download(self):
        """Quick download from URL input."""
        url = self.quick_url.text().strip()
        
        if not url:
            QMessageBox.warning(self, "No URL", "Please enter a URL")
            return
        
        if not is_valid_url(url):
            QMessageBox.warning(self, "Invalid URL", "Please enter a valid URL")
            return
        
        dialog = DownloadDialog(self, self.theme_manager)
        dialog.url_input.setText(url)
        dialog.on_url_changed(url)
        
        QTimer.singleShot(500, dialog.fetch_info)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            file_path = dialog.get_downloaded_file()
            if file_path:
                self.add_download_to_list(file_path)
                self.download_complete.emit(file_path)
                self.quick_url.clear()
    
    def add_download_to_list(self, file_path):
        """Add a downloaded file to the list."""
        if file_path not in self.downloads:
            self.downloads.append(file_path)
            
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            size_str = format_file_size(file_size)
            
            item = QListWidgetItem(f"{file_name} ({size_str})")
            item.setData(Qt.ItemDataRole.UserRole, file_path)
            
            self.download_list.addItem(item)
    
    def get_latest_download(self) -> Optional[str]:
        """Get the path of the most recent download."""
        return self.downloads[-1] if self.downloads else None


# ==================== INTEGRATION FUNCTIONS ====================

def add_download_tab_to_app(parent_widget, theme_manager=None):
    """Add a download manager tab to an existing application."""
    return DownloadManagerWidget(parent_widget, theme_manager)


def create_download_button(parent, theme_manager=None):
    """Create a download button that opens the download dialog."""
    button = QPushButton("Download Media")
    
    def open_download():
        dialog = DownloadDialog(parent, theme_manager)
        dialog.exec()
    
    button.clicked.connect(open_download)
    return button


# ==================== EXPORTS ====================

__all__ = [
    'DownloadDialog',
    'DownloadManagerWidget',
    'DownloadThread',
    'InfoFetchThread',
    'is_valid_url',
    'detect_platform',
    'format_file_size',
    'format_duration',
    'add_download_tab_to_app',
    'create_download_button',
    'YT_DLP_AVAILABLE',
    'PYDUB_AVAILABLE',
    'SUPPORTED_PLATFORMS'
]