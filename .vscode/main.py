"""
AI Studio Pro - Unified Application (Fully Scrollable Edition)
- Speech-to-Text, TTS Merger, Download Manager, Overlap Detector
- Fully scrollable interface for any screen size
- Lazy loading for instant startup (<5 seconds)
- Modern UI with smooth transitions
"""

import sys
import os
import importlib.util
import traceback
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QStatusBar, QScrollArea, QFrame,
    QMessageBox, QDialog, QProgressBar
)
from PyQt6.QtCore import (
    Qt, QTimer, QSize, pyqtSignal, QPropertyAnimation,
    QEasingCurve
)
from PyQt6.QtGui import (
    QFont, QPalette, QColor, QFontDatabase,
    QLinearGradient, QBrush, QAction
)


# ==================== MODULE IMPORTS (LAZY) ====================

# We'll import modules only when needed
SETTINGS_MODULE_AVAILABLE = False
DOWNLOAD_MODULE_AVAILABLE = False
OVERLAP_MODULE_AVAILABLE = False
SPEECH_MODULE_AVAILABLE = False
TTS_MODULE_AVAILABLE = False

# Module constructors (will be set on first use)
SpeechToTextWindow = None
TTSMergerWindow = None
DownloadManagerWidget = None
OverlapDetectorTab = None
SettingsDialog = None
SettingsManager = None
AppSettings = None

def lazy_import(module_name: str) -> bool:
    """Lazily import a module when needed."""
    global SETTINGS_MODULE_AVAILABLE, DOWNLOAD_MODULE_AVAILABLE
    global OVERLAP_MODULE_AVAILABLE, SPEECH_MODULE_AVAILABLE
    global TTS_MODULE_AVAILABLE
    global SpeechToTextWindow, TTSMergerWindow
    global DownloadManagerWidget, OverlapDetectorTab
    global SettingsDialog, SettingsManager, AppSettings
    
    try:
        if module_name == 'settings':
            from settings import (
                SettingsManager as SM, AppSettings as AS,
                SettingsDialog as SD
            )
            SettingsManager = SM
            AppSettings = AS
            SettingsDialog = SD
            SETTINGS_MODULE_AVAILABLE = True
            print("✅ Settings module loaded lazily")
            return True
            
        elif module_name == 'download':
            from download import DownloadManagerWidget as DMW
            DownloadManagerWidget = DMW
            DOWNLOAD_MODULE_AVAILABLE = True
            print("✅ Download module loaded lazily")
            return True
            
        elif module_name == 'audio_detect_overlap':
            from audio_detect_overlap import OverlapDetectorTab as ODT
            OverlapDetectorTab = ODT
            OVERLAP_MODULE_AVAILABLE = True
            print("✅ Overlap module loaded lazily")
            return True
            
        elif module_name == 'speech_to_text':
            from speech_to_text import MainWindow as STT
            SpeechToTextWindow = STT
            SPEECH_MODULE_AVAILABLE = True
            print("✅ Speech module loaded lazily")
            return True
            
        elif module_name == 'text_to_speech':
            from text_to_speech import KhmerTTSStudio as TTS
            TTSMergerWindow = TTS
            TTS_MODULE_AVAILABLE = True
            print("✅ TTS module loaded lazily")
            return True
            
    except ImportError as e:
        print(f"⚠️ Module {module_name} not available: {e}")
        return False
    
    return False


# ==================== SCRIPT DIRECTORY ====================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"📁 Script directory: {SCRIPT_DIR}")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


# ==================== UNIFIED THEME MANAGER ====================

class UnifiedThemeManager:
    """Manages dark/light themes for all apps with modern styling."""
    
    DARK = {
        "bg_primary": "#1e1e1e",
        "bg_secondary": "#252526",
        "bg_tertiary": "#2d2d30",
        "bg_hover": "#3a3a3a",
        "bg_pressed": "#4a4a4a",
        "text_primary": "#ffffff",
        "text_secondary": "#cccccc",
        "text_disabled": "#666666",
        "accent": "#4CAF50",
        "accent_hover": "#5CBF60",
        "accent_gradient_start": "#4CAF50",
        "accent_gradient_end": "#45a049",
        "border": "#3c3c3c",
        "border_light": "#4a4a4a",
        "nav_bg": "#252526",
        "nav_text": "#cccccc",
        "nav_hover": "#3a3a3a",
        "nav_active": "#4CAF50",
        "tool_button": "#3a3a3a",
        "tool_button_text": "#ffffff",
        "status_bg": "#007acc",
        "status_text": "#ffffff",
        "shadow": "rgba(0,0,0,0.3)"
    }
    
    LIGHT = {
        "bg_primary": "#f8f9fa",
        "bg_secondary": "#ffffff",
        "bg_tertiary": "#e9ecef",
        "bg_hover": "#dee2e6",
        "bg_pressed": "#ced4da",
        "text_primary": "#212529",
        "text_secondary": "#495057",
        "text_disabled": "#adb5bd",
        "accent": "#4CAF50",
        "accent_hover": "#5CBF60",
        "accent_gradient_start": "#4CAF50",
        "accent_gradient_end": "#3d8b40",
        "border": "#dee2e6",
        "border_light": "#e9ecef",
        "nav_bg": "#ffffff",
        "nav_text": "#495057",
        "nav_hover": "#e9ecef",
        "nav_active": "#4CAF50",
        "tool_button": "#e9ecef",
        "tool_button_text": "#212529",
        "status_bg": "#e9ecef",
        "status_text": "#212529",
        "shadow": "rgba(0,0,0,0.1)"
    }
    
    def __init__(self):
        self.is_dark = True
        self.current = self.DARK
        self._cached_stylesheet = None
    
    def toggle(self):
        self.is_dark = not self.is_dark
        self.current = self.DARK if self.is_dark else self.LIGHT
        self._cached_stylesheet = None
        return self.current
    
    def get_nav_stylesheet(self):
        if self._cached_stylesheet is None:
            c = self.current
            self._cached_stylesheet = f"""
                /* ===== Global ===== */
                QWidget#navigationBar {{  
                    background-color: {c['nav_bg']};
                    border-bottom: 1px solid {c['border']};
                }}
                
                /* ===== Navigation Buttons ===== */
                QPushButton#navButton {{
                    background-color: transparent;
                    color: {c['nav_text']};
                    padding: 10px 20px;
                    border: none;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: 500;
                    margin: 4px 2px;
                }}
                
                QPushButton#navButton:hover {{
                    background-color: {c['nav_hover']};
                    color: {c['accent']};
                }}
                
                QPushButton#navButton:checked {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                              stop:0 {c['accent_gradient_start']},
                                              stop:1 {c['accent_gradient_end']});
                    color: white;
                }}
                
                QPushButton#navButton:pressed {{
                    background-color: {c['accent_hover']};
                }}
                
                /* ===== Tool Buttons ===== */
                QPushButton#toolButton {{
                    background-color: {c['tool_button']};
                    color: {c['tool_button_text']};
                    border: none;
                    border-radius: 6px;
                    font-size: 16px;
                    font-weight: bold;
                    min-width: 35px;
                    min-height: 30px;
                }}
                
                QPushButton#toolButton:hover {{
                    background-color: {c['accent']};
                    color: white;
                }}
                
                /* ===== Logo ===== */
                QLabel#logo {{
                    color: {c['accent']};
                    font-size: 20px;
                    font-weight: bold;
                    padding: 0 20px;
                }}
                
                /* ===== Separator ===== */
                QFrame#separator {{
                    color: {c['border']};
                    font-size: 20px;
                }}
                
                /* ===== Status Bar ===== */
                QStatusBar {{
                    background-color: {c['status_bg']};
                    color: {c['status_text']};
                    border-top: 1px solid {c['border']};
                    padding: 4px;
                }}
                
                /* ===== Stacked Widget ===== */
                QStackedWidget {{
                    background-color: {c['bg_primary']};
                }}
                
                /* ===== Scroll Area ===== */
                QScrollArea {{
                    background-color: transparent;
                    border: none;
                }}
                
                QScrollArea > QWidget > QWidget {{
                    background-color: transparent;
                }}
                
                /* ===== Loading Overlay ===== */
                QWidget#loadingOverlay {{
                    background-color: {c['bg_secondary']};
                    border: 1px solid {c['border']};
                    border-radius: 12px;
                }}
                
                QLabel#loadingTitle {{
                    color: {c['accent']};
                    font-size: 16px;
                    font-weight: bold;
                }}
                
                QLabel#loadingMessage {{
                    color: {c['text_secondary']};
                }}
                
                QProgressBar {{
                    border: none;
                    background-color: {c['bg_tertiary']};
                    border-radius: 4px;
                    text-align: center;
                    height: 8px;
                }}
                
                QProgressBar::chunk {{
                    background-color: {c['accent']};
                    border-radius: 4px;
                }}
            """
        return self._cached_stylesheet
    
    def apply_to_app(self, app):
        if self.is_dark:
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
            dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
            dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
            dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
            dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(76, 175, 80))
            dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
            app.setPalette(dark_palette)
        else:
            light_palette = QPalette()
            light_palette.setColor(QPalette.ColorRole.Window, QColor(248, 249, 250))
            light_palette.setColor(QPalette.ColorRole.WindowText, QColor(33, 37, 41))
            light_palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
            light_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(233, 236, 239))
            light_palette.setColor(QPalette.ColorRole.Text, QColor(33, 37, 41))
            light_palette.setColor(QPalette.ColorRole.Button, QColor(233, 236, 239))
            light_palette.setColor(QPalette.ColorRole.ButtonText, QColor(33, 37, 41))
            light_palette.setColor(QPalette.ColorRole.Highlight, QColor(76, 175, 80))
            light_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
            app.setPalette(light_palette)


# ==================== FONT SETUP (OPTIMIZED) ====================

def setup_application_font():
    """Fast font setup without heavy database query."""
    font = QFont()
    font.setPointSize(10)
    
    # Use safe defaults instead of scanning all families
    if sys.platform == 'win32':
        font.setFamily("Segoe UI")
    elif sys.platform == 'darwin':
        font.setFamily("SF Pro Text, Helvetica")
    else:
        font.setFamily("Ubuntu, Arial")
    
    return font


# ==================== SCROLLABLE WIDGET ====================

class ScrollableWidget(QScrollArea):
    """Makes any widget scrollable."""
    
    def __init__(self, widget):
        super().__init__()
        self.setWidget(widget)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFrameShape(QFrame.Shape.NoFrame)


# ==================== LOADING OVERLAY ====================

class LoadingOverlay(QWidget):
    """Modern loading overlay with progress indication."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("loadingOverlay")
        self.setup_ui()
        self.hide()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)
        
        # Title
        self.title_label = QLabel("Loading Module...")
        self.title_label.setObjectName("loadingTitle")
        layout.addWidget(self.title_label)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Infinite progress
        self.progress.setFixedWidth(300)
        layout.addWidget(self.progress)
        
        # Message
        self.message_label = QLabel("Please wait while the module initializes...")
        self.message_label.setObjectName("loadingMessage")
        layout.addWidget(self.message_label)
    
    def set_module(self, module_name: str):
        """Set which module is loading."""
        self.title_label.setText(f"Loading {module_name}...")
    
    def set_message(self, message: str):
        """Set loading message."""
        self.message_label.setText(message)


# ==================== MAIN WINDOW ====================

class AIStudioPro(QMainWindow):
    """Main unified application with lazy loading and modern UI."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Studio Pro")
        self.setMinimumSize(1200, 700)
        
        # Initialize theme manager
        self.theme = UnifiedThemeManager()
        
        # Initialize settings (lazy)
        self.settings_manager = None
        self.settings = None
        self._load_settings_lazy()
        
        # Store app instances (lazy loaded)
        self.speech_app = None
        self.tts_app = None
        self.download_manager = None
        self.overlap_tab = None
        self.speech_scroll = None
        self.tts_scroll = None
        
        # Store module constructors for lazy loading
        self.module_constructors = {}
        
        # Track which modules have been loaded
        self.loaded_modules = set()
        
        # Loading overlay
        self.loading_overlay = None
        
        # Navigation buttons
        self.speech_btn = None
        self.tts_btn = None
        self.download_btn = None
        self.overlap_btn = None
        
        # Setup UI first (fast)
        self.setup_ui()
        
        # Create placeholders (fast)
        self.create_placeholders()
        
        # Apply theme
        QTimer.singleShot(10, self.apply_initial_theme)
    
    def _load_settings_lazy(self):
        """Lazily load settings module."""
        if lazy_import('settings'):
            self.settings_manager = SettingsManager()
            self.settings = self.settings_manager.load()
    
    def setup_ui(self):
        """Create the modern unified UI with scrollable content area."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Navigation Bar (fixed, not scrollable)
        nav_bar = QWidget()
        nav_bar.setObjectName("navigationBar")
        nav_bar.setFixedHeight(60)
        
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(15, 0, 15, 0)
        nav_layout.setSpacing(10)
        
        # Logo with icon
        self.logo_label = QLabel("🎙️ AI STUDIO PRO")
        self.logo_label.setObjectName("logo")
        self.logo_label.setStyleSheet("font-size: 18px;")
        nav_layout.addWidget(self.logo_label)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFixedWidth(2)
        separator.setObjectName("separator")
        nav_layout.addWidget(separator)
        
        # Navigation buttons container
        nav_buttons = QHBoxLayout()
        nav_buttons.setSpacing(5)

        # Speech-to-Text button
        self.speech_btn = QPushButton("🎤 Speech-to-Text")
        self.speech_btn.setObjectName("navButton")
        self.speech_btn.setCheckable(True)
        self.speech_btn.setChecked(True)
        self.speech_btn.clicked.connect(lambda: self.switch_to('speech'))
        nav_buttons.addWidget(self.speech_btn)

        # TTS Merger button
        self.tts_btn = QPushButton("🔊 TTS Merger")
        self.tts_btn.setObjectName("navButton")
        self.tts_btn.setCheckable(True)
        self.tts_btn.clicked.connect(lambda: self.switch_to('tts'))
        nav_buttons.addWidget(self.tts_btn)

        # Download button
        self.download_btn = QPushButton("📥 Download")
        self.download_btn.setObjectName("navButton")
        self.download_btn.setCheckable(True)
        self.download_btn.clicked.connect(lambda: self.switch_to('download'))
        nav_buttons.addWidget(self.download_btn)

        # Overlap Detection button
        self.overlap_btn = QPushButton("🔊 Detect Overlaps")
        self.overlap_btn.setObjectName("navButton")
        self.overlap_btn.setCheckable(True)
        self.overlap_btn.clicked.connect(lambda: self.switch_to('overlap'))
        nav_buttons.addWidget(self.overlap_btn)

        nav_buttons.addStretch()
        nav_layout.addLayout(nav_buttons, 1)
        
        # Tool buttons
        tool_buttons = QHBoxLayout()
        tool_buttons.setSpacing(8)
        
        # Settings button
        self.settings_btn = QPushButton("⚙️")
        self.settings_btn.setObjectName("toolButton")
        self.settings_btn.setToolTip("Settings")
        self.settings_btn.setFixedSize(36, 32)
        self.settings_btn.clicked.connect(self.open_settings)
        tool_buttons.addWidget(self.settings_btn)
        
        # Theme toggle button
        self.theme_btn = QPushButton("🌓")
        self.theme_btn.setObjectName("toolButton")
        self.theme_btn.setToolTip("Toggle Theme")
        self.theme_btn.setFixedSize(36, 32)
        self.theme_btn.clicked.connect(self.toggle_theme)
        tool_buttons.addWidget(self.theme_btn)
        
        nav_layout.addLayout(tool_buttons)
        layout.addWidget(nav_bar)
        
        # ===== SCROLLABLE CONTENT AREA =====
        # Create a scroll area for the stacked widget
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Container for stacked widget
        scroll_container = QWidget()
        scroll_container.setObjectName("scrollContainer")
        scroll_layout = QVBoxLayout(scroll_container)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(0)
        
        # Stacked Widget for apps with fade effect
        self.stacked = QStackedWidget()
        self.stacked.setObjectName("stackedWidget")
        
        # Add fade animation
        self.fade_animation = QPropertyAnimation(self.stacked, b"windowOpacity")
        self.fade_animation.setDuration(150)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        scroll_layout.addWidget(self.stacked)
        self.scroll_area.setWidget(scroll_container)
        layout.addWidget(self.scroll_area, 1)
        
        # Loading overlay (initially hidden)
        self.loading_overlay = LoadingOverlay(self.stacked)
        self.loading_overlay.resize(self.stacked.size())
        
        # Status Bar with module icon
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status("Ready", "🎙️")
    
    def resizeEvent(self, event):
        """Handle resize to keep overlay positioned correctly."""
        super().resizeEvent(event)
        if self.loading_overlay:
            self.loading_overlay.resize(self.stacked.size())
    
    def update_status(self, message: str, icon: str = "🎙️"):
        """Update status bar with icon and message."""
        self.status_bar.showMessage(f"{icon} {message}")
    
    def create_placeholders(self):
        """Create placeholder widgets for lazy loading."""
        # Clear existing widgets if any
        while self.stacked.count() > 0:
            widget = self.stacked.widget(0)
            self.stacked.removeWidget(widget)
            widget.deleteLater()
        
        # Speech placeholder
        speech_placeholder = QWidget()
        speech_placeholder.setObjectName("speechPlaceholder")
        self.stacked.addWidget(speech_placeholder)
        
        # TTS placeholder
        tts_placeholder = QWidget()
        tts_placeholder.setObjectName("ttsPlaceholder")
        self.stacked.addWidget(tts_placeholder)
        
        # Download placeholder
        download_placeholder = QWidget()
        download_placeholder.setObjectName("downloadPlaceholder")
        self.stacked.addWidget(download_placeholder)
        
        # Overlap placeholder
        overlap_placeholder = QWidget()
        overlap_placeholder.setObjectName("overlapPlaceholder")
        self.stacked.addWidget(overlap_placeholder)
        
        # Store indices
        self.app_indices = {
            'speech': 0,
            'tts': 1,
            'download': 2,
            'overlap': 3
        }
        
        print(f"📊 Created {self.stacked.count()} placeholders")
    
    def _create_speech_module(self):
        """Lazily create and load speech module."""
        if 'speech' in self.loaded_modules:
            return self.speech_app
        
        self.loading_overlay.set_module("Speech-to-Text")
        self.loading_overlay.show()
        QApplication.processEvents()
        
        try:
            if lazy_import('speech_to_text') and SpeechToTextWindow:
                self.speech_app = SpeechToTextWindow()
                
                # Hide its menu bar and toolbar
                if hasattr(self.speech_app, 'menuBar') and self.speech_app.menuBar():
                    self.speech_app.menuBar().hide()
                if hasattr(self.speech_app, 'toolbar'):
                    self.speech_app.toolbar.hide()
                if hasattr(self.speech_app, 'theme_btn'):
                    self.speech_app.theme_btn.hide()
                
                # Wrap in scrollable widget
                self.speech_scroll = ScrollableWidget(self.speech_app)
                
                # Replace placeholder
                self.stacked.removeWidget(self.stacked.widget(self.app_indices['speech']))
                self.stacked.insertWidget(self.app_indices['speech'], self.speech_scroll)
                
                self.loaded_modules.add('speech')
                print("✅ Speech-to-Text loaded lazily")
            else:
                error_widget = self._create_error_widget("Speech-to-Text module not available")
                self.stacked.removeWidget(self.stacked.widget(self.app_indices['speech']))
                self.stacked.insertWidget(self.app_indices['speech'], error_widget)
        
        except Exception as e:
            print(f"❌ Error loading Speech-to-Text: {e}")
            traceback.print_exc()
            error_widget = self._create_error_widget(f"Failed to load: {str(e)}")
            self.stacked.removeWidget(self.stacked.widget(self.app_indices['speech']))
            self.stacked.insertWidget(self.app_indices['speech'], error_widget)
        
        finally:
            self.loading_overlay.hide()
        
        return self.speech_app
    
    def _create_tts_module(self):
        """Lazily create and load TTS module."""
        if 'tts' in self.loaded_modules:
            return self.tts_app
        
        self.loading_overlay.set_module("TTS Merger")
        self.loading_overlay.show()
        QApplication.processEvents()
        
        try:
            if lazy_import('text_to_speech') and TTSMergerWindow:
                self.tts_app = TTSMergerWindow()
                
                # Hide toolbar if exists
                if hasattr(self.tts_app, 'toolbar'):
                    self.tts_app.toolbar.hide()
                if hasattr(self.tts_app, 'theme_btn'):
                    self.tts_app.theme_btn.hide()
                
                # Wrap in scrollable widget
                self.tts_scroll = ScrollableWidget(self.tts_app)
                
                # Replace placeholder
                self.stacked.removeWidget(self.stacked.widget(self.app_indices['tts']))
                self.stacked.insertWidget(self.app_indices['tts'], self.tts_scroll)
                
                self.loaded_modules.add('tts')
                print("✅ TTS Merger loaded lazily")
            else:
                error_widget = self._create_error_widget("TTS Merger module not available")
                self.stacked.removeWidget(self.stacked.widget(self.app_indices['tts']))
                self.stacked.insertWidget(self.app_indices['tts'], error_widget)
        
        except Exception as e:
            print(f"❌ Error loading TTS Merger: {e}")
            traceback.print_exc()
            error_widget = self._create_error_widget(f"Failed to load: {str(e)}")
            self.stacked.removeWidget(self.stacked.widget(self.app_indices['tts']))
            self.stacked.insertWidget(self.app_indices['tts'], error_widget)
        
        finally:
            self.loading_overlay.hide()
        
        return self.tts_app
    
    def _create_download_module(self):
        """Lazily create and load download module."""
        if 'download' in self.loaded_modules:
            return self.download_manager
        
        self.loading_overlay.set_module("Download Manager")
        self.loading_overlay.show()
        QApplication.processEvents()
        
        try:
            if lazy_import('download') and DownloadManagerWidget:
                self.download_manager = DownloadManagerWidget(self, self.theme)
                if hasattr(self.download_manager, 'download_complete'):
                    self.download_manager.download_complete.connect(self.on_download_complete)
                
                self.stacked.removeWidget(self.stacked.widget(self.app_indices['download']))
                self.stacked.insertWidget(self.app_indices['download'], self.download_manager)
                
                self.loaded_modules.add('download')
                print("✅ Download Manager loaded lazily")
            else:
                error_widget = self._create_error_widget("Download module not available")
                self.stacked.removeWidget(self.stacked.widget(self.app_indices['download']))
                self.stacked.insertWidget(self.app_indices['download'], error_widget)
        
        except Exception as e:
            print(f"❌ Error loading Download Manager: {e}")
            traceback.print_exc()
            error_widget = self._create_error_widget(f"Failed to load: {str(e)}")
            self.stacked.removeWidget(self.stacked.widget(self.app_indices['download']))
            self.stacked.insertWidget(self.app_indices['download'], error_widget)
        
        finally:
            self.loading_overlay.hide()
        
        return self.download_manager
    
    def _create_overlap_module(self):
        """Lazily create and load overlap detection module."""
        if 'overlap' in self.loaded_modules:
            return self.overlap_tab
        
        self.loading_overlay.set_module("Overlap Detector")
        self.loading_overlay.show()
        QApplication.processEvents()
        
        try:
            if lazy_import('audio_detect_overlap') and OverlapDetectorTab:
                self.overlap_tab = OverlapDetectorTab(self, self.theme)
                if hasattr(self.overlap_tab, 'file_loaded'):
                    self.overlap_tab.file_loaded.connect(self.on_file_loaded)
                
                self.stacked.removeWidget(self.stacked.widget(self.app_indices['overlap']))
                self.stacked.insertWidget(self.app_indices['overlap'], self.overlap_tab)
                
                self.loaded_modules.add('overlap')
                print("✅ Overlap Detector loaded lazily")
            else:
                error_widget = self._create_error_widget("Overlap Detection module not available")
                self.stacked.removeWidget(self.stacked.widget(self.app_indices['overlap']))
                self.stacked.insertWidget(self.app_indices['overlap'], error_widget)
        
        except Exception as e:
            print(f"❌ Error loading Overlap Detector: {e}")
            traceback.print_exc()
            error_widget = self._create_error_widget(f"Failed to load: {str(e)}")
            self.stacked.removeWidget(self.stacked.widget(self.app_indices['overlap']))
            self.stacked.insertWidget(self.app_indices['overlap'], error_widget)
        
        finally:
            self.loading_overlay.hide()
        
        return self.overlap_tab
    
    def _create_error_widget(self, message: str) -> QWidget:
        """Create a simple error widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        error_label = QLabel(f"❌ {message}")
        error_label.setStyleSheet("color: #f44336; font-size: 14px;")
        layout.addWidget(error_label)
        
        return widget
    
    def switch_to(self, app_name):
        """Switch between apps with lazy loading and fade effect."""
        if app_name not in self.app_indices:
            return
        
        index = self.app_indices[app_name]
        
        # Fade out
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.3)
        self.fade_animation.start()
        
        # Load module if needed
        if app_name == 'speech' and app_name not in self.loaded_modules:
            self._create_speech_module()
        elif app_name == 'tts' and app_name not in self.loaded_modules:
            self._create_tts_module()
        elif app_name == 'download' and app_name not in self.loaded_modules:
            self._create_download_module()
        elif app_name == 'overlap' and app_name not in self.loaded_modules:
            self._create_overlap_module()
        
        # Switch tab
        if index < self.stacked.count():
            self.stacked.setCurrentIndex(index)
            
            # Update button states
            if self.speech_btn:
                self.speech_btn.setChecked(app_name == 'speech')
            if self.tts_btn:
                self.tts_btn.setChecked(app_name == 'tts')
            if self.download_btn:
                self.download_btn.setChecked(app_name == 'download')
            if self.overlap_btn:
                self.overlap_btn.setChecked(app_name == 'overlap')
            
            # Update status
            icons = {
                'speech': '🎤',
                'tts': '🔊',
                'download': '📥',
                'overlap': '🔊'
            }
            names = {
                'speech': 'Speech-to-Text',
                'tts': 'TTS Merger',
                'download': 'Download Manager',
                'overlap': 'Overlap Detector'
            }
            self.update_status(names.get(app_name, app_name), icons.get(app_name, '🎙️'))
        
        # Fade in
        QTimer.singleShot(100, lambda: self.fade_animation.setEndValue(1.0))
    
    def open_settings(self):
        """Open settings dialog (lazy loaded)."""
        try:
            if not lazy_import('settings'):
                QMessageBox.information(self, "Settings", "Settings module not available")
                return
            
            if not self.settings_manager or not self.settings:
                # Try to initialize settings again
                self._load_settings_lazy()
                if not self.settings_manager or not self.settings:
                    QMessageBox.warning(self, "Settings", "Could not load settings")
                    return
            
            if SettingsDialog:
                dialog = SettingsDialog(self, self.settings, self.theme)
                result = dialog.exec()
                
                if result == QDialog.DialogCode.Accepted:
                    if self.settings_manager:
                        self.settings_manager.save(self.settings)
                        self.status_bar.showMessage("⚙️ Settings saved", 3000)
        except Exception as e:
            print(f"❌ Error opening settings: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to open settings: {str(e)}")

    def on_download_complete(self, file_path):
        """Handle download completion."""
        self.status_bar.showMessage(f"✅ Download complete: {os.path.basename(file_path)}", 3000)
    
    def on_file_loaded(self, file_path):
        """Handle file loaded in overlap detector or video editor."""
        self.status_bar.showMessage(f"📂 Loaded: {os.path.basename(file_path)}", 2000)
    
    def apply_initial_theme(self):
        """Apply initial theme."""
        self.apply_theme_to_all()
    
    def apply_theme_to_all(self):
        """Apply current theme to all components."""
        # Update navigation bar
        nav_bar = self.findChild(QWidget, "navigationBar")
        if nav_bar:
            nav_bar.setStyleSheet(self.theme.get_nav_stylesheet())
        
        # Update all navigation buttons
        for btn in [self.speech_btn, self.tts_btn, self.download_btn, self.overlap_btn]:
            if btn:
                btn.setStyleSheet(self.theme.get_nav_stylesheet())
        
        # Update tool buttons
        if hasattr(self, 'settings_btn') and self.settings_btn:
            self.settings_btn.setStyleSheet(self.theme.get_nav_stylesheet())
        if hasattr(self, 'theme_btn') and self.theme_btn:
            self.theme_btn.setStyleSheet(self.theme.get_nav_stylesheet())
        
        # Update logo
        if hasattr(self, 'logo_label') and self.logo_label:
            c = self.theme.current
            self.logo_label.setStyleSheet(f"""
                QLabel#logo {{
                    color: {c['accent']};
                    font-size: 18px;
                    font-weight: bold;
                    padding: 0 15px;
                }}
            """)
        
        # Apply to QApplication
        self.theme.apply_to_app(QApplication.instance())
        
        # Apply to loaded modules
        if hasattr(self, 'speech_app') and self.speech_app:
            try:
                if hasattr(self.speech_app, 'apply_theme'):
                    self.speech_app.apply_theme(self.theme.is_dark)
            except Exception as e:
                print(f"⚠️ Error applying theme to speech module: {e}")
        
        if hasattr(self, 'tts_app') and self.tts_app:
            try:
                if hasattr(self.tts_app, 'apply_theme'):
                    self.tts_app.apply_theme(self.theme.is_dark)
            except Exception as e:
                print(f"⚠️ Error applying theme to TTS module: {e}")
        
        if hasattr(self, 'download_manager') and self.download_manager:
            try:
                if hasattr(self.download_manager, 'setStyleSheet'):
                    self.download_manager.setStyleSheet(self.theme.get_nav_stylesheet())
            except Exception as e:
                print(f"⚠️ Error applying theme to download module: {e}")
        
        if hasattr(self, 'overlap_tab') and self.overlap_tab:
            try:
                if hasattr(self.overlap_tab, 'apply_theme'):
                    self.overlap_tab.apply_theme(self.theme.is_dark)
            except Exception as e:
                print(f"⚠️ Error applying theme to overlap module: {e}")
        
        # Update status bar
        if hasattr(self, 'status_bar') and self.status_bar:
            self.status_bar.setStyleSheet(f"""
                QStatusBar {{
                    background-color: {self.theme.current['status_bg']};
                    color: {self.theme.current['status_text']};
                    border-top: 1px solid {self.theme.current['border']};
                    padding: 4px;
                }}
            """)
        
        # Update loading overlay if it exists
        if hasattr(self, 'loading_overlay') and self.loading_overlay:
            self.loading_overlay.setStyleSheet(f"""
                QWidget#loadingOverlay {{
                    background-color: {self.theme.current['bg_secondary']};
                    border: 1px solid {self.theme.current['border']};
                    border-radius: 12px;
                }}
                QLabel#loadingTitle {{
                    color: {self.theme.current['accent']};
                    font-size: 16px;
                    font-weight: bold;
                }}
                QLabel#loadingMessage {{
                    color: {self.theme.current['text_secondary']};
                }}
                QProgressBar {{
                    border: none;
                    background-color: {self.theme.current['bg_tertiary']};
                    border-radius: 4px;
                    text-align: center;
                    height: 8px;
                }}
                QProgressBar::chunk {{
                    background-color: {self.theme.current['accent']};
                    border-radius: 4px;
                }}
            """)
        
        # Force update
        QApplication.processEvents()
        
        print(f"🎨 Theme applied ({'Dark' if self.theme.is_dark else 'Light'})")
    
    def toggle_theme(self):
        """Toggle between dark and light theme."""
        self.theme.toggle()
        self.theme_btn.setText("🌞" if self.theme.is_dark else "🌓")
        self.apply_theme_to_all()
        self.status_bar.showMessage(f"🎨 Switched to {'Dark' if self.theme.is_dark else 'Light'} theme", 2000)
    
    def closeEvent(self, event):
        """Clean up on close."""
        try:
            if self.settings_manager and hasattr(self, 'settings'):
                self.settings_manager.save(self.settings)
        except:
            pass
        event.accept()


# ==================== MAIN ====================

def main():
    """Main entry point with fast startup."""
    app = QApplication(sys.argv)
    app.setApplicationName("AI Studio Pro")
    
    # Set font
    app_font = setup_application_font()
    app.setFont(app_font)
    
    # Show startup message
    print("\n" + "="*50)
    print("🚀 AI Studio Pro - Starting up...")
    print("="*50)
    
    # Create and show window
    window = AIStudioPro()
    window.show()
    
    # Show missing packages warning (non-blocking)
    QTimer.singleShot(500, lambda: show_missing_warnings(window))
    
    sys.exit(app.exec())
def show_missing_warnings(parent):
    """Show non-blocking warnings for missing packages."""
    missing = []
    
    # Check each module
    if not lazy_import('download'):
        missing.append("yt-dlp (Download module)")
    if not lazy_import('audio_detect_overlap'):
        missing.append("pydub (Overlap Detection)")
    if not lazy_import('speech_to_text'):
        missing.append("faster-whisper (Speech-to-Text)")
    if not lazy_import('text_to_speech'):
        missing.append("edge-tts (TTS Merger)")
    
    if missing:
        msg = "Some features may be limited:\n\n"
        for pkg in missing:
            msg += f"• {pkg}\n"
        msg += "\nInstall missing packages with:\npip install yt-dlp pydub faster-whisper edge-tts"       
        QMessageBox.information(parent, "Optional Features", msg)
if __name__ == "__main__":
    main()