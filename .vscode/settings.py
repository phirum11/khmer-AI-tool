"""
AI Studio Pro - Settings Module (Modernized)
- Centralized settings management
- Modern, premium UI with live preview
- Category-based organization
- Real-time theme preview
- Fast search with highlighting
- Professional styling with reduced code
"""

import os
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QWidget, QFormLayout, QLineEdit, QComboBox,
    QSpinBox, QCheckBox, QGroupBox, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QDialogButtonBox, QColorDialog,
    QFontDialog, QSlider, QFrame, QStackedWidget,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QApplication,
    QScrollArea, QKeySequenceEdit, QToolButton, QSplitter
)
from PyQt6.QtCore import Qt, QSettings, QByteArray, QSize, QTimer, pyqtSignal, QRect
from PyQt6.QtGui import QFont, QColor, QPalette, QKeySequence, QShortcut, QPainter, QLinearGradient


# ==================== ENUMS & CONSTANTS ====================

class Theme(Enum):
    DARK = "dark"
    LIGHT = "light"
    SYSTEM = "system"


class LayoutStyle(Enum):
    COMPACT = "compact"
    COMFORTABLE = "comfortable"
    SPACIOUS = "spacious"


# ==================== DATA CLASSES ====================

@dataclass
class AppSettings:
    """Main application settings data class."""
    # Theme settings
    theme: str = "dark"
    accent_color: str = "#4CAF50"
    font_family: str = "Segoe UI"
    font_size: int = 10
    
    # Layout settings
    layout_style: str = "comfortable"
    show_toolbar: bool = True
    show_statusbar: bool = True
    show_waveform: bool = True
    panel_sizes: Dict[str, int] = field(default_factory=lambda: {
        'left_panel': 300,
        'right_panel': 900
    })
    
    # Download settings
    download_location: str = os.path.expanduser("~/Downloads")
    default_format: str = "audio"
    audio_format: str = "mp3"
    audio_quality: str = "192"
    video_quality: str = "best"
    max_concurrent_downloads: int = 3
    auto_open_folder: bool = True
    auto_load_to_app: bool = True
    
    # Speech-to-Text settings
    stt_default_engine: str = "faster-whisper"
    stt_model_size: str = "base"
    stt_language: str = "auto"
    stt_vad_filter: bool = True
    stt_auto_save: bool = True
    
    # TTS settings
    tts_default_voice: str = "km-KH-SreymomNeural"
    tts_speed: int = 100
    tts_volume: int = 100
    tts_crossfade: int = 50
    tts_normalize: bool = True
    
    # Window settings
    window_geometry: Optional[QByteArray] = None
    window_state: Optional[QByteArray] = None
    last_tab: int = 0
    
    # Recent files
    recent_files: List[str] = field(default_factory=list)
    recent_urls: List[str] = field(default_factory=list)
    
    # Shortcuts
    shortcuts: Dict[str, str] = field(default_factory=lambda: {
        'new_download': 'Ctrl+D',
        'start_transcribe': 'Ctrl+R',
        'open_tts': 'Ctrl+T',
        'settings': 'Ctrl+,',
        'quit': 'Ctrl+Q',
        'switch_theme': 'Ctrl+Shift+T',
        'search': 'Ctrl+F',
        'export': 'Ctrl+E',
        'save': 'Ctrl+S',
        'open_file': 'Ctrl+O'
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.window_geometry:
            data['window_geometry'] = self.window_geometry.toBase64().data().decode()
        if self.window_state:
            data['window_state'] = self.window_state.toBase64().data().decode()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppSettings':
        """Create from dictionary."""
        if 'window_geometry' in data and data['window_geometry']:
            data['window_geometry'] = QByteArray.fromBase64(data['window_geometry'].encode())
        if 'window_state' in data and data['window_state']:
            data['window_state'] = QByteArray.fromBase64(data['window_state'].encode())
        return cls(**data)


# ==================== SETTING BINDING SYSTEM ====================

class Setting:
    """Declarative setting binding."""
    
    def __init__(self, key: str, widget_getter: Callable, widget_setter: Callable,
                 converter: Callable = None, reverse_converter: Callable = None):
        self.key = key
        self.widget_getter = widget_getter
        self.widget_setter = widget_setter
        self.converter = converter or (lambda x: x)
        self.reverse_converter = reverse_converter or (lambda x: x)
    
    def load(self, settings: AppSettings):
        """Load value from settings to widget."""
        value = getattr(settings, self.key, None)
        if value is not None:
            self.widget_setter(self.converter(value))
    
    def save(self, settings: AppSettings):
        """Save value from widget to settings."""
        value = self.widget_getter()
        setattr(settings, self.key, self.reverse_converter(value))


# ==================== THEME MANAGER ====================

class ThemeManager:
    """Manages application themes and styling with real-time preview."""
    
    DARK_PALETTE = {
        'window': '#1e1e1e',
        'window_text': '#ffffff',
        'base': '#252526',
        'alternate_base': '#2d2d30',
        'text': '#ffffff',
        'text_muted': '#aaaaaa',
        'button': '#3a3a3a',
        'button_text': '#ffffff',
        'highlight': '#4CAF50',
        'highlighted_text': '#000000',
        'tooltip_base': '#2d2d30',
        'tooltip_text': '#ffffff',
        'disabled': '#666666',
        'border': '#3c3c3c',
        'border_light': '#4a4a4a',
        'shadow': '#00000040',
        'success': '#4CAF50',
        'warning': '#ff9800',
        'error': '#f44336',
        'info': '#2196F3'
    }
    
    LIGHT_PALETTE = {
        'window': '#f5f5f5',
        'window_text': '#333333',
        'base': '#ffffff',
        'alternate_base': '#f0f0f0',
        'text': '#333333',
        'text_muted': '#666666',
        'button': '#e0e0e0',
        'button_text': '#333333',
        'highlight': '#4CAF50',
        'highlighted_text': '#ffffff',
        'tooltip_base': '#ffffe1',
        'tooltip_text': '#333333',
        'disabled': '#a0a0a0',
        'border': '#cccccc',
        'border_light': '#dddddd',
        'shadow': '#00000020',
        'success': '#4CAF50',
        'warning': '#ff9800',
        'error': '#f44336',
        'info': '#2196F3'
    }
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.accent_color = QColor(settings.accent_color)
    
    def get_palette(self, theme_override: str = None) -> QPalette:
        """Get QPalette based on current theme or override."""
        theme = theme_override if theme_override else self.settings.theme
        palette_data = self.DARK_PALETTE if theme == 'dark' else self.LIGHT_PALETTE
        
        palette = QPalette()
        
        palette.setColor(QPalette.ColorRole.Window, QColor(palette_data['window']))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(palette_data['window_text']))
        palette.setColor(QPalette.ColorRole.Base, QColor(palette_data['base']))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(palette_data['alternate_base']))
        palette.setColor(QPalette.ColorRole.Text, QColor(palette_data['text']))
        palette.setColor(QPalette.ColorRole.Button, QColor(palette_data['button']))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(palette_data['button_text']))
        palette.setColor(QPalette.ColorRole.Highlight, self.accent_color)
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(palette_data['highlighted_text']))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(palette_data['tooltip_base']))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(palette_data['tooltip_text']))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(palette_data['disabled']))
        
        return palette
    
    def get_stylesheet(self, theme_override: str = None) -> str:
        """Generate dynamic stylesheet with proper color variables."""
        theme = theme_override if theme_override else self.settings.theme
        is_dark = theme == 'dark'
        c = self.DARK_PALETTE if is_dark else self.LIGHT_PALETTE
        
        # Color variables for reuse
        colors = {
            'bg': c['window'],
            'bg-light': c['base'],
            'bg-dark': c['alternate_base'],
            'text': c['text'],
            'text-muted': c['text_muted'],
            'border': c['border'],
            'border-light': c['border_light'],
            'accent': self.accent_color.name(),
            'accent-light': self.lighten_color(self.accent_color, 30),
            'accent-dark': self.darken_color(self.accent_color, 20),
            'success': c['success'],
            'warning': c['warning'],
            'error': c['error'],
            'info': c['info'],
            'shadow': c['shadow']
        }
        
        return f"""
        /* ===== Global ===== */
        * {{
            font-family: '{self.settings.font_family}';
            font-size: {self.settings.font_size}pt;
        }}
        
        QWidget {{
            background-color: {colors['bg']};
            color: {colors['text']};
        }}
        
        QDialog, QMainWindow {{
            background: {colors['bg']};
        }}
        
        /* ===== Buttons ===== */
        QPushButton {{
            background: {colors['bg-light']};
            color: {colors['text']};
            border: 1px solid {colors['border']};
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 500;
            min-width: 80px;
        }}
        
        QPushButton:hover {{
            background: {colors['bg-dark']};
            border-color: {colors['accent']};
        }}
        
        QPushButton:pressed {{
            background: {colors['accent-dark']};
            color: white;
        }}
        
        QPushButton:checked {{
            background: {colors['accent']};
            color: white;
            border: none;
        }}
        
        QPushButton:checked:hover {{
            background: {colors['accent-light']};
        }}
        
        QPushButton:disabled {{
            opacity: 0.5;
        }}
        
        /* Primary buttons */
        QPushButton.primary {{
            background: {colors['accent']};
            color: white;
            border: none;
        }}
        
        QPushButton.primary:hover {{
            background: {colors['accent-light']};
        }}
        
        QPushButton.danger {{
            background: transparent;
            color: {colors['error']};
            border: 1px solid {colors['error']};
        }}
        
        QPushButton.danger:hover {{
            background: {colors['error']};
            color: white;
        }}
        
        QPushButton.success {{
            background: {colors['success']};
            color: white;
            border: none;
        }}
        
        QPushButton.success:hover {{
            background: {self.lighten_color(QColor(colors['success']), 20)};
        }}
        
        /* Tool buttons */
        QToolButton {{
            background: transparent;
            border: none;
            border-radius: 6px;
            padding: 6px;
        }}
        
        QToolButton:hover {{
            background: {colors['bg-dark']};
        }}
        
        QToolButton:checked {{
            background: {colors['accent']}20;
            color: {colors['accent']};
        }}
        
        /* ===== Inputs ===== */
        QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QKeySequenceEdit {{
            background: {colors['bg-light']};
            color: {colors['text']};
            border: 1px solid {colors['border']};
            border-radius: 6px;
            padding: 6px 10px;
            selection-background-color: {colors['accent']}30;
        }}
        
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QKeySequenceEdit:focus {{
            border: 2px solid {colors['accent']};
            padding: 5px 9px;
        }}
        
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid {colors['text']};
            width: 0;
            height: 0;
        }}
        
        /* ===== GroupBoxes ===== */
        QGroupBox {{
            border: 1px solid {colors['border']};
            border-radius: 10px;
            margin-top: 16px;
            padding-top: 12px;
            font-weight: 600;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 16px;
            padding: 0 8px;
            color: {colors['accent']};
        }}
        
        /* ===== ScrollBars ===== */
        QScrollBar:vertical {{
            background: {colors['bg-light']};
            width: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background: {colors['border']};
            border-radius: 6px;
            min-height: 30px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background: {colors['accent']};
        }}
        
        QScrollBar:horizontal {{
            background: {colors['bg-light']};
            height: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:horizontal {{
            background: {colors['border']};
            border-radius: 6px;
            min-width: 30px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background: {colors['accent']};
        }}
        
        /* ===== Sliders ===== */
        QSlider::groove:horizontal {{
            height: 6px;
            background: {colors['border']};
            border-radius: 3px;
        }}
        
        QSlider::handle:horizontal {{
            background: {colors['accent']};
            width: 18px;
            height: 18px;
            margin: -6px 0;
            border-radius: 9px;
        }}
        
        QSlider::handle:horizontal:hover {{
            background: {colors['accent-light']};
            transform: scale(1.1);
        }}
        
        QSlider::sub-page:horizontal {{
            background: {colors['accent']};
            border-radius: 3px;
        }}
        
        /* ===== Checkboxes & Radio ===== */
        QCheckBox, QRadioButton {{
            spacing: 8px;
        }}
        
        QCheckBox::indicator, QRadioButton::indicator {{
            width: 20px;
            height: 20px;
            border: 2px solid {colors['border']};
            border-radius: 4px;
            background: {colors['bg-light']};
        }}
        
        QCheckBox::indicator:checked {{
            background: {colors['accent']};
            border-color: {colors['accent']};
            image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='white'><path d='M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z'/></svg>");
        }}
        
        QRadioButton::indicator {{
            border-radius: 10px;
        }}
        
        QRadioButton::indicator:checked {{
            background: {colors['accent']};
            border-color: {colors['accent']};
        }}
        
        QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
            border-color: {colors['accent']};
        }}
        
        /* ===== Lists & Trees ===== */
        QListWidget, QTreeWidget {{
            background: {colors['bg-light']};
            border: 1px solid {colors['border']};
            border-radius: 8px;
            outline: none;
        }}
        
        QListWidget::item, QTreeWidget::item {{
            padding: 8px;
            border-radius: 6px;
            margin: 2px 4px;
        }}
        
        QListWidget::item:selected, QTreeWidget::item:selected {{
            background: {colors['accent']}30;
            color: {colors['accent']};
        }}
        
        QListWidget::item:hover, QTreeWidget::item:hover {{
            background: {colors['bg-dark']};
        }}
        
        /* ===== Progress Bar ===== */
        QProgressBar {{
            border: 1px solid {colors['border']};
            border-radius: 5px;
            text-align: center;
            background: {colors['bg-light']};
        }}
        
        QProgressBar::chunk {{
            background: {colors['accent']};
            border-radius: 5px;
        }}
        
        /* ===== Status Bar ===== */
        QStatusBar {{
            background: {colors['bg-light']};
            color: {colors['text-muted']};
            border-top: 1px solid {colors['border']};
        }}
        
        /* ===== Tab Bar ===== */
        QTabWidget::pane {{
            border: 1px solid {colors['border']};
            border-radius: 8px;
            background: {colors['bg-light']};
        }}
        
        QTabBar::tab {{
            background: transparent;
            color: {colors['text-muted']};
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
        }}
        
        QTabBar::tab:selected {{
            background: {colors['bg-light']};
            color: {colors['accent']};
            border: 1px solid {colors['border']};
            border-bottom: none;
        }}
        
        QTabBar::tab:hover:!selected {{
            background: {colors['bg-dark']};
            color: {colors['text']};
        }}
        """
    
    @staticmethod
    def lighten_color(color: QColor, percent: int) -> str:
        """Lighten a color by percentage."""
        h = color.hue()
        s = color.saturation()
        v = min(255, color.value() + int(255 * percent / 100))
        return QColor.fromHsv(h, s, v).name()
    
    @staticmethod
    def darken_color(color: QColor, percent: int) -> str:
        """Darken a color by percentage."""
        h = color.hue()
        s = color.saturation()
        v = max(0, color.value() - int(255 * percent / 100))
        return QColor.fromHsv(h, s, v).name()


# ==================== THEME PREVIEW WIDGET ====================

class ThemePreviewWidget(QWidget):
    """Realistic theme preview widget."""
    
    def __init__(self, theme_manager):
        super().__init__()
        self.theme_manager = theme_manager
        self.setMinimumHeight(150)
        self.setMaximumHeight(150)
        self.setMinimumWidth(200)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Handle both ThemeManager and UnifiedThemeManager
        if hasattr(self.theme_manager, 'settings'):
            is_dark = self.theme_manager.settings.theme == 'dark'
            if hasattr(self.theme_manager, 'DARK_PALETTE'):
                c = self.theme_manager.DARK_PALETTE if is_dark else self.theme_manager.LIGHT_PALETTE
            else:
                c = self.theme_manager.DARK if is_dark else self.theme_manager.LIGHT
        else:
            is_dark = self.theme_manager.is_dark
            c = self.theme_manager.DARK if is_dark else self.theme_manager.LIGHT
        
        accent = self.theme_manager.accent_color
        
        # Draw main window background
        painter.fillRect(0, 0, width, height, QColor(c.get('window', c.get('bg_primary', '#1e1e1e'))))
        
        # Draw title bar
        title_rect = QRect(0, 0, width, 30)
        painter.fillRect(title_rect, accent)
        
        # Window controls
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 180))
        painter.drawEllipse(10, 10, 10, 10)
        painter.drawEllipse(25, 10, 10, 10)
        painter.drawEllipse(40, 10, 10, 10)
        
        # Draw navigation sidebar
        sidebar_rect = QRect(0, 30, 50, height - 30)
        painter.fillRect(sidebar_rect, QColor(c.get('base', c.get('bg_secondary', '#252526'))))
        
        # Draw content area
        content_rect = QRect(50, 30, width - 50, height - 30)
        painter.fillRect(content_rect, QColor(c.get('alternate_base', c.get('bg_tertiary', '#2d2d30'))))
        
        # Draw sample UI elements
        y = 50
        
        # Sample button
        btn_rect = QRect(70, y, 80, 24)
        painter.fillRect(btn_rect, QColor(c.get('button', c.get('tool_button', '#3a3a3a'))))
        painter.setPen(QColor(c.get('border', '#3c3c3c')))
        painter.drawRect(btn_rect)
        painter.setPen(QColor(c.get('text', c.get('text_primary', '#ffffff'))))
        painter.drawText(btn_rect, Qt.AlignmentFlag.AlignCenter, "Button")
        
        # Sample input
        y += 35
        input_rect = QRect(70, y, 120, 24)
        painter.fillRect(input_rect, QColor(c.get('base', c.get('bg_secondary', '#252526'))))
        painter.setPen(QColor(c.get('border', '#3c3c3c')))
        painter.drawRect(input_rect)
        painter.setPen(QColor(c.get('text_muted', c.get('text_secondary', '#cccccc'))))
        painter.drawText(input_rect.adjusted(5, 0, -5, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "Input")
        
        # Sample slider
        y += 35
        slider_y = y + 10
        painter.setPen(QColor(c.get('border', '#3c3c3c')))
        painter.drawLine(70, slider_y, 190, slider_y)
        painter.setBrush(accent)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(130, slider_y - 6, 12, 12)
        
        # Sample checkbox
        y += 35
        painter.setPen(QColor(c.get('border', '#3c3c3c')))
        painter.drawRect(70, y, 16, 16)
        painter.setPen(QColor(c.get('text', c.get('text_primary', '#ffffff'))))
        painter.drawText(95, y + 13, "Option")
        
        # Sample accent elements
        accent_rect = QRect(70, y + 25, 30, 30)
        painter.fillRect(accent_rect, accent)


# ==================== UNIFIED THEME MANAGER ====================

class UnifiedThemeManager:
    """Manages dark/light themes for all apps with modern styling."""
    
    def __init__(self):
        self.is_dark = True
        self.accent_color = QColor(76, 175, 80)  # Default green accent
        self.settings = None  # Will be set from AppSettings
        
        # Dark theme palette
        self.DARK = {
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
        
        # Light theme palette
        self.LIGHT = {
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
        
        # Start with dark theme
        self.current = self.DARK
    
    def toggle(self):
        """Toggle between dark and light themes."""
        self.is_dark = not self.is_dark
        self.current = self.DARK if self.is_dark else self.LIGHT
        return self.current
    
    def set_theme(self, is_dark: bool):
        """Set theme explicitly."""
        self.is_dark = is_dark
        self.current = self.DARK if self.is_dark else self.LIGHT
    
    def get_stylesheet(self) -> str:
        """Get the complete stylesheet for the current theme."""
        c = self.current
        accent_hex = self.accent_color.name() if hasattr(self.accent_color, 'name') else str(self.accent_color)
        
        return f"""
            /* ===== Global ===== */
            QWidget {{
                background-color: {c['bg_primary']};
                color: {c['text_primary']};
            }}
            
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
                color: {accent_hex};
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
                background-color: {accent_hex};
                color: white;
            }}
            
            /* ===== Logo ===== */
            QLabel#logo {{
                color: {accent_hex};
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
            
            /* ===== Group Boxes ===== */
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {accent_hex};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: {accent_hex};
            }}
            
            /* ===== Progress Bar ===== */
            QProgressBar {{
                border: none;
                background-color: {c['bg_tertiary']};
                border-radius: 4px;
                text-align: center;
                height: 8px;
            }}
            
            QProgressBar::chunk {{
                background-color: {accent_hex};
                border-radius: 4px;
            }}
            
            /* ===== Buttons ===== */
            QPushButton {{
                background-color: {c['tool_button']};
                color: {c['tool_button_text']};
                border: 1px solid {c['border']};
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 13px;
            }}
            
            QPushButton:hover {{
                background-color: {c['bg_hover']};
                border-color: {accent_hex};
            }}
            
            QPushButton:pressed {{
                background-color: {c['bg_pressed']};
            }}
            
            QPushButton[class="primary"] {{
                background-color: {accent_hex};
                color: white;
                border: none;
            }}
            
            QPushButton[class="primary"]:hover {{
                background-color: {c['accent_hover']};
            }}
            
            QPushButton[class="danger"] {{
                background-color: #dc3545;
                color: white;
                border: none;
            }}
            
            QPushButton[class="danger"]:hover {{
                background-color: #c82333;
            }}
            
            QPushButton[class="success"] {{
                background-color: #28a745;
                color: white;
                border: none;
            }}
            
            QPushButton[class="success"]:hover {{
                background-color: #218838;
            }}
            
            /* ===== Line Edit ===== */
            QLineEdit {{
                background-color: {c['bg_secondary']};
                color: {c['text_primary']};
                border: 1px solid {c['border']};
                border-radius: 4px;
                padding: 6px 8px;
                font-size: 13px;
            }}
            
            QLineEdit:focus {{
                border: 2px solid {accent_hex};
            }}
            
            /* ===== Combo Box ===== */
            QComboBox {{
                background-color: {c['bg_secondary']};
                color: {c['text_primary']};
                border: 1px solid {c['border']};
                border-radius: 4px;
                padding: 5px;
                font-size: 13px;
                min-height: 24px;
            }}
            
            QComboBox:hover {{
                border-color: {accent_hex};
            }}
            
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 1px;
                border-left-color: {c['border']};
                border-left-style: solid;
            }}
            
            /* ===== Check Box ===== */
            QCheckBox {{
                color: {c['text_primary']};
                font-size: 13px;
                spacing: 8px;
            }}
            
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 1px solid {c['border']};
                border-radius: 3px;
                background-color: {c['bg_secondary']};
            }}
            
            QCheckBox::indicator:checked {{
                background-color: {accent_hex};
                border-color: {accent_hex};
                image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjMiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PHBvbHlsaW5lIHBvaW50cz0iMjAgNiA5IDE3IDQgMTIiLz48L3N2Zz4=);
            }}
            
            /* ===== Radio Button ===== */
            QRadioButton {{
                color: {c['text_primary']};
                font-size: 13px;
                spacing: 8px;
            }}
            
            QRadioButton::indicator {{
                width: 18px;
                height: 18px;
                border: 1px solid {c['border']};
                border-radius: 9px;
                background-color: {c['bg_secondary']};
            }}
            
            QRadioButton::indicator:checked {{
                background-color: {accent_hex};
                border-color: {accent_hex};
            }}
            
            /* ===== Slider ===== */
            QSlider::groove:horizontal {{
                border: 1px solid {c['border']};
                height: 6px;
                background: {c['bg_tertiary']};
                margin: 2px 0;
                border-radius: 3px;
            }}
            
            QSlider::handle:horizontal {{
                background: {accent_hex};
                border: 1px solid {c['border']};
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            
            QSlider::handle:horizontal:hover {{
                background: {c['accent_hover']};
                transform: scale(1.1);
            }}
            
            /* ===== List Widget ===== */
            QListWidget {{
                background-color: {c['bg_secondary']};
                border: 1px solid {c['border']};
                border-radius: 4px;
                padding: 4px;
                outline: none;
            }}
            
            QListWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {c['border_light']};
                border-radius: 3px;
            }}
            
            QListWidget::item:selected {{
                background-color: {accent_hex};
                color: white;
            }}
            
            QListWidget::item:hover:!selected {{
                background-color: {c['bg_hover']};
            }}
            
            /* ===== Tree Widget ===== */
            QTreeWidget {{
                background-color: {c['bg_secondary']};
                border: 1px solid {c['border']};
                border-radius: 4px;
                outline: none;
            }}
            
            QTreeWidget::item {{
                padding: 6px;
                border-bottom: 1px solid {c['border_light']};
            }}
            
            QTreeWidget::item:selected {{
                background-color: {accent_hex};
                color: white;
            }}
            
            /* ===== Tab Widget ===== */
            QTabWidget::pane {{
                border: 1px solid {c['border']};
                border-radius: 4px;
                background-color: {c['bg_secondary']};
            }}
            
            QTabBar::tab {{
                background-color: {c['bg_tertiary']};
                color: {c['text_secondary']};
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            
            QTabBar::tab:selected {{
                background-color: {c['bg_secondary']};
                color: {c['text_primary']};
                border-bottom: 2px solid {accent_hex};
            }}
            
            QTabBar::tab:hover:!selected {{
                background-color: {c['bg_hover']};
            }}
            
            /* ===== Scroll Bar ===== */
            QScrollBar:vertical {{
                border: none;
                background-color: {c['bg_tertiary']};
                width: 12px;
                border-radius: 6px;
            }}
            
            QScrollBar::handle:vertical {{
                background-color: {c['border']};
                min-height: 20px;
                border-radius: 6px;
            }}
            
            QScrollBar::handle:vertical:hover {{
                background-color: {accent_hex};
            }}
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                border: none;
                background: none;
            }}
            
            QScrollBar:horizontal {{
                border: none;
                background-color: {c['bg_tertiary']};
                height: 12px;
                border-radius: 6px;
            }}
            
            QScrollBar::handle:horizontal {{
                background-color: {c['border']};
                min-width: 20px;
                border-radius: 6px;
            }}
            
            QScrollBar::handle:horizontal:hover {{
                background-color: {accent_hex};
            }}
        """
    
    def get_nav_stylesheet(self) -> str:
        """Get navigation bar stylesheet."""
        c = self.current
        accent_hex = self.accent_color.name() if hasattr(self.accent_color, 'name') else str(self.accent_color)
        
        return f"""
            /* ===== Navigation ===== */
            QWidget#navigationBar {{
                background-color: {c['nav_bg']};
                border-bottom: 1px solid {c['border']};
            }}
            
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
                color: {accent_hex};
            }}
            
            QPushButton#navButton:checked {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 {c['accent_gradient_start']},
                                          stop:1 {c['accent_gradient_end']});
                color: white;
            }}
            
            QLabel#logo {{
                color: {accent_hex};
                font-size: 20px;
                font-weight: bold;
                padding: 0 20px;
            }}
            
            QFrame#separator {{
                color: {c['border']};
                font-size: 20px;
            }}
            
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
                background-color: {accent_hex};
                color: white;
            }}
        """
    
    def apply_to_app(self, app):
        """Apply theme to QApplication."""
        if self.is_dark:
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
            dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
            dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
            dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
            dark_palette.setColor(QPalette.ColorRole.Highlight, self.accent_color)
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
            light_palette.setColor(QPalette.ColorRole.Highlight, self.accent_color)
            light_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
            app.setPalette(light_palette)


# ==================== SETTINGS DIALOG ====================

class SettingsDialog(QDialog):
    """Modern settings dialog with live preview."""
    
    settings_applied = pyqtSignal()
    
    def __init__(self, parent, settings: AppSettings, theme_manager):
        super().__init__(parent)
        self.parent = parent
        self.settings = settings
        self.theme_manager = theme_manager
        self.modified_settings = AppSettings.from_dict(settings.to_dict())
        self.bindings: List[Setting] = []
        
        self.setWindowTitle("Settings")
        self.setMinimumSize(1000, 700)
        self.setModal(True)
        
        # Set window flags for modern look
            # Set window flags for modern look
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint)
        
        # Ensure theme_manager has required attributes
        if not hasattr(self.theme_manager, 'accent_color'):
            self.theme_manager.accent_color = QColor(76, 175, 80)
        
        self.setup_ui()
        self.setup_bindings()
        self.load_settings()
        self.setup_connections()
        
        # Apply initial theme
        self.apply_preview_theme()
    def setup_ui(self):
        """Setup the modern settings dialog UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # ========== HEADER ==========
        header = self.create_header()
        main_layout.addWidget(header)
        
        # ========== MAIN CONTENT ==========
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(1)
        
        # Left sidebar with categories
        sidebar = self.create_sidebar()
        content_layout.addWidget(sidebar)
        
        # Right content area
        self.content_stack = QStackedWidget()
        self.content_stack.setStyleSheet("""
            QStackedWidget {
                background-color: transparent;
                padding: 24px;
            }
        """)
        
        # Create pages
        self.pages = []
        self.pages.append(self.create_appearance_page())
        self.pages.append(self.create_general_page())
        self.pages.append(self.create_download_page())
        self.pages.append(self.create_stt_page())
        self.pages.append(self.create_tts_page())
        self.pages.append(self.create_shortcuts_page())
        self.pages.append(self.create_about_page())
        
        for page in self.pages:
            self.content_stack.addWidget(page)
        
        content_layout.addWidget(self.content_stack, 1)
        main_layout.addWidget(content, 1)
        
        # ========== FOOTER ==========
        footer = self.create_footer()
        main_layout.addWidget(footer)
    
    def create_header(self) -> QWidget:
        """Create modern header with search."""
        header = QWidget()
        header.setFixedHeight(70)
        
        # Get accent color safely from theme_manager
        if hasattr(self.theme_manager, 'accent_color'):
            accent_color = self.theme_manager.accent_color
            if hasattr(accent_color, 'name'):
                accent_hex = accent_color.name()
            else:
                accent_hex = str(accent_color)
        else:
            # Fallback to current theme accent
            accent_hex = '#4CAF50'
        
        header.setStyleSheet(f"""
            QWidget {{
                background-color: {accent_hex};
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }}
        """)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 0, 24, 0)
        
        title = QLabel("Settings")
        title.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Search with icon
        search_container = QWidget()
        search_container.setFixedWidth(280)
        search_container.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 0.15);
                border-radius: 20px;
            }
        """)
        
        search_layout = QHBoxLayout(search_container)
        search_layout.setContentsMargins(12, 6, 12, 6)
        
        search_icon = QLabel("🔍")
        search_icon.setStyleSheet("color: rgba(255, 255, 255, 0.8); font-size: 14px;")
        search_layout.addWidget(search_icon)
        
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search settings...")
        self.search_box.setStyleSheet("""
            QLineEdit {
                background: transparent;
                border: none;
                color: white;
                font-size: 13px;
                padding: 0;
            }
            QLineEdit::placeholder {
                color: rgba(255, 255, 255, 0.6);
            }
        """)
        search_layout.addWidget(self.search_box)
        
        layout.addWidget(search_container)
        
        return header
    
    def create_sidebar(self) -> QWidget:
        """Create modern category sidebar."""
        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 0.03);
                border-right: 1px solid rgba(0, 0, 0, 0.08);
            }
        """)
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 24, 12, 24)
        layout.setSpacing(4)
        
        categories = [
            ("🎨", "Appearance", 0),
            ("⚙️", "General", 1),
            ("📥", "Download", 2),
            ("🎤", "Speech-to-Text", 3),
            ("🔊", "Text-to-Speech", 4),
            ("⌨️", "Shortcuts", 5),
            ("ℹ️", "About", 6)
        ]
        
        self.category_buttons = []
        
        for icon, name, index in categories:
            btn = QToolButton()
            btn.setText(f"{icon}  {name}")
            btn.setCheckable(True)
            btn.setChecked(index == 0)
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            btn.setFixedHeight(44)
            btn.setProperty("category_index", index)
            btn.clicked.connect(lambda checked, idx=index: self.switch_category(idx))
            
            # Store reference
            self.category_buttons.append(btn)
            layout.addWidget(btn)
        
        layout.addStretch()
        
        # Theme preview at bottom
        preview_label = QLabel("Live Preview")
        preview_label.setStyleSheet("""
            font-weight: bold;
            color: #4CAF50;
            padding: 16px 8px 8px 8px;
            border-top: 1px solid rgba(0, 0, 0, 0.08);
        """)
        layout.addWidget(preview_label)
        
        self.theme_preview = ThemePreviewWidget(self.theme_manager)
        layout.addWidget(self.theme_preview)
        
        return sidebar
    
    def create_footer(self) -> QWidget:
        """Create modern footer with action buttons."""
        footer = QWidget()
        footer.setFixedHeight(70)
        footer.setStyleSheet("""
            QWidget {
                border-top: 1px solid rgba(0, 0, 0, 0.08);
                background-color: transparent;
            }
        """)
        
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(24, 0, 24, 0)
        
        # Left side - Reset button
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.setProperty("class", "danger")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        layout.addWidget(self.reset_btn)
        
        layout.addStretch()
        
        # Right side - Action buttons
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setProperty("class", "primary")
        self.apply_btn.clicked.connect(self.apply_settings)
        self.apply_btn.setFixedWidth(100)
        layout.addWidget(self.apply_btn)
        
        self.ok_btn = QPushButton("OK")
        self.ok_btn.setProperty("class", "success")
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setFixedWidth(100)
        layout.addWidget(self.ok_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setFixedWidth(100)
        layout.addWidget(self.cancel_btn)
        
        return footer
    
    def setup_bindings(self):
        """Setup declarative bindings between widgets and settings."""
        # Appearance page
        self.add_binding('theme', 
                        lambda: self.theme_combo.currentIndex(),
                        lambda v: self.theme_combo.setCurrentIndex(v if v is not None else 0),
                        converter=lambda x: {'dark': 0, 'light': 1, 'system': 2}.get(x, 0),
                        reverse_converter=lambda x: ['dark', 'light', 'system'][x] if 0 <= x <= 2 else 'dark')
        
        self.add_binding('accent_color',
                        lambda: self.accent_color_btn.property("color") or "#4CAF50",
                        lambda v: self.update_accent_color_display(v if v else "#4CAF50"))
        
        self.add_binding('font_family',
                        lambda: self.font_label.text().split()[:-1] if self.font_label.text() else ["Segoe UI"],
                        lambda v: None,  # Handled separately
                        converter=lambda x: x if x else "Segoe UI")
        
        self.add_binding('font_size',
                        lambda: int(self.font_label.text().split()[-1].replace('pt', '')) if self.font_label.text() else 10,
                        lambda v: self.font_label.setText(f"{self.modified_settings.font_family} {v}pt"))
        
        self.add_binding('layout_style',
                        lambda: self.layout_combo.currentIndex(),
                        lambda v: self.layout_combo.setCurrentIndex(v if v is not None else 1),
                        converter=lambda x: {'compact': 0, 'comfortable': 1, 'spacious': 2}.get(x, 1),
                        reverse_converter=lambda x: ['compact', 'comfortable', 'spacious'][x] if 0 <= x <= 2 else 'comfortable')
        
        self.add_binding('show_toolbar', lambda: self.toolbar_check.isChecked(), lambda v: self.toolbar_check.setChecked(v if v is not None else True))
        self.add_binding('show_statusbar', lambda: self.statusbar_check.isChecked(), lambda v: self.statusbar_check.setChecked(v if v is not None else True))
        self.add_binding('show_waveform', lambda: self.waveform_check.isChecked(), lambda v: self.waveform_check.setChecked(v if v is not None else True))
        
        # Download page
        self.add_binding('download_location', 
                        lambda: self.download_location.text() or os.path.expanduser("~/Downloads"), 
                        lambda v: self.download_location.setText(v if v else os.path.expanduser("~/Downloads")))
        
        self.add_binding('default_format',
                        lambda: self.default_format.currentIndex(),
                        lambda v: self.default_format.setCurrentIndex(v if v is not None else 0),
                        converter=lambda x: {'audio': 0, 'video': 1}.get(x, 0),
                        reverse_converter=lambda x: ['audio', 'video'][x] if 0 <= x <= 1 else 'audio')
        
        self.add_binding('audio_format',
                        lambda: self.audio_format.currentText() or "mp3",
                        lambda v: self.audio_format.setCurrentText(v if v else "mp3"))
        
        self.add_binding('audio_quality',
                        lambda: self.audio_quality.currentText().split()[0] if self.audio_quality.currentText() else "192",
                        lambda v: self.audio_quality.setCurrentText(f"{v} kbps" if v else "192 kbps"))
        
        self.add_binding('video_quality',
                        lambda: self.video_quality.currentText().lower() if self.video_quality.currentText() else "best",
                        lambda v: self.video_quality.setCurrentText(v.capitalize() if v else "Best"))
        
        self.add_binding('max_concurrent_downloads', 
                        lambda: self.max_concurrent.value(), 
                        lambda v: self.max_concurrent.setValue(v if v is not None else 3))
        
        self.add_binding('auto_open_folder', lambda: self.auto_open.isChecked(), lambda v: self.auto_open.setChecked(v if v is not None else True))
        self.add_binding('auto_load_to_app', lambda: self.auto_load.isChecked(), lambda v: self.auto_load.setChecked(v if v is not None else True))
        
        # STT page
        self.add_binding('stt_default_engine',
                        lambda: self.stt_engine.currentIndex(),
                        lambda v: self.stt_engine.setCurrentIndex(v if v is not None else 0),
                        converter=lambda x: {'faster-whisper': 0, 'google': 1, 'vosk': 2}.get(x, 0),
                        reverse_converter=lambda x: ['faster-whisper', 'google', 'vosk'][x] if 0 <= x <= 2 else 'faster-whisper')
        
        self.add_binding('stt_model_size',
                        lambda: self.stt_model.currentText() or "base",
                        lambda v: self.stt_model.setCurrentText(v if v else "base"))
        
        self.add_binding('stt_vad_filter', lambda: self.vad_check.isChecked(), lambda v: self.vad_check.setChecked(v if v is not None else True))
        
        # TTS page
        self.add_binding('tts_speed', 
                        lambda: self.tts_speed_slider.value(), 
                        lambda v: self.tts_speed_slider.setValue(v if v is not None else 100))
        
        self.add_binding('tts_volume', 
                        lambda: self.tts_volume_slider.value(), 
                        lambda v: self.tts_volume_slider.setValue(v if v is not None else 100))
        
        self.add_binding('tts_crossfade', 
                        lambda: self.tts_crossfade.value(), 
                        lambda v: self.tts_crossfade.setValue(v if v is not None else 50))
        
        self.add_binding('tts_normalize', lambda: self.tts_normalize.isChecked(), lambda v: self.tts_normalize.setChecked(v if v is not None else True))
    def add_binding(self, key: str, getter: Callable, setter: Callable, 
                    converter: Callable = None, reverse_converter: Callable = None):
        """Add a setting binding."""
        self.bindings.append(Setting(key, getter, setter, converter, reverse_converter))
    
    def create_appearance_page(self) -> QWidget:
        """Create appearance settings page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(24)
        
        # Theme section
        theme_group = QGroupBox("Theme")
        theme_layout = QFormLayout()
        theme_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        theme_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "System"])
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)
        theme_layout.addRow("Theme:", self.theme_combo)
        
        # Accent color
        color_layout = QHBoxLayout()
        self.accent_color_btn = QPushButton()
        self.accent_color_btn.setFixedSize(60, 36)
        self.accent_color_btn.clicked.connect(self.choose_accent_color)
        color_layout.addWidget(self.accent_color_btn)
        
        self.accent_color_label = QLabel("Click to change")
        self.accent_color_label.setStyleSheet("color: #888;")
        color_layout.addWidget(self.accent_color_label)
        color_layout.addStretch()
        theme_layout.addRow("Accent:", color_layout)
        
        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)
        
        # Font section
        font_group = QGroupBox("Font")
        font_layout = QHBoxLayout()
        
        self.font_label = QLabel("Segoe UI 10pt")
        self.font_label.setStyleSheet("padding: 8px; background: rgba(0,0,0,0.03); border-radius: 4px;")
        font_layout.addWidget(self.font_label)
        
        self.font_btn = QPushButton("Change...")
        self.font_btn.clicked.connect(self.choose_font)
        font_layout.addWidget(self.font_btn)
        
        font_group.setLayout(font_layout)
        layout.addWidget(font_group)
        
        # Layout section
        layout_group = QGroupBox("Layout")
        layout_form = QFormLayout()
        
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["Compact", "Comfortable", "Spacious"])
        self.layout_combo.currentTextChanged.connect(self.on_layout_changed)
        layout_form.addRow("Style:", self.layout_combo)
        
        self.toolbar_check = QCheckBox("Show Toolbar")
        layout_form.addRow("", self.toolbar_check)
        
        self.statusbar_check = QCheckBox("Show Status Bar")
        layout_form.addRow("", self.statusbar_check)
        
        self.waveform_check = QCheckBox("Show Waveform")
        layout_form.addRow("", self.waveform_check)
        
        layout_group.setLayout(layout_form)
        layout.addWidget(layout_group)
        
        layout.addStretch()
        return page
    
    def create_general_page(self) -> QWidget:
        """Create general settings page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(24)
        
        # Startup section
        startup_group = QGroupBox("Startup")
        startup_layout = QVBoxLayout()
        
        self.last_tab_check = QCheckBox("Restore last active tab")
        startup_layout.addWidget(self.last_tab_check)
        
        self.check_updates_check = QCheckBox("Check for updates on startup")
        startup_layout.addWidget(self.check_updates_check)
        
        startup_group.setLayout(startup_layout)
        layout.addWidget(startup_group)
        
        # Behavior section
        behavior_group = QGroupBox("Behavior")
        behavior_layout = QVBoxLayout()
        
        self.auto_save_check = QCheckBox("Auto-save sessions")
        behavior_layout.addWidget(self.auto_save_check)
        
        self.confirm_exit_check = QCheckBox("Confirm before exit")
        behavior_layout.addWidget(self.confirm_exit_check)
        
        behavior_group.setLayout(behavior_layout)
        layout.addWidget(behavior_group)
        
        # Recent files
        recent_group = QGroupBox("Recent Files")
        recent_layout = QFormLayout()
        
        self.recent_count = QSpinBox()
        self.recent_count.setRange(5, 50)
        self.recent_count.setValue(20)
        recent_layout.addRow("Maximum:", self.recent_count)
        
        self.clear_recent_btn = QPushButton("Clear History")
        self.clear_recent_btn.clicked.connect(self.clear_recent_files)
        recent_layout.addRow("", self.clear_recent_btn)
        
        recent_group.setLayout(recent_layout)
        layout.addWidget(recent_group)
        
        # Language
        lang_group = QGroupBox("Language")
        lang_layout = QFormLayout()
        
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "Khmer", "Chinese", "Japanese"])
        lang_layout.addRow("Interface:", self.language_combo)
        
        lang_group.setLayout(lang_layout)
        layout.addWidget(lang_group)
        
        layout.addStretch()
        return page
    
    def create_download_page(self) -> QWidget:
        """Create download settings page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(24)
        
        # Location
        location_group = QGroupBox("Download Location")
        location_layout = QHBoxLayout()
        
        self.download_location = QLineEdit()
        location_layout.addWidget(self.download_location)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_download_location)
        browse_btn.setFixedWidth(100)
        location_layout.addWidget(browse_btn)
        
        location_group.setLayout(location_layout)
        layout.addWidget(location_group)
        
        # Format
        format_group = QGroupBox("Default Format")
        format_layout = QFormLayout()
        
        self.default_format = QComboBox()
        self.default_format.addItems(["Audio", "Video"])
        format_layout.addRow("Type:", self.default_format)
        
        self.audio_format = QComboBox()
        self.audio_format.addItems(["mp3", "m4a", "wav", "flac"])
        format_layout.addRow("Audio Format:", self.audio_format)
        
        self.audio_quality = QComboBox()
        self.audio_quality.addItems(["128 kbps", "192 kbps", "256 kbps", "320 kbps"])
        format_layout.addRow("Audio Quality:", self.audio_quality)
        
        self.video_quality = QComboBox()
        self.video_quality.addItems(["Best", "1080p", "720p", "480p"])
        format_layout.addRow("Video Quality:", self.video_quality)
        
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)
        
        # Advanced
        advanced_group = QGroupBox("Advanced")
        advanced_layout = QFormLayout()
        
        self.max_concurrent = QSpinBox()
        self.max_concurrent.setRange(1, 10)
        advanced_layout.addRow("Max Concurrent:", self.max_concurrent)
        
        self.auto_open = QCheckBox("Auto-open folder after download")
        advanced_layout.addRow("", self.auto_open)
        
        self.auto_load = QCheckBox("Auto-load to current app")
        advanced_layout.addRow("", self.auto_load)
        
        self.subtitles_check = QCheckBox("Download subtitles by default")
        advanced_layout.addRow("", self.subtitles_check)
        
        self.thumbnail_check = QCheckBox("Download thumbnail by default")
        advanced_layout.addRow("", self.thumbnail_check)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        layout.addStretch()
        return page
    
    def create_stt_page(self) -> QWidget:
        """Create Speech-to-Text settings page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(24)
        
        # Engine
        engine_group = QGroupBox("Engine")
        engine_layout = QFormLayout()
        
        self.stt_engine = QComboBox()
        self.stt_engine.addItems(["faster-whisper", "Google Cloud", "Vosk"])
        engine_layout.addRow("Default:", self.stt_engine)
        
        engine_group.setLayout(engine_layout)
        layout.addWidget(engine_group)
        
        # Model
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout()
        
        self.stt_model = QComboBox()
        self.stt_model.addItems(["tiny", "base", "small", "medium", "large"])
        model_layout.addRow("Size:", self.stt_model)
        
        self.stt_language = QComboBox()
        self.stt_language.addItems(["Auto", "English", "Khmer", "Chinese", "Japanese"])
        model_layout.addRow("Language:", self.stt_language)
        
        self.vad_check = QCheckBox("Enable VAD filter")
        model_layout.addRow("", self.vad_check)
        
        self.enhanced_check = QCheckBox("Use enhanced model")
        model_layout.addRow("", self.enhanced_check)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Output
        output_group = QGroupBox("Output")
        output_layout = QFormLayout()
        
        self.auto_save_stt = QCheckBox("Auto-save transcriptions")
        output_layout.addRow("", self.auto_save_stt)
        
        self.save_format = QComboBox()
        self.save_format.addItems(["TXT", "SRT", "VTT", "JSON"])
        output_layout.addRow("Format:", self.save_format)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        layout.addStretch()
        return page
    
    def create_tts_page(self) -> QWidget:
        """Create Text-to-Speech settings page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(24)
        
        # Voice
        voice_group = QGroupBox("Voice Settings")
        voice_layout = QFormLayout()
        
        self.tts_voice = QComboBox()
        self.tts_voice.addItems([
            "km-KH-SreymomNeural (Khmer Female)",
            "km-KH-ThearithNeural (Khmer Male)",
            "en-US-JennyNeural (English Female)",
            "en-US-GuyNeural (English Male)",
            "zh-CN-XiaoxiaoNeural (Chinese Female)",
            "zh-CN-YunxiNeural (Chinese Male)"
        ])
        voice_layout.addRow("Voice:", self.tts_voice)
        
        # Speed
        speed_layout = QHBoxLayout()
        self.tts_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.tts_speed_slider.setRange(50, 200)
        self.tts_speed_slider.setValue(100)
        self.tts_speed_slider.setTickInterval(25)
        self.tts_speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        speed_layout.addWidget(self.tts_speed_slider)
        
        self.tts_speed_label = QLabel("100%")
        self.tts_speed_label.setFixedWidth(50)
        speed_layout.addWidget(self.tts_speed_label)
        
        self.tts_speed_slider.valueChanged.connect(
            lambda v: self.tts_speed_label.setText(f"{v}%")
        )
        voice_layout.addRow("Speed:", speed_layout)
        
        # Volume
        volume_layout = QHBoxLayout()
        self.tts_volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.tts_volume_slider.setRange(0, 200)
        self.tts_volume_slider.setValue(100)
        self.tts_volume_slider.setTickInterval(25)
        self.tts_volume_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        volume_layout.addWidget(self.tts_volume_slider)
        
        self.tts_volume_label = QLabel("100%")
        self.tts_volume_label.setFixedWidth(50)
        volume_layout.addWidget(self.tts_volume_label)
        
        self.tts_volume_slider.valueChanged.connect(
            lambda v: self.tts_volume_label.setText(f"{v}%")
        )
        voice_layout.addRow("Volume:", volume_layout)
        
        voice_group.setLayout(voice_layout)
        layout.addWidget(voice_group)
        
        # Audio processing
        audio_group = QGroupBox("Audio Processing")
        audio_layout = QFormLayout()
        
        self.tts_crossfade = QSpinBox()
        self.tts_crossfade.setRange(0, 500)
        self.tts_crossfade.setSuffix(" ms")
        audio_layout.addRow("Crossfade:", self.tts_crossfade)
        
        self.tts_normalize = QCheckBox("Normalize audio levels")
        audio_layout.addRow("", self.tts_normalize)
        
        self.tts_overlap = QCheckBox("Prevent overlap")
        audio_layout.addRow("", self.tts_overlap)
        
        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)
        
        layout.addStretch()
        return page
    
    def create_shortcuts_page(self) -> QWidget:
        """Create keyboard shortcuts page with QKeySequenceEdit."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(16)
        
        # Shortcuts tree
        self.shortcuts_tree = QTreeWidget()
        self.shortcuts_tree.setHeaderLabels(["Action", "Shortcut"])
        self.shortcuts_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.shortcuts_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.shortcuts_tree.setAlternatingRowColors(True)
        self.shortcuts_tree.setIndentation(0)
        self.shortcuts_tree.setMinimumHeight(300)
        
        layout.addWidget(self.shortcuts_tree)
        
        # Edit section with QKeySequenceEdit
        edit_group = QGroupBox("Edit Shortcut")
        edit_layout = QHBoxLayout()
        
        self.shortcut_edit = QKeySequenceEdit()
        self.shortcut_edit.setMaximumHeight(36)
        edit_layout.addWidget(self.shortcut_edit, 2)
        
        self.set_shortcut_btn = QPushButton("Set")
        self.set_shortcut_btn.clicked.connect(self.set_shortcut)
        self.set_shortcut_btn.setEnabled(False)
        self.set_shortcut_btn.setFixedWidth(80)
        edit_layout.addWidget(self.set_shortcut_btn)
        
        self.reset_shortcut_btn = QPushButton("Reset")
        self.reset_shortcut_btn.clicked.connect(self.reset_shortcut)
        self.reset_shortcut_btn.setEnabled(False)
        self.reset_shortcut_btn.setFixedWidth(80)
        edit_layout.addWidget(self.reset_shortcut_btn)
        
        edit_group.setLayout(edit_layout)
        layout.addWidget(edit_group)
        
        # Reset all button
        reset_all_btn = QPushButton("Reset All to Defaults")
        reset_all_btn.setProperty("class", "danger")
        reset_all_btn.clicked.connect(self.reset_all_shortcuts)
        layout.addWidget(reset_all_btn)
        
        # Connect selection
        self.shortcuts_tree.itemClicked.connect(self.on_shortcut_selected)
        
        return page
    
    def create_about_page(self) -> QWidget:
        """Create about page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(24)
        
        # App info
        info_group = QGroupBox("AI Studio Pro")
        info_layout = QVBoxLayout()
        info_layout.setSpacing(16)
        
        title = QLabel("🎙️ AI STUDIO PRO")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #4CAF50;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(title)
        
        version = QLabel("Version 2.0.0")
        version.setStyleSheet("font-size: 14px; color: #888;")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_layout.addWidget(version)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: #ddd; max-height: 1px;")
        info_layout.addWidget(separator)
        
        features = QLabel(
            "A comprehensive speech and download suite featuring:\n\n"
            "• Speech-to-Text with multiple engines\n"
            "• Text-to-Speech Merger\n"
            "• Media Downloader\n"
            "• Audio Overlap Detection\n"
            "• Customizable Themes\n"
            "• Keyboard Shortcuts"
        )
        features.setWordWrap(True)
        features.setStyleSheet("font-size: 13px; line-height: 1.6; padding: 8px;")
        info_layout.addWidget(features)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # System info
        sys_group = QGroupBox("System Information")
        sys_layout = QFormLayout()
        
        sys_layout.addRow("Python:", QLabel(sys.version.split()[0]))
        sys_layout.addRow("Platform:", QLabel(sys.platform))
        sys_layout.addRow("Theme:", QLabel(self.settings.theme.capitalize()))
        sys_layout.addRow("Font:", QLabel(f"{self.settings.font_family} {self.settings.font_size}pt"))
        
        sys_group.setLayout(sys_layout)
        layout.addWidget(sys_group)
        
        # Credits
        credits_group = QGroupBox("Credits")
        credits_layout = QVBoxLayout()
        credits_layout.addWidget(QLabel("Developed with ❤️ using PyQt6"))
        credits_layout.addWidget(QLabel("© 2024 AI Studio"))
        credits_group.setLayout(credits_layout)
        layout.addWidget(credits_group)
        
        layout.addStretch()
        return page
    
    def setup_connections(self):
        """Setup signal connections."""
        self.search_box.textChanged.connect(self.filter_settings)
    
    def switch_category(self, index):
        """Switch between category pages with animation."""
        if 0 <= index < len(self.pages):
            self.content_stack.setCurrentIndex(index)
            # Update button states
            for i, btn in enumerate(self.category_buttons):
                btn.setChecked(i == index)
    
    def filter_settings(self, text):
        """Filter settings based on search text with highlighting."""
        text = text.lower()
        
        if not text:
            # Show all pages
            for i, page in enumerate(self.pages):
                self.content_stack.widget(i).setVisible(True)
            return
        
        # Find matching pages
        for i, page in enumerate(self.pages):
            visible = False
            # Search in all child widgets
            for child in page.findChildren(QWidget):
                if isinstance(child, (QLabel, QGroupBox, QCheckBox, QPushButton)):
                    widget_text = getattr(child, 'text', lambda: '')()
                    if text in widget_text.lower():
                        visible = True
                        break
            
            self.content_stack.widget(i).setVisible(visible)
    
    def load_settings(self):
        """Load settings into UI using bindings."""
        for binding in self.bindings:
            try:
                binding.load(self.modified_settings)
            except Exception as e:
                print(f"⚠️ Error loading binding {binding.key}: {e}")
        
        # Load shortcuts
        self.refresh_shortcuts_tree()
        
        # Update accent color display
        if hasattr(self.modified_settings, 'accent_color') and self.modified_settings.accent_color:
            self.update_accent_color_display(self.modified_settings.accent_color)
        else:
            self.update_accent_color_display("#4CAF50")
    def save_settings(self):
        """Save settings from UI to modified_settings using bindings."""
        for binding in self.bindings:
            binding.save(self.modified_settings)
        
        # Save shortcuts
        self.save_shortcuts()
    
    def refresh_shortcuts_tree(self):
        """Refresh the shortcuts tree display."""
        self.shortcuts_tree.clear()
        
        shortcut_names = {
            'new_download': 'New Download',
            'start_transcribe': 'Start Transcription',
            'open_tts': 'Open TTS',
            'settings': 'Open Settings',
            'quit': 'Quit Application',
            'switch_theme': 'Switch Theme',
            'search': 'Search',
            'export': 'Export',
            'save': 'Save',
            'open_file': 'Open File'
        }
        
        self.shortcut_items = {}
        
        for key, name in shortcut_names.items():
            shortcut = self.modified_settings.shortcuts.get(key, "")
            item = QTreeWidgetItem([name, shortcut])
            item.setData(0, Qt.ItemDataRole.UserRole, key)
            self.shortcuts_tree.addTopLevelItem(item)
            self.shortcut_items[key] = item
    
    def save_shortcuts(self):
        """Save shortcuts from tree to settings."""
        for i in range(self.shortcuts_tree.topLevelItemCount()):
            item = self.shortcuts_tree.topLevelItem(i)
            key = item.data(0, Qt.ItemDataRole.UserRole)
            shortcut = item.text(1)
            if key and shortcut:
                self.modified_settings.shortcuts[key] = shortcut
    
    def on_shortcut_selected(self, item):
        """Handle shortcut selection."""
        self.current_shortcut_key = item.data(0, Qt.ItemDataRole.UserRole)
        current_shortcut = item.text(1)
        
        # Clear and set in QKeySequenceEdit
        self.shortcut_edit.clear()
        if current_shortcut:
            self.shortcut_edit.setKeySequence(QKeySequence(current_shortcut))
        
        self.set_shortcut_btn.setEnabled(True)
        self.reset_shortcut_btn.setEnabled(True)
    
    def set_shortcut(self):
        """Set custom shortcut for selected action."""
        if hasattr(self, 'current_shortcut_key'):
            new_shortcut = self.shortcut_edit.keySequence().toString()
            if new_shortcut:
                # Update tree
                for i in range(self.shortcuts_tree.topLevelItemCount()):
                    item = self.shortcuts_tree.topLevelItem(i)
                    if item.data(0, Qt.ItemDataRole.UserRole) == self.current_shortcut_key:
                        item.setText(1, new_shortcut)
                        break
                
                # Update settings
                self.modified_settings.shortcuts[self.current_shortcut_key] = new_shortcut
    
    def reset_shortcut(self):
        """Reset selected shortcut to default."""
        if hasattr(self, 'current_shortcut_key'):
            defaults = {
                'new_download': 'Ctrl+D',
                'start_transcribe': 'Ctrl+R',
                'open_tts': 'Ctrl+T',
                'settings': 'Ctrl+,',
                'quit': 'Ctrl+Q',
                'switch_theme': 'Ctrl+Shift+T',
                'search': 'Ctrl+F',
                'export': 'Ctrl+E',
                'save': 'Ctrl+S',
                'open_file': 'Ctrl+O'
            }
            
            if self.current_shortcut_key in defaults:
                default_shortcut = defaults[self.current_shortcut_key]
                
                # Update tree
                for i in range(self.shortcuts_tree.topLevelItemCount()):
                    item = self.shortcuts_tree.topLevelItem(i)
                    if item.data(0, Qt.ItemDataRole.UserRole) == self.current_shortcut_key:
                        item.setText(1, default_shortcut)
                        break
                
                # Update settings
                self.modified_settings.shortcuts[self.current_shortcut_key] = default_shortcut
                self.shortcut_edit.setKeySequence(QKeySequence(default_shortcut))
    
    def reset_all_shortcuts(self):
        """Reset all shortcuts to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset Shortcuts",
            "Reset all keyboard shortcuts to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            defaults = {
                'new_download': 'Ctrl+D',
                'start_transcribe': 'Ctrl+R',
                'open_tts': 'Ctrl+T',
                'settings': 'Ctrl+,',
                'quit': 'Ctrl+Q',
                'switch_theme': 'Ctrl+Shift+T',
                'search': 'Ctrl+F',
                'export': 'Ctrl+E',
                'save': 'Ctrl+S',
                'open_file': 'Ctrl+O'
            }
            
            self.modified_settings.shortcuts.update(defaults)
            self.refresh_shortcuts_tree()
    
    def on_layout_changed(self, layout_text):
        """Handle layout change with live preview (simulated)."""
        layout_map = {"Compact": "compact", "Comfortable": "comfortable", "Spacious": "spacious"}
        self.modified_settings.layout_style = layout_map.get(layout_text, "comfortable")
    
    def choose_font(self):
        """Open font dialog with preview."""
        font, ok = QFontDialog.getFont(
            QFont(self.modified_settings.font_family, self.modified_settings.font_size),
            self,
            "Choose Font"
        )
        if ok:
            self.modified_settings.font_family = font.family()
            self.modified_settings.font_size = font.pointSize()
            self.font_label.setText(f"{font.family()} {font.pointSize()}pt")
            self.apply_preview_theme()
    
    def browse_download_location(self):
        """Browse for download location."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Download Folder",
            self.download_location.text()
        )
        if folder:
            self.download_location.setText(folder)
    
    def clear_recent_files(self):
        """Clear recent files list."""
        reply = QMessageBox.question(
            self,
            "Clear Recent Files",
            "Are you sure you want to clear recent files history?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.modified_settings.recent_files = []
            QMessageBox.information(self, "Success", "Recent files cleared.")
    
    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Reset all settings to default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.modified_settings = AppSettings()
            self.load_settings()
            self.apply_preview_theme()
    
    def apply_preview_theme(self):
        """Apply current theme to preview."""
        # Update theme manager with modified settings
        if hasattr(self.theme_manager, 'settings'):
            self.theme_manager.settings = self.modified_settings
        
        # Update accent color
        if hasattr(self.modified_settings, 'accent_color'):
            accent_color = QColor(self.modified_settings.accent_color)
            if hasattr(self.theme_manager, 'accent_color'):
                self.theme_manager.accent_color = accent_color
        
        # Update preview widget
        if hasattr(self, 'theme_preview'):
            self.theme_preview.update()
        
        # Update header color
        header = self.findChild(QWidget, "")
        if header and hasattr(self.theme_manager, 'accent_color'):
            accent_hex = self.theme_manager.accent_color.name()
            header.setStyleSheet(f"""
                QWidget {{
                    background-color: {accent_hex};
                    border-top-left-radius: 8px;
                    border-top-right-radius: 8px;
                }}
            """)
    
    def on_theme_changed(self, theme_text):
        """Handle theme change with live preview."""
        theme_map = {"Dark": "dark", "Light": "light", "System": "system"}
        self.modified_settings.theme = theme_map.get(theme_text, "dark")
        self.apply_preview_theme()
    
    def choose_accent_color(self):
        """Open color picker for accent color with live preview."""
        color = QColorDialog.getColor(
            QColor(self.modified_settings.accent_color),
            self,
            "Choose Accent Color",
            QColorDialog.ColorDialogOption.ShowAlphaChannel
        )
        if color.isValid():
            self.modified_settings.accent_color = color.name()
            self.update_accent_color_display(color.name())
            self.apply_preview_theme()
    
    def update_accent_color_display(self, color: str):
        """Update accent color button display."""
        if not color:
            color = "#4CAF50"
        
        self.accent_color_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                border: 2px solid #ffffff;
                border-radius: 18px;
            }}
            QPushButton:hover {{
                border: 3px solid #ffffff;
            }}
        """)
        self.accent_color_btn.setProperty("color", color)
        self.accent_color_label.setText(color)
    def apply_settings(self):
        """Apply settings without closing dialog."""
        self.save_settings()
        
        # Get the main window safely
        main_window = None
        if hasattr(self, 'parent') and self.parent is not None:
            main_window = self.parent
        elif hasattr(self, 'parent()') and self.parent() is not None:
            main_window = self.parent()
        
        # Update main window if it has apply_settings method
        if main_window and hasattr(main_window, 'apply_settings'):
            main_window.apply_settings(self.modified_settings)
        elif main_window and hasattr(main_window, 'parent') and hasattr(main_window.parent(), 'apply_settings'):
            main_window.parent().apply_settings(self.modified_settings)
        
        # Update theme manager
        if hasattr(self.theme_manager, 'settings'):
            self.theme_manager.settings = self.modified_settings
        if hasattr(self.modified_settings, 'accent_color') and hasattr(self.theme_manager, 'accent_color'):
            self.theme_manager.accent_color = QColor(self.modified_settings.accent_color)
        
        # Emit signal
        self.settings_applied.emit()
        
        QMessageBox.information(self, "Settings", "Settings applied successfully!")
    def accept(self):
        """Accept and save settings."""
        self.save_settings()
        
        # Update main window - use self.parent not self.parent()
        if hasattr(self.parent, 'apply_settings'):
            self.parent.apply_settings(self.modified_settings)
        elif hasattr(self.parent, 'parent') and hasattr(self.parent.parent(), 'apply_settings'):
            self.parent.parent().apply_settings(self.modified_settings)
        
        # Update original settings
        for key, value in asdict(self.modified_settings).items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        
        super().accept()
    def reject(self):
        """Reject and discard changes."""
        # Optional: Ask for confirmation if there are unsaved changes
        super().reject()


# ==================== SHORTCUT MANAGER ====================

class ShortcutManager:
    """Manages keyboard shortcuts across the application."""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.shortcuts = {}
        self.actions = {}
    
    def register_shortcut(self, name: str, callback, parent_widget, default_key: str = None):
        """Register a keyboard shortcut."""
        key = self.settings.shortcuts.get(name, default_key)
        if key:
            shortcut = QShortcut(QKeySequence(key), parent_widget)
            shortcut.activated.connect(callback)
            self.shortcuts[name] = shortcut
            self.actions[name] = callback
            return shortcut
        return None
    
    def update_shortcut(self, name: str, new_key: str):
        """Update an existing shortcut."""
        if name in self.shortcuts:
            self.shortcuts[name].setKey(QKeySequence(new_key))
            self.settings.shortcuts[name] = new_key
    
    def reset_to_defaults(self):
        """Reset all shortcuts to defaults."""
        defaults = {
            'new_download': 'Ctrl+D',
            'start_transcribe': 'Ctrl+R',
            'open_tts': 'Ctrl+T',
            'settings': 'Ctrl+,',
            'quit': 'Ctrl+Q',
            'switch_theme': 'Ctrl+Shift+T',
            'search': 'Ctrl+F',
            'export': 'Ctrl+E',
            'save': 'Ctrl+S',
            'open_file': 'Ctrl+O'
        }
        self.settings.shortcuts.update(defaults)


# ==================== LAYOUT MANAGER ====================

class LayoutManager:
    """Manages application layout and panel sizes."""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
    
    def get_spacing(self) -> int:
        """Get spacing based on layout style."""
        style_map = {
            'compact': 4,
            'comfortable': 10,
            'spacious': 16
        }
        return style_map.get(self.settings.layout_style, 10)
    
    def get_margin(self) -> int:
        """Get margin based on layout style."""
        style_map = {
            'compact': 5,
            'comfortable': 10,
            'spacious': 20
        }
        return style_map.get(self.settings.layout_style, 10)
    
    def get_font_size_modifier(self) -> float:
        """Get font size modifier based on layout style."""
        style_map = {
            'compact': 0.9,
            'comfortable': 1.0,
            'spacious': 1.2
        }
        return style_map.get(self.settings.layout_style, 1.0)
    
    def save_panel_size(self, panel_name: str, size: int):
        """Save panel size to settings."""
        self.settings.panel_sizes[panel_name] = size
    
    def get_panel_size(self, panel_name: str, default: int = 300) -> int:
        """Get panel size from settings."""
        return self.settings.panel_sizes.get(panel_name, default)


# ==================== SETTINGS MANAGER ====================

class SettingsManager:
    """Manages loading/saving settings to disk."""
    
    def __init__(self):
        self.config_dir = Path.home() / '.aistudio'
        self.config_file = self.config_dir / 'settings.json'
        self.settings = AppSettings()
        
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(exist_ok=True)
    
    def load(self) -> AppSettings:
        """Load settings from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.settings = AppSettings.from_dict(data)
                    print(f"✅ Settings loaded from {self.config_file}")
            except Exception as e:
                print(f"⚠️ Failed to load settings: {e}")
                self.settings = AppSettings()
        else:
            print(f"ℹ️ No settings file found, using defaults")
        
        return self.settings
    
    def save(self, settings: AppSettings):
        """Save settings to file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(settings.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"✅ Settings saved to {self.config_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to save settings: {e}")
            return False
    
    def export_settings(self, file_path: str):
        """Export settings to a custom location."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings.to_dict(), f, indent=2, ensure_ascii=False)
            return True
        except:
            return False
    
    def import_settings(self, file_path: str) -> bool:
        """Import settings from a custom location."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.settings = AppSettings.from_dict(data)
            return True
        except:
            return False


# ==================== EXPORTS ====================

__all__ = [
    'AppSettings',
    'ThemeManager',
    'UnifiedThemeManager',
    'ShortcutManager',
    'LayoutManager',
    'SettingsDialog',
    'SettingsManager',
    'Theme',
    'LayoutStyle',
    'ThemePreviewWidget'
]