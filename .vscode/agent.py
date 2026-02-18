"""
agent.py - AI Assistant Bot using DeepSeek API
- Chat with AI assistant using DeepSeek
- Clean modern UI with message bubbles
- Conversation history management
- Copy messages, clear chat, export conversations
- Markdown support for formatted responses
- Streaming responses for real-time feedback
- Integration with AI Studio Pro main app
"""

import os
import json
import time
import hashlib
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import markdown
from dataclasses import dataclass

# PyQt6 imports
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QScrollArea, QFrame, QApplication, QMessageBox,
    QSizePolicy, QSplitter, QGroupBox, QLineEdit, QCheckBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QFileDialog, QMenu,
    QInputDialog, QProgressBar ,QDialog ,QFormLayout
)
from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QThread, QSize, QPoint, QPropertyAnimation,
    QEasingCurve, QParallelAnimationGroup, QByteArray
)
from PyQt6.QtGui import (
    QFont, QColor, QPixmap, QIcon, QTextCursor, QTextDocument,
    QTextBlockFormat, QTextCharFormat, QTextTableFormat, QTextImageFormat,
    QAction, QPalette, QBrush, QLinearGradient
)

# Try to import DeepSeek API
try:
    from openai import OpenAI
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False

# Try to import markdown for rendering
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False


# ==================== CONFIGURATION ====================

@dataclass
class AgentConfig:
    """Configuration for AI agent."""
    api_key: str = ""
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    system_prompt: str = "You are a helpful AI assistant. Provide clear, concise, and accurate responses."
    max_history: int = 50
    stream_response: bool = True
    save_history: bool = True
    history_dir: str = "chat_history"


# ==================== MESSAGE WIDGET ====================

class MessageBubble(QFrame):
    """A single message bubble in the chat."""
    
    def __init__(self, message: str, is_user: bool = True, timestamp: str = None, parent=None):
        super().__init__(parent)
        self.message = message
        self.is_user = is_user
        self.timestamp = timestamp or datetime.now().strftime("%H:%M")
        
        self.setup_ui()
        self.apply_style()
    
    def setup_ui(self):
        """Setup the message bubble UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        
        # Header with sender and timestamp
        header_layout = QHBoxLayout()
        
        sender_label = QLabel("You" if self.is_user else "AI Assistant")
        sender_label.setStyleSheet("""
            font-weight: bold;
            color: #4CAF50;
        """)
        header_layout.addWidget(sender_label)
        
        header_layout.addStretch()
        
        timestamp_label = QLabel(self.timestamp)
        timestamp_label.setStyleSheet("""
            color: #888888;
            font-size: 11px;
        """)
        header_layout.addWidget(timestamp_label)
        
        layout.addLayout(header_layout)
        
        # Message content
        self.content_label = QLabel()
        self.content_label.setWordWrap(True)
        self.content_label.setTextFormat(Qt.TextFormat.RichText)
        self.content_label.setOpenExternalLinks(True)
        
        # Convert markdown to HTML if available
        if MARKDOWN_AVAILABLE and not self.is_user:
            html = markdown.markdown(
                self.message,
                extensions=['extra', 'codehilite', 'tables']
            )
            self.content_label.setText(html)
        else:
            self.content_label.setText(self.message)
        
        layout.addWidget(self.content_label)
        
        # Copy button (appears on hover)
        self.copy_btn = QPushButton("📋 Copy")
        self.copy_btn.setFixedSize(60, 24)
        self.copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #4CAF50;
            }
        """)
        self.copy_btn.clicked.connect(self.copy_message)
        
        # Position copy button in top-right corner
        self.copy_btn.setParent(self)
        self.copy_btn.hide()
    
    def apply_style(self):
        """Apply styling based on message type."""
        if self.is_user:
            self.setStyleSheet("""
                MessageBubble {
                    background-color: #2d2d30;
                    border: 1px solid #3a3a3a;
                    border-radius: 10px;
                    margin: 5px 50px 5px 10px;
                }
            """)
        else:
            self.setStyleSheet("""
                MessageBubble {
                    background-color: #1e1e1e;
                    border: 1px solid #4CAF50;
                    border-radius: 10px;
                    margin: 5px 10px 5px 50px;
                }
            """)
    
    def copy_message(self):
        """Copy message to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.message)
        
        # Show feedback
        self.copy_btn.setText("✅ Copied!")
        QTimer.singleShot(1000, lambda: self.copy_btn.setText("📋 Copy"))
    
    def enterEvent(self, event):
        """Show copy button on hover."""
        self.copy_btn.show()
        self.copy_btn.raise_()
        # Position in top-right corner
        self.copy_btn.move(self.width() - 70, 5)
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Hide copy button when mouse leaves."""
        self.copy_btn.hide()
        super().leaveEvent(event)
    
    def resizeEvent(self, event):
        """Update copy button position on resize."""
        if self.copy_btn.isVisible():
            self.copy_btn.move(self.width() - 70, 5)
        super().resizeEvent(event)


# ==================== CHAT DISPLAY ====================

class ChatDisplay(QScrollArea):
    """Scrollable area for displaying messages."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        
        # Container for messages
        self.container = QWidget()
        self.container.setObjectName("chatContainer")
        self.layout = QVBoxLayout(self.container)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        self.setWidget(self.container)
        
        # Welcome message
        self.add_welcome_message()
    
    def add_welcome_message(self):
        """Add welcome message to chat."""
        welcome_text = """
        <h3>👋 Welcome to AI Assistant!</h3>
        <p>I'm here to help you with any questions or tasks. Feel free to ask me about:</p>
        <ul>
            <li>General knowledge and information</li>
            <li>Programming and technical help</li>
            <li>Writing and editing</li>
            <li>Analysis and problem-solving</li>
            <li>And much more!</li>
        </ul>
        <p><i>Powered by DeepSeek AI</i></p>
        """
        
        welcome_bubble = MessageBubble(welcome_text, is_user=False)
        self.layout.addWidget(welcome_bubble)
    
    def add_message(self, message: str, is_user: bool = True):
        """Add a new message to the chat."""
        bubble = MessageBubble(message, is_user)
        self.layout.addWidget(bubble)
        
        # Scroll to bottom
        QTimer.singleShot(100, self.scroll_to_bottom)
        
        return bubble
    
    def add_streaming_message(self) -> MessageBubble:
        """Add an empty message for streaming content."""
        bubble = MessageBubble("", is_user=False)
        self.layout.addWidget(bubble)
        return bubble
    
    def update_streaming_message(self, bubble: MessageBubble, text: str):
        """Update streaming message content."""
        if MARKDOWN_AVAILABLE:
            html = markdown.markdown(
                text,
                extensions=['extra', 'codehilite', 'tables']
            )
            bubble.content_label.setText(html)
        else:
            bubble.content_label.setText(text)
        
        # Scroll to bottom
        self.scroll_to_bottom()
    
    def clear_chat(self):
        """Clear all messages except welcome."""
        # Remove all widgets
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add welcome message back
        self.add_welcome_message()
    
    def scroll_to_bottom(self):
        """Scroll to the bottom of the chat."""
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history for API."""
        history = []
        for i in range(self.layout.count()):
            widget = self.layout.itemAt(i).widget()
            if isinstance(widget, MessageBubble):
                # Skip welcome message
                if i == 0 and "Welcome to AI Assistant" in widget.message:
                    continue
                
                role = "user" if widget.is_user else "assistant"
                history.append({
                    "role": role,
                    "content": widget.message
                })
        
        return history


# ==================== DEEPSEEK AGENT ====================

class DeepSeekAgent:
    """AI agent using DeepSeek API."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        
        if DEEPSEEK_AVAILABLE and config.api_key:
            self.client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url
            )
            self.available = True
        else:
            self.client = None
            self.available = False
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> Any:
        """Get chat completion from DeepSeek."""
        if not self.available:
            raise Exception("DeepSeek API not configured")
        
        # Add system message if not present
        if not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {
                "role": "system",
                "content": self.config.system_prompt
            })
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                stream=stream
            )
            
            return response
            
        except Exception as e:
            raise Exception(f"API Error: {str(e)}")


# ==================== CHAT THREAD ====================

class ChatThread(QThread):
    """Background thread for chat communication."""
    
    message_received = pyqtSignal(str)
    stream_chunk = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, agent: DeepSeekAgent, messages: List[Dict[str, str]], stream: bool = True):
        super().__init__()
        self.agent = agent
        self.messages = messages
        self.stream = stream
        self._running = True
    
    def stop(self):
        self._running = False
    
    def run(self):
        try:
            if self.stream:
                # Streaming mode
                response = self.agent.chat_completion(self.messages, stream=True)
                full_response = ""
                
                for chunk in response:
                    if not self._running:
                        break
                    
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        self.stream_chunk.emit(content)
                
                if self._running:
                    self.message_received.emit(full_response)
            else:
                # Non-streaming mode
                response = self.agent.chat_completion(self.messages, stream=False)
                if response.choices and response.choices[0].message.content:
                    content = response.choices[0].message.content
                    self.message_received.emit(content)
            
            if self._running:
                self.finished.emit()
                
        except Exception as e:
            self.error.emit(str(e))


# ==================== CHAT HISTORY MANAGER ====================

class ChatHistoryManager:
    """Manages saving and loading chat history."""
    
    def __init__(self, history_dir: str = "chat_history"):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)
    
    def save_conversation(self, messages: List[Dict[str, str]], title: str = None):
        """Save conversation to file."""
        if not title:
            title = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = self.history_dir / f"chat_{title}.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "messages": messages
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return str(filename)
    
    def load_conversation(self, file_path: str) -> List[Dict[str, str]]:
        """Load conversation from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get("messages", [])
    
    def list_conversations(self) -> List[Dict[str, str]]:
        """List all saved conversations."""
        conversations = []
        for file in self.history_dir.glob("chat_*.json"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                conversations.append({
                    "file": str(file),
                    "timestamp": data.get("timestamp", ""),
                    "preview": str(file.name)
                })
            except:
                pass
        
        # Sort by timestamp (newest first)
        conversations.sort(key=lambda x: x["timestamp"], reverse=True)
        return conversations


# ==================== AI AGENT WIDGET ====================

class AIAgentWidget(QWidget):
    """Main AI agent widget with clean UI."""
    
    def __init__(self, parent=None, theme_manager=None):
        super().__init__(parent)
        self.theme_manager = theme_manager
        self.agent = None
        self.chat_thread = None
        self.current_stream_bubble = None
        self.history_manager = ChatHistoryManager()
        
        # Load config
        self.config = AgentConfig()
        self._load_config()
        
        self.setup_ui()
        
        # Apply theme if available
        if theme_manager:
            self.apply_theme(theme_manager.is_dark)
        
        # Initialize agent if API key is available
        if self.config.api_key:
            self.agent = DeepSeekAgent(self.config)
    
    def setup_ui(self):
        """Setup the AI agent UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # ===== Header with controls =====
        header = QWidget()
        header.setFixedHeight(50)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title = QLabel("🤖 AI Assistant")
        title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #4CAF50;
        """)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # New Chat button
        self.new_chat_btn = QPushButton("🔄 New Chat")
        self.new_chat_btn.setFixedSize(100, 30)
        self.new_chat_btn.clicked.connect(self.new_chat)
        header_layout.addWidget(self.new_chat_btn)
        
        # History button
        self.history_btn = QPushButton("📋 History")
        self.history_btn.setFixedSize(100, 30)
        self.history_btn.clicked.connect(self.show_history)
        header_layout.addWidget(self.history_btn)
        
        # Settings button
        self.settings_btn = QPushButton("⚙️ Settings")
        self.settings_btn.setFixedSize(100, 30)
        self.settings_btn.clicked.connect(self.show_settings)
        header_layout.addWidget(self.settings_btn)
        
        main_layout.addWidget(header)
        
        # ===== Chat Display =====
        self.chat_display = ChatDisplay()
        main_layout.addWidget(self.chat_display, 1)
        
        # ===== Input Area =====
        input_container = QWidget()
        input_container.setFixedHeight(100)
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(5, 5, 5, 5)
        input_layout.setSpacing(5)
        
        # Message input
        self.message_input = QTextEdit()
        self.message_input.setPlaceholderText("Type your message here... (Ctrl+Enter to send)")
        self.message_input.setMaximumHeight(60)
        self.message_input.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d30;
                color: #ffffff;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
            }
        """)
        
        # Handle Ctrl+Enter to send
        self.message_input.installEventFilter(self)
        
        input_layout.addWidget(self.message_input)
        
        # Send button and status
        button_layout = QHBoxLayout()
        
        self.send_btn = QPushButton("📤 Send")
        self.send_btn.setFixedSize(100, 35)
        self.send_btn.clicked.connect(self.send_message)
        button_layout.addWidget(self.send_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setFixedSize(100, 35)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_generation)
        button_layout.addWidget(self.stop_btn)
        
        button_layout.addStretch()
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888888;")
        button_layout.addWidget(self.status_label)
        
        input_layout.addLayout(button_layout)
        
        main_layout.addWidget(input_container)
    
    def eventFilter(self, obj, event):
        """Handle Ctrl+Enter to send message."""
        if obj == self.message_input and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                self.send_message()
                return True
        return super().eventFilter(obj, event)
    
    def _load_config(self):
        """Load agent config from file."""
        config_file = Path("agent_config.json")
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
        """Save agent config to file."""
        try:
            with open("agent_config.json", 'w', encoding='utf-8') as f:
                data = {
                    'api_key': self.config.api_key,
                    'model': self.config.model,
                    'temperature': self.config.temperature,
                    'max_tokens': self.config.max_tokens,
                    'system_prompt': self.config.system_prompt,
                    'stream_response': self.config.stream_response
                }
                json.dump(data, f, indent=2)
        except:
            pass
    
    def new_chat(self):
        """Start a new chat."""
        reply = QMessageBox.question(
            self, "New Chat",
            "Start a new chat? Current conversation will be cleared.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.chat_display.clear_chat()
            self.status_label.setText("New chat started")
    
    def show_history(self):
        """Show chat history dialog."""
        conversations = self.history_manager.list_conversations()
        
        if not conversations:
            QMessageBox.information(self, "History", "No saved conversations found.")
            return
        
        # Simple dialog to show history
        items = [f"{c['timestamp'][:19]} - {c['preview']}" for c in conversations]
        
        item, ok = QInputDialog.getItem(
            self, "Chat History",
            "Select a conversation to load:",
            items, 0, False
        )
        
        if ok and item:
            index = items.index(item)
            file_path = conversations[index]["file"]
            
            try:
                messages = self.history_manager.load_conversation(file_path)
                
                # Clear current chat
                self.chat_display.clear_chat()
                
                # Load messages (skip system message)
                for msg in messages:
                    if msg["role"] != "system":
                        self.chat_display.add_message(
                            msg["content"],
                            is_user=(msg["role"] == "user")
                        )
                
                self.status_label.setText(f"Loaded conversation from {conversations[index]['timestamp'][:19]}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load conversation: {e}")
    
    def show_settings(self):
        """Show settings dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("AI Agent Settings")
        dialog.setMinimumWidth(500)
        
        layout = QVBoxLayout(dialog)
        
        # API Settings
        api_group = QGroupBox("API Configuration")
        api_layout = QFormLayout()
        
        # API Key
        api_key_input = QLineEdit()
        api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        api_key_input.setText(self.config.api_key)
        api_layout.addRow("API Key:", api_key_input)
        
        # Model
        model_combo = QComboBox()
        model_combo.addItems(["deepseek-chat", "deepseek-coder"])
        model_combo.setCurrentText(self.config.model)
        api_layout.addRow("Model:", model_combo)
        
        # Temperature
        temp_spin = QDoubleSpinBox()
        temp_spin.setRange(0.0, 2.0)
        temp_spin.setSingleStep(0.1)
        temp_spin.setValue(self.config.temperature)
        api_layout.addRow("Temperature:", temp_spin)
        
        # Max Tokens
        tokens_spin = QSpinBox()
        tokens_spin.setRange(100, 8000)
        tokens_spin.setSingleStep(100)
        tokens_spin.setValue(self.config.max_tokens)
        api_layout.addRow("Max Tokens:", tokens_spin)
        
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)
        
        # System Prompt
        prompt_group = QGroupBox("System Prompt")
        prompt_layout = QVBoxLayout()
        
        prompt_edit = QTextEdit()
        prompt_edit.setPlainText(self.config.system_prompt)
        prompt_edit.setMaximumHeight(100)
        prompt_layout.addWidget(prompt_edit)
        
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        
        stream_check = QCheckBox("Enable streaming responses")
        stream_check.setChecked(self.config.stream_response)
        options_layout.addWidget(stream_check)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(lambda: self._save_settings(
            dialog, api_key_input.text(), model_combo.currentText(),
            temp_spin.value(), tokens_spin.value(),
            prompt_edit.toPlainText(), stream_check.isChecked()
        ))
        button_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def _save_settings(self, dialog, api_key, model, temperature, max_tokens, system_prompt, stream):
        """Save settings from dialog."""
        self.config.api_key = api_key
        self.config.model = model
        self.config.temperature = temperature
        self.config.max_tokens = max_tokens
        self.config.system_prompt = system_prompt
        self.config.stream_response = stream
        
        self._save_config()
        
        # Reinitialize agent
        if self.config.api_key:
            self.agent = DeepSeekAgent(self.config)
        
        dialog.accept()
        self.status_label.setText("Settings saved")
    
    def send_message(self):
        """Send message to AI agent."""
        message = self.message_input.toPlainText().strip()
        
        if not message:
            return
        
        # Check if agent is available
        if not self.agent or not self.agent.available:
            QMessageBox.warning(
                self, "API Key Required",
                "Please configure your DeepSeek API key in Settings.\n\n"
                "You can get one from: https://platform.deepseek.com/"
            )
            self.show_settings()
            return
        
        # Add user message to chat
        self.chat_display.add_message(message, is_user=True)
        self.message_input.clear()
        
        # Prepare messages for API
        messages = self.chat_display.get_conversation_history()
        
        # Start streaming response
        self.current_stream_bubble = self.chat_display.add_streaming_message()
        
        # Start chat thread
        self.chat_thread = ChatThread(
            self.agent,
            messages,
            stream=self.config.stream_response
        )
        
        self.chat_thread.stream_chunk.connect(self.on_stream_chunk)
        self.chat_thread.message_received.connect(self.on_message_received)
        self.chat_thread.finished.connect(self.on_chat_finished)
        self.chat_thread.error.connect(self.on_chat_error)
        
        # Update UI
        self.send_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("AI is thinking...")
        
        self.chat_thread.start()
    
    def on_stream_chunk(self, chunk: str):
        """Handle streaming chunk."""
        if self.current_stream_bubble:
            current_text = self.current_stream_bubble.message
            self.current_stream_bubble.message = current_text + chunk
            self.chat_display.update_streaming_message(
                self.current_stream_bubble,
                self.current_stream_bubble.message
            )
    
    def on_message_received(self, message: str):
        """Handle complete message."""
        if self.current_stream_bubble:
            self.current_stream_bubble.message = message
            self.chat_display.update_streaming_message(
                self.current_stream_bubble,
                message
            )
    
    def on_chat_finished(self):
        """Handle chat completion."""
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Ready")
        
        # Save conversation if enabled
        if self.config.save_history:
            try:
                messages = self.chat_display.get_conversation_history()
                if messages:
                    # Generate title from first user message
                    title = None
                    for msg in messages:
                        if msg["role"] == "user":
                            title = msg["content"][:30] + "..."
                            break
                    
                    self.history_manager.save_conversation(messages, title)
            except:
                pass
        
        self.current_stream_bubble = None
    
    def on_chat_error(self, error_msg: str):
        """Handle chat error."""
        self.send_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Error")
        
        # Show error in chat
        self.chat_display.add_message(
            f"❌ Error: {error_msg}",
            is_user=False
        )
        
        self.current_stream_bubble = None
    
    def stop_generation(self):
        """Stop current generation."""
        if self.chat_thread and self.chat_thread.isRunning():
            self.chat_thread.stop()
            self.chat_thread.wait()
            
            self.send_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Stopped")
            
            # Add stopped message
            self.chat_display.add_message(
                "⚠️ Generation stopped by user.",
                is_user=False
            )
    
    def apply_theme(self, is_dark):
        """Apply theme to agent widget."""
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
                QScrollArea {
                    border: none;
                    background-color: transparent;
                }
                #chatContainer {
                    background-color: #1e1e1e;
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
                QScrollArea {
                    border: none;
                    background-color: transparent;
                }
                #chatContainer {
                    background-color: #f8f9fa;
                }
            """)


# ==================== INTEGRATION FUNCTION ====================

def create_agent_tab(parent, theme_manager=None):
    """Create AI agent tab for AI Studio Pro."""
    return AIAgentWidget(parent, theme_manager)


# ==================== EXPORTS ====================

__all__ = [
    'AIAgentWidget',
    'DeepSeekAgent',
    'ChatHistoryManager',
    'create_agent_tab',
    'DEEPSEEK_AVAILABLE'
]