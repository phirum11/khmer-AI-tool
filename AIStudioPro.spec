# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('silero_vad.onnx', '.'), ('live_style.css', '.'), ('agent_config.json', '.'), ('khmer_lexicon.json', '.'), ('translation_config.json', '.')]
binaries = []
hiddenimports = ['PyQt6', 'edge_tts', 'faster_whisper', 'vosk', 'pydub', 'yt_dlp', 'onnxruntime', 'librosa', 'scipy', 'webrtcvad']
tmp_ret = collect_all('silero_vad')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['.vscode\\main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AIStudioPro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
