# -*- mode: python ; coding: utf-8 -*-

# --- 1. 定义正确的运行时DLL路径 ---
# 根据官方说明，这个路径在 "Common Files" 文件夹下。
# 假设你的软件名称是 "MVS" (通常是这样)。
# 假设你的Python是64位的，所以我们使用 Win64_x64。
# 注意：路径中的斜杠，使用'/'可以避免转义问题。
MVCAM_RUNTIME_DLL_PATH = 'C:/Program Files (x86)/Common Files/MVS/Runtime/Win64_x64'

block_cipher = None

a = Analysis(
    ['MultipleCameras.py'],
    pathex=[],
    binaries=[],
    # --- 2. 在datas中包含正确的运行时DLL ---
    datas=[
        (MVCAM_RUNTIME_DLL_PATH, './') 
    ],
    hiddenimports=['platform'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DualCameraApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='logo.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DualCameraApp',
)