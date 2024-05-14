from PyInstaller.building.api import COLLECT as COLLECT_BINARY, EXE, PYZ, Analysis
from PyInstaller.utils.hooks import collect_data_files

# Collect data files from lightning package
datas = collect_data_files('lightning')

# Add other data files if needed, e.g., 'whisperx'
# datas.extend(collect_data_files('whisperx'))

# Specify Analysis configuration
a = Analysis(
    ['your_program.py'],  # Replace 'your_program.py' with your actual Python file
    pathex=['.'],  # List additional directories to search for modules
    binaries=[],
    datas=datas,
    hiddenimports=['loguru', 'whisperx', 'torch'],  # Add any additional hidden imports
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False
)

# Build the EXE using the Analysis
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name='your_program',  # Replace 'your_program' with desired EXE name
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True
)

# Finalize the build process
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='dist')
