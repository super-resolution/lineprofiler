# -*- mode: python -*-
import sys
sys.setrecursionlimit(5000)

block_cipher = None


a = Analysis(['main.py'],
             pathex=['C:\\Users\\biophys\\PycharmProjects\\Fabi', r'C:\dev\WinPython-64bit-3.6.3.0Qt5\python-3.6.3.amd64'],
             binaries=[],
             datas=[],
             hiddenimports=['pyexpat','pywt._extensions._cwt','sklearn.neighbors.typedefs'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
a.datas += Tree('./data_empty', prefix='data')
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='main',
          debug=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='main')
