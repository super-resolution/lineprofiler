for /r "src\gui" %%d in (*.*) do python -m PyQt5.uic.pyuic src\gui\%%~nd.ui -o src\viewer\%%~nd.py