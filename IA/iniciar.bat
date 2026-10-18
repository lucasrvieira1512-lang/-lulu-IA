@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python nao foi encontrado no PATH. Instale o Python e tente novamente.
    pause
    exit /b 1
)

python -c "import sys; print('Python do assistente:', sys.executable)"

python -c "import tkinter" >nul 2>nul
if errorlevel 1 (
    echo Esta instalacao do Python nao inclui Tkinter.
    pause
    exit /b 1
)

python -c "import pyttsx3" >nul 2>nul
if errorlevel 1 (
    echo Instalando pyttsx3 para o mesmo Python que iniciara o assistente...
    python -m pip install pyttsx3
    if errorlevel 1 (
        echo Nao foi possivel instalar pyttsx3 neste Python.
        pause
        exit /b 1
    )
)

python main.py
if errorlevel 1 pause