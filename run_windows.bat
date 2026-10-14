@echo off
setlocal
cd /d "%~dp0"
title SignBridge Launcher

set "PYTHON_CMD="
where python >nul 2>nul
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.13 --version >nul 2>nul
        if not errorlevel 1 set "PYTHON_CMD=py -3.13"
    )
)
if not defined PYTHON_CMD (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.11 --version >nul 2>nul
        if not errorlevel 1 set "PYTHON_CMD=py -3.11"
    )
)
if not defined PYTHON_CMD (
    echo ERROR: Compatible 64-bit Python 3.13 or 3.11 was not found.
    echo Install Python manually. This script will not change system settings.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the project virtual environment with %PYTHON_CMD%...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto :failed
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :failed
python -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3, 11), (3, 13)) else 1)"
if errorlevel 1 goto :wrongpython
python -m pip install --upgrade pip
if errorlevel 1 goto :failed
python -m pip install -r requirements.txt
if errorlevel 1 goto :failed
python -c "import cv2, mediapipe, numpy, pyttsx3, sklearn, joblib; from mediapipe.tasks.python import vision; assert hasattr(vision, 'GestureRecognizer'); print('Dependencies ready: MediaPipe', mediapipe.__version__, '| OpenCV', cv2.__version__, '| scikit-learn', sklearn.__version__)"
if errorlevel 1 goto :failed
python main.py
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo SignBridge could not continue. Check the error above and docs\TROUBLESHOOTING.md.
pause
exit /b 1

:wrongpython
echo.
echo ERROR: The existing .venv uses an unsupported Python version.
echo Remove only the project .venv folder, then run this launcher again.
pause
exit /b 1
