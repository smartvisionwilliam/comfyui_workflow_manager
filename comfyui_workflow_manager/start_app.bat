@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM Smart Vision - ComfyUI Workflow Manager Startup Script (Windows)
REM This script will automatically activate virtual environment and start app.py

echo ==========================================
echo Starting Smart Vision - ComfyUI Workflow Manager...
echo ==========================================

REM Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Current directory: %CD%

REM Check if .venv directory exists (check current directory and parent directory)
if exist ".venv" (
    echo Found .venv virtual environment in current directory, activating...
    
    REM Activate virtual environment
    if exist ".venv\Scripts\activate.bat" (
        call .venv\Scripts\activate.bat
        echo Virtual environment activated
    ) else if exist ".venv\bin\activate" (
        call .venv\bin\activate
        echo Virtual environment activated (Linux/WSL)
    ) else (
        echo Warning: Cannot find virtual environment activation script
    )
) else if exist "..\.venv" (
    echo Found .venv virtual environment in parent directory, activating...
    
    REM Activate virtual environment from parent directory
    if exist "..\.venv\Scripts\activate.bat" (
        call ..\.venv\Scripts\activate.bat
        echo Virtual environment activated
    ) else if exist "..\.venv\bin\activate" (
        call ..\.venv\bin\activate
        echo Virtual environment activated (Linux/WSL)
    ) else (
        echo Warning: Cannot find virtual environment activation script
    )
) else (
    echo .venv directory not found, using system Python environment
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found
    echo Please ensure Python is installed and in PATH
    pause
    exit /b 1
)

echo Python version:
python --version

REM Check if app.py exists
if not exist "app.py" (
    echo Error: app.py file not found
    echo Please ensure you are running this script in the correct directory
    pause
    exit /b 1
)

REM Check dependencies
echo Checking dependencies...
if exist "requirements.txt" (
    echo Found requirements.txt, checking dependencies...
    python -c "import sys; import importlib.util; required_packages = []; [required_packages.append(line.strip().split('==')[0].split('>=')[0].split('<=')[0]) for line in open('requirements.txt', 'r') if line.strip() and not line.strip().startswith('#')]; missing_packages = [pkg for pkg in required_packages if importlib.util.find_spec(pkg) is None]; print(f'Warning: The following dependencies may be missing: {missing_packages}') if missing_packages else print('All dependencies check completed'); print('If the application cannot run properly, please run: pip install -r requirements.txt') if missing_packages else None"
)

echo.
echo Starting Smart Vision - ComfyUI Workflow Manager...
echo The application will automatically open in your browser
echo If it doesn't open automatically, please manually visit: http://localhost:7860
echo.
echo Press Ctrl+C to stop the application
echo ==========================================

REM Start the application
python app.py

REM If the application exits, show message
echo.
echo Application has stopped
pause
