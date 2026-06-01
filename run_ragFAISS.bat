@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create .venv. Ensure Python is installed and on PATH.
        pause
        exit /b 1
    )
)

echo Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip

::echo Installing dependencies from requirements.txt...
::.venv\Scripts\python.exe -m pip install -r requirements.txt
::if errorlevel 1 (
::    echo pip install failed.
::    pause
::    exit /b 1
::)

echo Installing dependencies from requirements.txt...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo pip install failed.
    pause
    exit /b 1
)
echo Running ragFAISS.py ...
.venv\Scripts\python.exe ragFAISS.py
pause
