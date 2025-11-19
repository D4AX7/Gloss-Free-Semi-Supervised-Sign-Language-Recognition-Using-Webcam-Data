@echo off
cd /d "%~dp0"
echo ============================================================
echo TESTING YOUR NEW TRAINED MODEL!
echo ============================================================
echo.
echo This will test if your model predicts correctly now.
echo Previous issue: Stuck on "THANKYOU" with 99%% confidence
echo Expected now: Diverse predictions with 80-95%% confidence
echo.
echo Starting webcam inference...
echo.

python src\infer_simple.py

if errorlevel 1 (
    echo.
    echo Error occurred. Press any key to exit.
)
pause
