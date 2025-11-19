@echo off
echo ============================================================
echo ASL MODEL TRAINING - AUTO GPU/CPU DETECTION
echo ============================================================
echo.

REM Check if we're in the right directory
if not exist "src\train.py" (
    echo ERROR: Run this from the signssl-project folder!
    pause
    exit /b 1
)

REM Check if data files exist
if not exist "data\labeled_new\train_data.npz" (
    echo ERROR: train_data.npz not found!
    echo Copy train_data.npz to data\labeled_new\
    pause
    exit /b 1
)

echo Files found! Starting training...
echo Dataset: 10,134 samples
echo.
echo IMPORTANT:
echo - GPU: 2-3 hours
echo - CPU: 4-5 days
echo - Do NOT close this window or shut down laptop
echo.
echo Press Ctrl+C to cancel, or
pause

echo.
echo ============================================================
echo Attempting GPU training first...
echo ============================================================
echo.

python src\train.py --labeled_data data\labeled_new\train_data.npz --device cuda --config configs\config.yaml

if %errorlevel% neq 0 (
    echo.
    echo ============================================================
    echo GPU failed - Using CPU (this will take 4-5 days)
    echo ============================================================
    echo.
    python src\train.py --labeled_data data\labeled_new\train_data.npz --device cpu --config configs\config.yaml
)

echo.
echo ============================================================
echo TRAINING COMPLETE!
echo ============================================================
echo.
echo Model saved to: models\best_model.pth
echo.
pause
