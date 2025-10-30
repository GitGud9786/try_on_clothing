@echo off
REM ===============================
REM Run PromptDresser Setup Checker
REM ===============================

REM Check if the correct Conda environment is active
if "%CONDA_DEFAULT_ENV%"=="promptdresser" (
    echo ✅ Conda environment "promptdresser" is active.
) else (
    echo [WARNING] Conda environment "promptdresser" is NOT active!
    echo Please activate it before running this script.
)

REM Check folders
echo Verifying folder structure...

IF NOT EXIST "configs\VITONHD.yaml" (
    echo [ERROR] configs\VITONHD.yaml not found!
) ELSE (
    echo [OK] Config found.
)

IF NOT EXIST "checkpoints\VITONHD\model\pytorch_model.bin" (
    echo [ERROR] Pretrained model not found!
) ELSE (
    echo [OK] VITONHD checkpoint found.
)

IF NOT EXIST "checkpoints\human_parsing\parsing_atr.onnx" (
    echo [ERROR] Human parsing model not found!
) ELSE (
    echo [OK] Human parsing model found.
)

IF NOT EXIST "datasets\VITON-HD\test_coarse" (
    echo [ERROR] Dataset folder test_coarse not found!
) ELSE (
    echo [OK] Dataset folder test_coarse found.
)

IF NOT EXIST "datasets\VITON-HD\test_fine" (
    echo [ERROR] Dataset folder test_fine not found!
) ELSE (
    echo [OK] Dataset folder test_fine found.
)

IF NOT EXIST "datasets\VITON-HD\test_coarse_agnostic_mask" (
    echo [WARNING] Optional: test_coarse_agnostic_mask not found
) ELSE (
    echo [OK] Optional test_coarse_agnostic_mask found.
)

IF NOT EXIST "datasets\VITON-HD\test_fine_agnostic_mask" (
    echo [WARNING] Optional: test_fine_agnostic_mask not found
) ELSE (
    echo [OK] Optional test_fine_agnostic_mask found.
)

IF NOT EXIST "datasets\VITON-HD\test_gpt4o.json" (
    echo [WARNING] Optional: test_gpt4o.json not found
) ELSE (
    echo [OK] Optional test_gpt4o.json found.
)

echo.
echo Setup verification complete. Your environment is ready!
pause
