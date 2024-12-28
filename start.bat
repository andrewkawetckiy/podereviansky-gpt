@echo off
REM Запуск Python скрипта з Gradio інтерфейсом

REM Перевірка наявності Python у системі
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Помилка: Python не знайдено у вашій системі.
    echo Будь ласка, встановіть Python і додайте його до PATH.
    pause
    exit /b
)

REM Запуск скрипта
python generation.py

REM Очікування закінчення роботи
pause
