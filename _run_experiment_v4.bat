@echo off
cd /d D:\maxo
python -X utf8 -u run_experiment.py > results\experiment_v4_stdout.txt 2> results\experiment_v4_stderr.txt
echo %ERRORLEVEL% > results\experiment_v4_exit.txt
