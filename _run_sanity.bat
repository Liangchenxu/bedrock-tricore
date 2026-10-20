@echo off
cd /d D:\maxo
python -X utf8 -u tests\test_sanity.py > results\sanity_v4_stdout.txt 2> results\sanity_v4_stderr.txt
echo %ERRORLEVEL% > results\sanity_v4_exit.txt
