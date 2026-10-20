@echo off
cd /d D:\maxo
python -X utf8 -u _v4_smoke.py > results\v4_smoke_out.txt 2> results\v4_smoke_err.txt
echo %ERRORLEVEL% > results\v4_smoke_exit.txt
