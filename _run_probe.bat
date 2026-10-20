@echo off
cd /d D:\maxo
python -X utf8 -u _import_probe.py > results\probe3_out.txt 2> results\probe3_err.txt
echo %ERRORLEVEL% > results\probe3_exit.txt
