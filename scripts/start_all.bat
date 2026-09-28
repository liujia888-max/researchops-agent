@echo off
rem ResearchOps-Agent one-click launcher (Windows, no Docker). ASCII only.
rem Starts Qdrant + FastAPI backend + Next.js frontend, then opens the browser.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_all.ps1"
