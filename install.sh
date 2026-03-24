#!/bin/bash
cd "$(dirname "$0")"
source venv/Scripts/activate
pip install --upgrade pip
pip install fastapi uvicorn pydantic python-dotenv python-multipart
pip install pandas numpy yfinance httpx aiohttp
pip install feedparser beautifulsoup4 ta
pip install openai langchain langchain-openai
echo "Installation complete!"
pip list
