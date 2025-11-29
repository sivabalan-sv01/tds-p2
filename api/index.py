"""
Vercel serverless function handler for FastAPI app
This file imports the app from main.py without modifying core functions
"""
from mangum import Mangum
import sys
import os

# Add parent directory to path so we can import main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the FastAPI app from main.py
from main import app

# Wrap the FastAPI app with Mangum ASGI adapter for Vercel
# lifespan="auto" will handle startup/shutdown events properly
handler = Mangum(app, lifespan="auto")

# Export handler for Vercel (Vercel looks for 'handler' variable)

