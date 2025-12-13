"""
Hugging Face Spaces entry point
This file wraps the FastAPI app for Hugging Face Spaces deployment
"""
import os
import uvicorn
from main import app

# Hugging Face Spaces expects the app to be available
# The port is set by Hugging Face automatically via PORT environment variable
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

