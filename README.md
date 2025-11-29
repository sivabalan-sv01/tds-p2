# LLM Quiz Solver API

FastAPI-based quiz solving service that uses GPT models and various data processing tools to solve quizzes automatically.

## Features

- Automatic quiz solving using GPT-5-nano
- Audio transcription using Whisper
- PDF text extraction
- CSV data analysis
- Image analysis using GPT-4 Vision
- Web scraping with Selenium
- Support for multiple quiz formats

## Hugging Face Spaces Deployment

This application is configured for deployment on Hugging Face Spaces using Docker.

### Setup Instructions

1. **Create a new Space on Hugging Face:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Select "Docker" as the SDK
   - Choose a name and visibility

2. **Configure Environment Variables:**
   In your Hugging Face Space settings, add these secrets:
   - `SECRET`: Your API secret key
   - `EMAIL`: Email address for quiz submissions
   - `AIPIPE_TOKEN`: Token for OpenAI API access

3. **Deploy:**
   - Push your code to the Space repository
   - Hugging Face will automatically build and deploy using the Dockerfile

### Environment Variables Required

The following environment variables must be set in Hugging Face Spaces:

- `SECRET`: Secret key for API authentication
- `EMAIL`: Email address used for quiz submissions
- `AIPIPE_TOKEN`: API token for OpenAI services (via aipipe.org)

### API Endpoints

- `GET /`: Health check endpoint
- `POST /quiz`: Main quiz solving endpoint

### Request Format

```json
{
  "email": "your-email@example.com",
  "secret": "your-secret",
  "url": "https://quiz-url.com"
}
```

## Local Development

To run locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export SECRET=your-secret
export EMAIL=your-email@example.com
export AIPIPE_TOKEN=your-token

# Run the application
python main.py
```

The API will be available at `http://localhost:8000`

## Docker Build

To build and run locally with Docker:

```bash
# Build the image
docker build -t quiz-solver .

# Run the container
docker run -p 7860:7860 \
  -e SECRET=your-secret \
  -e EMAIL=your-email@example.com \
  -e AIPIPE_TOKEN=your-token \
  quiz-solver
```

## Notes

- The application uses Chrome/Chromium for web scraping via Selenium
- Whisper model is loaded at startup for faster audio transcription
- Large file processing is optimized for memory efficiency
- The service supports quizzes with time limits up to 170 seconds per request

