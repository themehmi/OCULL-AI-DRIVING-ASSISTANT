# Use official lightweight Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgles2 \
    libegl1 \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the default Hugging Face Spaces port
EXPOSE 7860

# Run the application using standard Python to avoid Gunicorn/MediaPipe C++ threading deadlocks on Hugging Face
CMD ["python", "app.py"]
