# Use official Python 3.9 base image
FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose port 8080 for Flask/Gunicorn
EXPOSE 8080

# Run with Gunicorn (production-ready server)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]