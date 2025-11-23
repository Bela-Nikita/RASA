# Dockerfile
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the core application logic
COPY main.py .
COPY custom_tools.py .

# Command to run the application
# For live deployment, this would be a web server (e.g., gunicorn)
CMD ["python", "main.py"]
