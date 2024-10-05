# Use the official Python image.
FROM python:3.9-slim

# Set the working directory.
WORKDIR /app

# Copy requirements.txt and install dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code.
COPY . .

# Expose port 8080.
EXPOSE 8080

# Run the application.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
