# Use an official Python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git

# Set the working directory inside the container
WORKDIR /app

# Copy the repository into the container
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

# Run tests when the container starts
CMD ["pytest"]
