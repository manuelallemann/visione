# Use a PyTorch base image with CUDA support, consistent with other services
FROM docker.io/pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install Python dependencies
# --no-cache-dir reduces image size
# Ensure pip is up-to-date first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Set environment variables
# Add the current directory to PYTHONPATH to allow imports from service.py
ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app"
# Set TORCH_HOME to a specific cache directory for PyTorch Hub models
ENV TORCH_HOME /cache/torch
# Create the cache directory and set appropriate permissions if needed
RUN mkdir -p /cache/torch && chmod -R 777 /cache/torch

# Copy the rest of the service's source code into the container
COPY . .

# Expose the port the Flask app will run on (if not already exposed by base image or handled by orchestrator)
# Default for these services seems to be 8080
EXPOSE 8080

# Command to run the Flask application
# -u for unbuffered Python output, good for container logs
CMD ["python", "-u", "service.py"]
