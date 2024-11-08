# Use CUDA base image for GPU support
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY builder/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy handler code
COPY handler.py /handler.py

# Set working directory
WORKDIR /

# Start handler
CMD [ "python", "-u", "/handler.py" ]
