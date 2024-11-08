# Use CUDA base image for GPU support
FROM runpod/base:0.6.2-cuda11.1.1

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
