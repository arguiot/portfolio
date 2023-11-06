# Use the Ubuntu Docker image as base
FROM ubuntu:latest

# Ensure Python 3.10 is Installed
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y python3.10 python3-pip

# Install Clang
RUN apt-get install -y clang cmake

# Set working directory
WORKDIR /app

# Copy the local project directory content to Docker image
COPY . /app

# Install Python dependencies
RUN pip3 install -e .
RUN pip3 install -r requirements.txt

# Run command (optional)
CMD ["python3", "run_backtest.py"]
