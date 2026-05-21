FROM python:3.11-slim

# Install system dependencies needed for some Python builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU-only version first to keep the image lightweight (~150MB instead of ~2GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    streamlit \
    plotly

# Copy codebase
COPY . .

# Expose ports for Streamlit and Server
EXPOSE 8501
EXPOSE 8765
