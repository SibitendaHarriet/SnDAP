# Use a different, available CUDA image
FROM nvidia/cuda:12.5-runtime-ubuntu20.04
# FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu20.04


# Set environment variables to reduce interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip3 install --upgrade pip

# Copy the requirements file into the Docker container
COPY requirements.txt .

# Install dependencies from the requirements.txt file
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit will run on
EXPOSE 8501

# Set the entrypoint for Streamlit
# CMD ["streamlit", "run", "app.py"]

# Define the command to run the app
CMD ["streamlit", "run", "topics_app.py"]
