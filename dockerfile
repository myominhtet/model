FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libicu-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the working directory
COPY requirements.txt /app/

# Install the necessary dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir lightgbm

RUN pip install --no-cache-dir xgboost

# Add the dummy function to the uvicorn script
RUN sed -i '1 a\
def dummy(text):\
    return text\
' /usr/local/bin/uvicorn
# Copy the rest of the application files into the working directory
COPY . /app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the FastAPI application using Uvicorn when the container launches
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
