# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all files into container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Command to run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
