# Horde Agent Docker Image

FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directories
RUN mkdir -p data/raw data/processed data/models data/results data/reports data/state

# Expose Streamlit port
EXPOSE 8501

# Default command: run Streamlit UI
CMD ["streamlit", "run", "ui/app.py", "--server.address", "0.0.0.0"]
