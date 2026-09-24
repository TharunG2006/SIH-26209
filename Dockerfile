FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some scientific packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY prototype/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the dashboard port
EXPOSE 8501

# Create a startup script that runs the background pipeline and the UI simultaneously
RUN echo '#!/bin/bash\n\
python prototype/src/pipeline.py --interval 60 &\n\
python -m streamlit run prototype/dashboard/app.py --server.port=8501 --server.address=0.0.0.0\n\
' > start.sh && chmod +x start.sh

# Run the system
CMD ["./start.sh"]
