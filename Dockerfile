# Base image: Python 3.10
FROM python:3.10-slim

# Avoid interactive prompts (not strictly needed here)
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Copy & install Python deps (including headless OpenCV)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY streamlit_app.py tea_leaf_model.py utils.py ./

# (Optional: if you actually need data/models at runtime, COPY them here; otherwise ignore)
# COPY data/ ./data/
# COPY models/ ./models/

# Expose Streamlit port
EXPOSE 8501

# Launch the app
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
