# 1) Base image: Python 3.10 slim
FROM python:3.10-slim

# 2) Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 3) Install OS‐level libs required by opencv-python
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1-mesa-glx \
      libgl1-mesa-dri \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# 4) Create and switch to the app directory
WORKDIR /app

# 5) Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6) Copy your application code and assets
COPY streamlit_app.py tea_leaf_model.py utils.py ./
COPY data/ ./data/
COPY models/ ./models/

# 7) Expose Streamlit’s port
EXPOSE 8501

# 8) Launch the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
