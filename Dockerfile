# owner: suryagautam21
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Build-time training so the image contains model artifacts.
# This avoids first-request latency on Render.
RUN python train_model.py

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

EXPOSE 8501

CMD ["bash", "start.sh"]
