FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install pip deps (add more as needed)
RUN pip install --upgrade pip

RUN pip install --upgrade pip \
    && pip install streamlit minio requests opencv-python==4.12.0.88 dotenv

COPY ./src/ahs/ui /app/src/ahs/ui
COPY ./src/ahs/utils /app/src/ahs/utils

EXPOSE 8501

CMD ["streamlit", "run", "/app/src/ahs/ui/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0"]