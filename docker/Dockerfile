FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

LABEL maintainer="hieplt"
LABEL organization="ahs"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install pip deps (add more as needed)
RUN pip install --upgrade pip
RUN pip install torchinfo==1.8.0 opencv-python==4.12.0.88 \
    albumentations==2.0.8 mlflow==3.3.1 uvicorn fastapi python-multipart prometheus_client

WORKDIR /workspace

# Copy project files from build context root into image
# Build this image from project root: `docker build -f docker/Dockerfile -t ahs:latest .`
COPY ./src /workspace/src
COPY ./Makefile /workspace/Makefile

# expose port for FastAPI (documentation)
EXPOSE 8000

CMD ["uvicorn", "src.ahs.api.fast_api:app", "--host", "0.0.0.0", "--port", "8000"]