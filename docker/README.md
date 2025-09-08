# Train in container
## On host, run mlflow server
```bash
mlflow server \
  --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlruns.db
```
## Built image
```bash
docker build -t bccd_train -f Dockerfile_train .
```
## Run docker container
```bash
docker run -it --network host --rm \
-v $(pwd)/../BCCD_Dataset:/workspace/BCCD_Dataset:ro \
-v $(pwd)/mlruns:/workspace/mlruns \
bccd_train bash
```