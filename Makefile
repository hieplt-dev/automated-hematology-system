PYTHON := python
MODEL_STAGE := production

MODEL_PATH := models/best_qint8.pt
GCS_MODEL_PATH := gs://ahsys-480510-model-registry-internal/hematology-model/$(MODEL_STAGE)

.PHONY: train infer api gce gke quantize_dynamic helm_install push_model alertmanager prometheus minio

train:  ## Run train
	$(PYTHON) -m src.ahs.cli.train_cli

## Run inference
infer:
	$(PYTHON) -m src.ahs.cli.infer_cli --img_path $(img_path)

## api
api:
	uvicorn src.ahs.api.fast_api:app --host 0.0.0.0 --port 8000 --reload

## Run dynamic quantization
quantize_dynamic:
	$(PYTHON) -m scripts.quantize_dynamic

## Create Compute Engine
## note: replace project ID, zone, machine type in create_compute_instance.yaml
instances:
	ansible-playbook iac/ansible/create_compute_instance.yaml

## note: replace external IP of instance in inventory file
jenkins_container:
	ansible-playbook -i iac/ansible/inventory iac/ansible/install_and_run_docker.yml

## Create kubernetes cluster
k8s:
	@echo "Authenticating with GCP..."
	@gcloud auth activate-service-account --key-file=iac/ansible/secrets/ahsys-480510-844a29b58a02.json
	@echo "Authentication successful."

	terraform -chdir=iac/terraform init
	terraform -chdir=iac/terraform plan
	terraform -chdir=iac/terraform apply

## Push model to GCS (optional)
push_model:
	@echo "Authenticating with GCP..."
	@gcloud auth activate-service-account --key-file=iac/ansible/secrets/ahsys-480510-844a29b58a02.json
	@echo "Authentication successful."

	@echo "Uploading model to GCS..."
	gsutil -m cp $(MODEL_PATH) $(GCS_MODEL_PATH)/
	@echo "Model uploaded to $(GCS_MODEL_PATH)"

## Install kube-prometheus-stack
kube-prometheus-stack:
	@echo "Installing Helm chart for kube-prometheus-stack..."
	@helm upgrade --install kube-prometheus-stack oci://ghcr.io/prometheus-community/charts/kube-prometheus-stack -n monitoring --create-namespace -f helm/monitoring/prometheus/values.yaml

	@echo "Applying PrometheusRule..."
	kubectl apply -f helm/monitoring/alertmanager/prometheus-rule.yaml -n monitoring

	@echo "Installing Helm chart for grafana ingress..."
	helm upgrade --install grafana-ingress helm/monitoring/grafana -n monitoring --create-namespace

## Helm install nginx & api
helm_install:
	@echo "Setting up Kubernetes cluster role binding..."
	kubectl create clusterrolebinding cluster-admin-binding \
		--clusterrole cluster-admin \
		--user $(shell gcloud config get-value account)

	@echo "Installing NGINX Ingress Controller..."
	@kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.14.1/deploy/static/provider/cloud/deploy.yaml

	@echo "Waiting for NGINX Ingress Controller to be ready..."
	@sleep 30

	@echo "Installing Helm chart for hematology-api..."
	helm upgrade --install hematology-api helm/apps/hematology-api -n model-serving --create-namespace

	@echo "Installing Helm chart for hematology-ui..."
	helm upgrade --install hematology-ui helm/apps/hematology-ui -n ui --create-namespace

minio:
	helm repo add minio https://charts.min.io/
	helm upgrade --install minio minio/minio -n minio -f helm/storage/minio/values.yaml --create-namespace

	@echo "Creating minio-credentials secret..."
	kubectl create secret generic minio-credentials \
		-n model-serving \
		--from-literal=S3_ENDPOINT="$S3_ENDPOINT_URL" \
		--from-literal=S3_ACCESS_KEY="$S3_ACCESS_KEY" \
		--from-literal=S3_SECRET_KEY="$S3_SECRET_KEY"\
		--from-literal=S3_BUCKET_NAME="$S3_BUCKET_NAME"
