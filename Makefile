PYTHON := python
TEST_PATH := tests/
SRC_PATH := src/
MODEL_STAGE := production

GCS_MODEL_PATH := gs://ahsys-480510-model-registry/hematology-model/$(MODEL_STAGE)

.PHONY: train infer api gce gke quantize_dynamic helm push_model
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

## Push model to GCS
push_model:
	@echo "Authenticating with GCP..."
	@gcloud auth activate-service-account --key-file=iac/ansible/secrets/ahsys-480510-844a29b58a02.json
	@echo "Authentication successful."

	@echo "Uploading model to GCS..."
	gsutil -m cp models/best_qint8.pt $(GCS_MODEL_PATH)/
	@echo "Model uploaded to $(GCS_MODEL_PATH)"

## Create Compute Engine & Jenkins instances
instances:
	ansible-playbook iac/ansible/create_compute_instance.yaml

jenkins_container:
	ansible-playbook -i iac/ansible/inventory iac/ansible/install_and_run_docker.yml

## Create kubernetes cluster
k8s:
	@echo "Authenticating with GCP..."
	gcloud auth activate-service-account --key-file=iac/ansible/secrets/ahsys-480510-844a29b58a02.json
	@echo "Authentication successful."

	terraform -chdir=iac/terraform init
	terraform -chdir=iac/terraform plan
	terraform -chdir=iac/terraform apply

## Helm install nginx & api
helm:
	kubectl create clusterrolebinding cluster-admin-binding \
	--clusterrole cluster-admin \
	--user $(gcloud config get-value account)

	kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.14.1/deploy/static/provider/cloud/deploy.yaml

	@echo "Installing Helm chart for hematology-api..."
	helm upgrade  --install hematology-api helm/apps/hematology-api -n model-serving --create-namespace