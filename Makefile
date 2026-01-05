PYTHON := python
TEST_PATH := tests/
SRC_PATH := src/

.PHONY: train infer api gce gke quantize_dynamic
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

## Create Compute Engine & Jenkins instances
instances:
	ansible-playbook iac/ansible/create_compute_instance.yaml

jenkins_container:
	ansible-playbook -i iac/ansible/inventory iac/ansible/install_and_run_docker.yml

## Create kubernetes cluster
k8s:
	gcloud auth application-default login

	terraform -chdir=iac/terraform init
	terraform -chdir=iac/terraform plan
	terraform -chdir=iac/terraform apply

## Helm install nginx & api
helm:
	kubectl create clusterrolebinding cluster-admin-binding \
	--clusterrole cluster-admin \
	--user $(gcloud config get-value account)

	kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.14.1/deploy/static/provider/cloud/deploy.yaml

	helm upgrade  --install hematology-api helm/apps/hematology-api -n model-serving --create-namespace