# Ref: https://github.com/terraform-google-modules/terraform-google-kubernetes-engine/blob/master/examples/simple_autopilot_public
# To define that we will use GCP
terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "4.80.0" // Provider version
    }
  }
  required_version = ">= 1.5.6" // Terraform version
}

// The library with methods for creating and
// managing the infrastructure in GCP, this will
// apply to all the resources in the project
provider "google" {
  project     = var.project_id
  region      = var.region
}

// Google Kubernetes Engine
# resource "google_container_cluster" "my-gke" {
#   name     = "${var.project_id}-gke"
#   location = var.region

#   # Initial node count
#   initial_node_count = 1

#   // Enabling Autopilot for this cluster
#   enable_autopilot = false
#   node_config {
#     machine_type = "e2-medium"
#     disk_type    = "pd-standard"
#     disk_size_gb = 50
#   }
# }

resource "google_storage_bucket" "model_registry" {
  name     = "${var.project_id}-model-registry"
  location = var.region

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 5
    }
    action {
      type = "Delete"
    }
  }
}