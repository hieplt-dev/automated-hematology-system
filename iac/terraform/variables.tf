// Variables to use accross the project
// which can be accessed by var.project_id
variable "project_id" {
  description = "The project ID to host the cluster in"
  default     = "ahsys-480510"
}

variable "region" {
  description = "The region the cluster in"
  default     = "asia-southeast2"
}

variable "bucket_name" {
  description = "Model registry bucket name"
  default     = "model-registry-bucket"
}