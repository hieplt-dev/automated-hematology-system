import yaml


# load_config is a utility function to load configuration parameters from a YAML file.
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
