import yaml

def load_training_config(path="ladybug/assets/training.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
