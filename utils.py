import os, random, numpy as np, torch, yaml

def load_config(path='config.yaml'):
    """Load a YAML file and return its contents as a dictionary."""
    with open(path, 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)
    
def set_seed(seed: int):
    """Set the random seed for reproducibility across various libraries."""
    random.seed(seed), np.random.seed(seed)
    torch.manual_seed(seed), torch.cuda.manual_seed_all(seed)

def get_device():
    """Return the available device: GPU if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path: str):
    """Ensure that a directory exists; create it if it doesn't."""
    if not os.path.exists(path):
        os.makedirs(path)


  