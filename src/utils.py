"""
Utility Functions

    save_models :  save models to models_dir
    load_models : load models from models_dir
"""

import os 
import joblib 



# save models 


def save_models(artifacts: dict, models_dir: str = "models") -> None:
    """
    Save a dict of {filename: object} to models_dir

    """
    os.makedirs(models_dir, exist_ok=True)
    for name, obj in artifacts.items():
        path = os.path.join(models_dir, name)
        joblib.dump(obj, path)
        print(f"Saved -> {path}")
        


#  load models
# ─────────────────────────────────────────────

def load_models(names: list, models_dir: str = "models") -> dict:
    """Load saved model 
    return dict """
    loaded = {}
    for name in names:
        path = os.path.join(models_dir, name)
        loaded[name] = joblib.load(path)
        print(f"Loaded ← {path}")
    return loaded

