import os
import numpy as np
import pandas as pd

def attempt_load_feature_from_cache(cache_dir: str, name: str) -> tuple[np.ndarray | None, bool]:
    if not os.path.exists(os.path.abspath(cache_dir)):
        os.mkdir(os.path.abspath(cache_dir))

    target = os.path.join(os.path.abspath(cache_dir), name)
    
    if os.path.isfile(target):
        data = pd.read_csv(target).to_numpy()
        return (data, True)
    else:
        return (None, False)
    
def save_feature_to_cache(cache_dir: str, name: str, data: np.ndarray) -> None:
    if not os.path.exists(os.path.abspath(cache_dir)):
        os.mkdir(os.path.abspath(cache_dir))

    target = os.path.join(os.path.abspath(cache_dir), name)

    df = pd.DataFrame(data)
    df.to_csv(target)
