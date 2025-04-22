import os
import numpy as np
import pandas as pd

CACHE_DIR = "./cache"

def ensure_cache_dir_exists(cache_dir: str):
    if not os.path.exists(os.path.abspath(cache_dir)):
        os.mkdir(os.path.abspath(cache_dir))

def attempt_load_feature_from_cache(name: str, cache_dir: str=CACHE_DIR) -> tuple[np.ndarray | None, bool]:
    ensure_cache_dir_exists(cache_dir)

    target = os.path.join(os.path.abspath(cache_dir), name)
    
    if os.path.isfile(target):
        data = pd.read_csv(target).to_numpy()
        return (data, True)
    else:
        return (None, False)
    
def save_feature_to_cache(name: str, data: np.ndarray, cache_dir: str=CACHE_DIR) -> None:
    ensure_cache_dir_exists(cache_dir)

    target = os.path.join(os.path.abspath(cache_dir), name)

    df = pd.DataFrame(data)
    df.to_csv(target)

def attempt_load_dataframe(name: str, cache_dir: str=CACHE_DIR) -> tuple[pd.DataFrame, bool]:
    ensure_cache_dir_exists(cache_dir)

    target = os.path.join(os.path.abspath(cache_dir), name + ".csv")

    if os.path.isfile(target):
        df = pd.read_csv(target)
        return (df, True)
    else:
        return (None, False)
    
def save_dataframe_to_cache(name: str, df: pd.DataFrame, cache_dir: str=CACHE_DIR) -> None:
    ensure_cache_dir_exists(cache_dir)
    
    target = os.path.join(os.path.abspath(cache_dir), name + ".csv")

    df.to_csv(target)
