import os
import shutil
from pandas import read_csv, DataFrame

class LocalStorageService:

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def file_exists(self, filename: str) -> bool:
        return os.path.exists(os.path.join(self.base_dir, filename))

    def save_model(self, source_path: str, dest_filename: str):
        dest_path = os.path.join(self.base_dir, dest_filename)
        shutil.copy2(source_path, dest_path)
        return dest_path

    def load_model(self, filename: str):
        import pickle
        file_path = os.path.join(self.base_dir, filename)
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def read_csv(self, filename: str) -> DataFrame:
        file_path = os.path.join(self.base_dir, filename)
        return read_csv(file_path)
