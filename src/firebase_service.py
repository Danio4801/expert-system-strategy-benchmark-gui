




import os
import sys
from typing import Optional, Dict, List


def resource_path(relative_path):



    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class FirebaseService:


    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not FirebaseService._initialized:
            self.db = None
            self.bucket = None
            self.current_user = None
            print("[INFO] Firebase is disabled in this build (offline mode)")
            FirebaseService._initialized = True

    def initialize_firebase(self):
        pass

    def create_user(self, username: str, password: str) -> bool:
        return False

    def login(self, username: str, password: str) -> bool:
        return False

    def logout(self):
        self.current_user = None

    def validate_session(self, user_id: str) -> bool:
        return False

    def auto_login(self, user_id: str) -> bool:
        return False

    def is_logged_in(self) -> bool:
        return False

    def get_current_user(self) -> Optional[Dict]:
        return None

    def get_all_users(self) -> list:
        return []

    def delete_user(self, username: str) -> bool:
        return False

    def upload_file(self, local_file_path: str, remote_filename: str = None) -> bool:
        return False

    def file_exists(self, filename: str, username: str = None) -> bool:
        return False

    def list_user_files(self, username: str = None) -> List[Dict]:
        return []

    def delete_file(self, file_id: str) -> bool:
        return False

    def download_file(self, file_id: str, destination_path: str) -> bool:
        return False

    def upload_experiment_folder(self, local_folder_path, run_id: str = None) -> bool:
        return False
