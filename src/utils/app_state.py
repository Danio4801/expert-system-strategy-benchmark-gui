








import json
import os
from typing import Optional, Dict, List
from datetime import datetime


class AppStateManager:


    _instance = None
    _initialized = False

    def __new__(cls):

        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):

        if not AppStateManager._initialized:
            self.state_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'app_state.json'
            )
            self.state = self._load()
            AppStateManager._initialized = True

    def _load(self) -> Dict:






        default_state = {
            'keep_logged_in': False,
            'last_user': None,
            'recent_files': []
        }

        if not os.path.exists(self.state_file):
            return default_state

        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                loaded_state = json.load(f)


            return {**default_state, **loaded_state}

        except Exception as e:
            print(f"[WARNING] Nie można wczytać app_state.json: {e}")
            return default_state

    def _save(self):

        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] Nie można zapisać app_state.json: {e}")



    def get_keep_logged_in(self) -> bool:

        return self.state.get('keep_logged_in', False)

    def get_last_user(self) -> Optional[Dict]:






        return self.state.get('last_user')

    def set_logged_in(self, username: str, user_id: str, keep_logged_in: bool = False):








        self.state['keep_logged_in'] = keep_logged_in
        self.state['last_user'] = {
            'username': username,
            'user_id': user_id
        }
        self._save()
        print(f"[STATE] Zapisano logowanie: {username} (keep_logged_in={keep_logged_in})")

    def clear_login(self):

        self.state['keep_logged_in'] = False
        self.state['last_user'] = None
        self._save()
        print("[STATE] Wyczyszczono stan logowania")



    def get_recent_files(self) -> List[Dict]:






        return self.state.get('recent_files', [])

    def add_recent_file(self, file_path: str):











        file_name = os.path.basename(file_path)


        self.state['recent_files'] = [
            f for f in self.state['recent_files']
            if f['path'] != file_path
        ]


        new_file = {
            'name': file_name,
            'path': file_path,
            'last_used': datetime.now().isoformat()
        }

        self.state['recent_files'].insert(0, new_file)


        self.state['recent_files'] = self.state['recent_files'][:5]

        self._save()
        print(f"[STATE] Dodano do historii: {file_name}")

    def remove_recent_file(self, file_path: str):






        self.state['recent_files'] = [
            f for f in self.state['recent_files']
            if f['path'] != file_path
        ]
        self._save()
        print(f"[STATE] Usunięto z historii: {file_path}")

    def validate_recent_files(self) -> List[str]:






        removed = []


        valid_files = []
        for file_data in self.state['recent_files']:
            if os.path.exists(file_data['path']):
                valid_files.append(file_data)
            else:
                removed.append(file_data['path'])
                print(f"[STATE] Plik nie istnieje (usunięto z historii): {file_data['path']}")


        self.state['recent_files'] = valid_files
        if removed:
            self._save()

        return removed



    def print_state(self):

        print("\n=== APP STATE ===")
        print(json.dumps(self.state, indent=2, ensure_ascii=False))
        print("=================\n")



if __name__ == "__main__":

    state = AppStateManager()


    state.set_logged_in("TestUser", "abc123", keep_logged_in=True)


    state.add_recent_file("C:\\Users\\Daniel\\Desktop\\iris.csv")
    state.add_recent_file("D:\\Data\\wine.csv")


    state.print_state()


    removed = state.validate_recent_files()
    print(f"Usunięto nieistniejące: {removed}")
