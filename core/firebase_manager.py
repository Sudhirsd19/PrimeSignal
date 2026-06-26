import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

class FirebaseManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirebaseManager, cls).__new__(cls)
            cls._instance._init_firebase()
        return cls._instance

    def _init_firebase(self):
        self.db = None
        self.is_connected = False
        
        # Try to load credentials from environment variable (Render)
        cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
        if cred_json:
            try:
                cred_dict = json.loads(cred_json)
                cred = credentials.Certificate(cred_dict)
                if not firebase_admin._apps:
                    firebase_admin.initialize_app(cred)
                self.db = firestore.client()
                self.is_connected = True
                print("[FIREBASE] Connected via environment variable.")
                return
            except Exception as e:
                print(f"[FIREBASE] Failed to load from env var: {e}")

        # Try to load from local file
        cred_path = "firebase_credentials.json"
        if os.path.exists(cred_path):
            try:
                cred = credentials.Certificate(cred_path)
                if not firebase_admin._apps:
                    firebase_admin.initialize_app(cred)
                self.db = firestore.client()
                self.is_connected = True
                print("[FIREBASE] Connected via local credentials file.")
                return
            except Exception as e:
                print(f"[FIREBASE] Failed to load from local file: {e}")

        print("[FIREBASE] No credentials found. Running in offline/local mode.")
