import firebase_admin
from firebase_admin import credentials, firestore, storage
import pandas as pd
import os
import json
import requests
from datetime import datetime

class FirebaseConnector:
    def __init__(self, key_path):






        if not os.path.exists(key_path):
            raise FileNotFoundError(f"Firebase key file not found at: {key_path}")


        try:
            with open(key_path, 'r') as f:
                key_data = json.load(f)
                self.project_id = key_data.get('project_id')
        except Exception as e:
            raise ValueError(f"Failed to read project_id from key file: {e}")


        self.bucket_name = "projektsystemekspertowy.firebasestorage.app"


        if not firebase_admin._apps:
            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred, {
                'storageBucket': self.bucket_name
            })
            print(f"Firebase initialized for project: {self.project_id}")
        else:
            print("Firebase app already initialized.")

        self.db = firestore.client()
        self.bucket = storage.bucket(self.bucket_name)

    def upload_experiments(self, csv_path, collection_name='experiments', run_id=None):








        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")

        print(f"Reading CSV from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        records = df.to_dict(orient='records')
        collection_ref = self.db.collection(collection_name)

        print(f"Uploading {len(records)} experiment records to Firestore...")
        
        batch = self.db.batch()
        count = 0
        total_uploaded = 0
        BATCH_LIMIT = 400
        

        upload_timestamp = datetime.now().isoformat()

        for record in records:
            doc_ref = collection_ref.document()
            

            clean_record = {k: v for k, v in record.items() if pd.notna(v)}
            

            clean_record['uploaded_at'] = upload_timestamp
            if run_id:
                clean_record['run_id'] = run_id
            
            batch.set(doc_ref, clean_record)
            count += 1
            
            if count >= BATCH_LIMIT:
                batch.commit()
                print(f"Committed batch of {count} records.")
                total_uploaded += count
                batch = self.db.batch()
                count = 0

        if count > 0:
            batch.commit()
            total_uploaded += count
            print(f"Committed final batch of {count} records.")

        print(f"Successfully uploaded {total_uploaded} records to Firestore.")


        if run_id:
            filename = os.path.basename(csv_path)
            blob_path = f"experiments/{run_id}/data/{filename}"
            blob = self.bucket.blob(blob_path)

            print(f"Uploading CSV to Storage: {blob_path}...")
            blob.upload_from_filename(csv_path)
            blob.make_public()

            print(f"CSV uploaded to Storage. Public URL: {blob.public_url}")

    def upload_image(self, image_path, destination_blob_name=None, run_id=None):











        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}, skipping.")
            return None


        if run_id:
            filename = os.path.basename(image_path)
            blob_path = f"experiments/{run_id}/visualizations/{filename}"
        elif destination_blob_name:
            blob_path = destination_blob_name
        else:

            filename = os.path.basename(image_path)
            blob_path = f"visualizations/{filename}"

        blob = self.bucket.blob(blob_path)
        
        print(f"Uploading {image_path} to {blob_path}...")
        blob.upload_from_filename(image_path)
        

        blob.make_public()
        
        print(f"File uploaded. Public URL: {blob.public_url}")
        return blob.public_url

    def upload_log(self, log_path, run_id):













        if not run_id:
            print("Warning: run_id is required for log upload. Skipping.")
            return []

        import glob

        uploaded_urls = []


        if os.path.isfile(log_path):
            log_files = [log_path]

        elif os.path.isdir(log_path):
            pattern = os.path.join(log_path, f"inference_{run_id}*.log")
            log_files = glob.glob(pattern)
            if not log_files:
                print(f"Warning: No log files found matching pattern {pattern}")
                return []
        else:
            print(f"Warning: Log path {log_path} is neither file nor directory. Skipping.")
            return []

        print(f"Found {len(log_files)} log file(s) to upload for run_id: {run_id}")


        for log_file in log_files:
            filename = os.path.basename(log_file)
            blob_path = f"experiments/{run_id}/logs/{filename}"

            blob = self.bucket.blob(blob_path)

            print(f"Uploading {filename} to {blob_path}...")
            blob.upload_from_filename(log_file)


            blob.make_public()

            print(f"  ✓ Uploaded: {blob.public_url}")
            uploaded_urls.append(blob.public_url)

        print(f"Successfully uploaded {len(uploaded_urls)} log file(s).")
        return uploaded_urls

    def check_connection(self):






        status = {
            "internet": False,
            "firestore": False,
            "storage": False,
            "bucket_name": self.bucket_name
        }


        try:

            response = requests.get("http://clients3.google.com/generate_204", timeout=3)
            status["internet"] = response.status_code == 204
        except Exception as e:
            print(f"Diagnostic - Internet check failed: {e}")
            status["internet"] = False


        try:

            collections = self.db.collections()

            for _ in collections:
                break
            status["firestore"] = True
        except Exception as e:
            print(f"Diagnostic - Firestore check failed: {e}")
            status["firestore"] = False


        try:
            status["storage"] = self.bucket.exists()
        except Exception as e:
            print(f"Diagnostic - Storage check failed: {e}")
            status["storage"] = False

        return status