import os
import io
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from tkinter import messagebox

SCOPES = [
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/drive.appdata',
    'https://www.googleapis.com/auth/drive.metadata',
    'https://www.googleapis.com/auth/drive.metadata.readonly',
    'https://www.googleapis.com/auth/drive',
]

class GoogleDriveManager:
    def __init__(self, client_secrets_path, token_path):
        self.client_secrets_path = client_secrets_path
        self.token_path = token_path
        self.creds = None
        self.service = None

    def ensure_drive_service(self):
        try:
            creds = None
            if os.path.exists(self.token_path):
                creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                    except Exception:
                        creds = None
                if not creds or not creds.valid:
                    if not os.path.exists(self.client_secrets_path):
                        raise FileNotFoundError(
                            f"No se encontró el archivo de credenciales: {self.client_secrets_path}\n"
                            f"Coloca tu SecretJsonGoogle.json en la misma carpeta que main.py."
                        )
                    flow = InstalledAppFlow.from_client_secrets_file(self.client_secrets_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                with open(self.token_path, 'w') as token:
                    token.write(creds.to_json())
            service = build('drive', 'v3', credentials=creds)
            self.creds = creds
            self.service = service
            return True
        except Exception as e:
            messagebox.showerror("Google Drive", f"No se pudo iniciar sesión en Drive:\n{e}")
            return False

    def list_drive_files(self, query_exts=(".csv", ".xlsx", ".xls", ".json", ".txt"), page_size=100):
        if not self.service and not self.ensure_drive_service():
            return []
        service = self.service
        q = "trashed = false"
        fields = "nextPageToken, files(id, name, mimeType, modifiedTime, size)"
        files = []
        page_token = None
        try:
            while True and len(files) < page_size:
                resp = service.files().list(q=q,
                                            spaces='drive',
                                            fields=fields,
                                            orderBy='modifiedTime desc',
                                            pageSize=min(100, page_size - len(files)),
                                            pageToken=page_token).execute()
                batch = resp.get('files', [])
                for f in batch:
                    name = f.get('name', '').lower()
                    if any(name.endswith(ext) for ext in query_exts):
                        files.append(f)
                page_token = resp.get('nextPageToken')
                if not page_token or len(files) >= page_size:
                    break
        except Exception as e:
            messagebox.showerror("Google Drive", f"Error al listar archivos:\n{e}")
            return []
        return files

    def download_drive_file_bytes(self, file_id):
        if not self.service and not self.ensure_drive_service():
            return None, None
        try:
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.seek(0)
            meta = self.service.files().get(fileId=file_id, fields='mimeType').execute()
            return fh.read(), meta.get('mimeType')
        except Exception as e:
            messagebox.showerror("Google Drive", f"No se pudo descargar el archivo:\n{e}")
            return None, None
