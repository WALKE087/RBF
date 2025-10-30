import pandas as pd
from tkinter import filedialog, messagebox

def load_local_dataset():
    file_path = filedialog.askopenfilename(
        title="Seleccionar Dataset",
        filetypes=[
            ("Archivos CSV", "*.csv"),
            ("Archivos Excel", "*.xlsx *.xls"),
            ("Archivos JSON", "*.json"),
            ("Archivos de texto", "*.txt"),
            ("Todos los archivos", "*.*")
        ]
    )
    if not file_path:
        return None
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.txt'):
            df = pd.read_csv(file_path, delimiter='\t')
        else:
            messagebox.showerror("Error", "Formato de archivo no soportado")
            return None
        return df
    except Exception as e:
        messagebox.showerror("Error", f"Error al cargar el archivo:\n{str(e)}")
        return None
