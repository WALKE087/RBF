import pickle
from tkinter import filedialog, messagebox

def save_model(model):
    file_path = filedialog.asksaveasfilename(
        title="Guardar Modelo",
        defaultextension=".rbf.pkl",
        filetypes=[("Modelo RBF", "*.rbf.pkl"), ("Todos", "*.*")]
    )
    if not file_path:
        return
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        messagebox.showinfo("Éxito", "Modelo guardado correctamente.")
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo guardar el modelo:\n{str(e)}")

def load_model():
    file_path = filedialog.askopenfilename(
        title="Cargar Modelo",
        filetypes=[("Modelo RBF", "*.rbf.pkl"), ("Todos", "*.*")]
    )
    if not file_path:
        return None
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{str(e)}")
        return None
