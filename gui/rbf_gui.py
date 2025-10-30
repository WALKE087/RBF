import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from rbf.network import RBF_Network
from rbf.utils import normalize_data, one_hot_encode_X, label_encode_Y, align_features_to_model
from data.loader import load_local_dataset
from data.save_load import save_model, load_model
from plot.graphs import plot_train_results
from drive.drive_integration import GoogleDriveManager

class RBF_GUI:
    def test_trained_model(self):
        """Probar el modelo recién entrenado con los datos de prueba"""
        try:
            if self.X_data is None or self.Y_data is None:
                messagebox.showerror("Error", "Primero carga un dataset.")
                return
            if not hasattr(self.rbf, 'centers') or self.rbf.centers is None or not hasattr(self.rbf, 'weights') or self.rbf.weights is None:
                messagebox.showerror("Error", "Primero entrena el modelo.")
                return
            
            # Usar todos los datos para predicción
            Y_pred = self.rbf.predict(self.X_data)
            errors = self.Y_data - Y_pred
            error_general = np.mean(np.abs(errors))
            
            self.results_text.insert(tk.END, "\n" + "="*60 + "\n")
            self.results_text.insert(tk.END, "PRUEBA DEL MODELO ENTRENADO\n")
            self.results_text.insert(tk.END, "="*60 + "\n\n")
            self.results_text.insert(tk.END, f"Predicción sobre {len(self.X_data)} patrones:\n")
            self.results_text.insert(tk.END, "-"*60 + "\n")
            self.results_text.insert(tk.END, "Patrón    YD       YR       Error    |Error|\n")
            self.results_text.insert(tk.END, "-"*60 + "\n")
            for i in range(len(self.Y_data)):
                self.results_text.insert(tk.END, 
                    f"  {i+1}      {self.Y_data[i]:.4f}   {Y_pred[i]:.4f}   {errors[i]:7.4f}  {abs(errors[i]):.4f}\n")
            self.results_text.insert(tk.END, "\n")
            self.results_text.insert(tk.END, f"ERROR GENERAL: {error_general:.4f}\n")
            self.results_text.insert(tk.END, "="*60 + "\n")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo probar el modelo:\n{str(e)}")

    def apply_loaded_model(self):
        try:
            if self.X_data is None:
                messagebox.showerror("Error", "Primero carga un dataset para predecir.")
                return
            if not hasattr(self.rbf, 'centers') or self.rbf.centers is None or not hasattr(self.rbf, 'weights') or self.rbf.weights is None:
                messagebox.showerror("Error", "No hay modelo cargado/entrenado para predecir.")
                return
            # Ajustar datos a las características del modelo si es necesario
            X = self.X_data
            if hasattr(self, 'model_feature_names') and self.model_feature_names is not None:
                X = align_features_to_model(X, self.feature_names, self.model_feature_names)
            # Normalizar si el modelo fue entrenado con datos normalizados
            if hasattr(self, 'model_is_normalized') and self.model_is_normalized:
                X, _, _, _, _, _ = normalize_data(X, self.Y_data, self.model_data_min, self.model_data_max, self.model_y_min, self.model_y_max)
            Y_pred = self.rbf.predict(X)
            self.results_text.insert(tk.END, f"\nPredicción con modelo cargado:\n")
            self.results_text.insert(tk.END, f"Patrón    YR (Predicho)\n")
            self.results_text.insert(tk.END, "-"*30 + "\n")
            for i, y in enumerate(Y_pred):
                self.results_text.insert(tk.END, f"  {i+1}      {y:.4f}\n")
            self.results_text.insert(tk.END, f"\nSe generaron {len(Y_pred)} predicciones.\n")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo aplicar el modelo:\n{str(e)}")
    def drive_logout(self):
        import os
        if os.path.exists(self.google_token_path):
            try:
                os.remove(self.google_token_path)
                self.drive_manager.creds = None
                self.drive_manager.service = None
                messagebox.showinfo("Google Drive", "Sesión de Google Drive cerrada correctamente.")
            except Exception as e:
                messagebox.showerror("Google Drive", f"No se pudo cerrar la sesión: {e}")
        else:
            messagebox.showinfo("Google Drive", "No hay sesión activa para cerrar.")

    def save_model_drive(self):
        import io, pickle
        if self.rbf.weights is None or self.rbf.centers is None:
            messagebox.showerror("Error", "No hay modelo entrenado para guardar.")
            return
        if not self.drive_manager.ensure_drive_service():
            return
        model = {
            'centers': self.rbf.centers,
            'weights': self.rbf.weights,
            'num_centers': self.rbf.num_centers,
            'error_optimo': self.rbf.error_optimo,
            'feature_names': self.feature_names,
            'is_normalized': self.is_normalized,
            'data_min': self.data_min,
            'data_max': self.data_max,
            'y_min': self.y_min,
            'y_max': self.y_max,
            'target_mapping': self.target_mapping
        }
        import tkinter.simpledialog
        filename = tkinter.simpledialog.askstring("Guardar en Drive", "Nombre del archivo de modelo (ej: modelo.rbf.pkl):", initialvalue="modelo.rbf.pkl")
        if not filename:
            return
        file_bytes = io.BytesIO()
        pickle.dump(model, file_bytes)
        file_bytes.seek(0)
        try:
            drive_service = self.drive_manager.service
            from googleapiclient.http import MediaIoBaseUpload
            media = MediaIoBaseUpload(file_bytes, mimetype='application/octet-stream')
            file_metadata = {'name': filename}
            drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            messagebox.showinfo("Google Drive", f"Modelo guardado en Drive como '{filename}'")
        except Exception as e:
            messagebox.showerror("Google Drive", f"No se pudo guardar el modelo en Drive:\n{e}")

    def load_model_drive(self):
        if not self.drive_manager.ensure_drive_service():
            return
        picker = tk.Toplevel(self.root)
        picker.title("Seleccionar modelo RBF en Google Drive")
        picker.geometry("700x400")
        picker.configure(bg='#f0f0f0')
        tk.Label(picker, text="Modelos RBF en Drive (*.rbf.pkl)", font=('Arial', 11, 'bold'), bg='#f0f0f0').pack(pady=(10, 5))
        frame = tk.Frame(picker, bg='#f0f0f0')
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        cols = ('Nombre', 'MIME', 'Modificado')
        tree = ttk.Treeview(frame, columns=cols, show='headings')
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=200 if c == 'Nombre' else 160, anchor='w')
        vs = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=vs.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vs.pack(side=tk.RIGHT, fill=tk.Y)

        # Listar archivos .rbf.pkl en Drive
        files = self.drive_manager.list_drive_files(query_exts=(".rbf.pkl", ".pkl"))
        for f in files:
            tree.insert('', 'end', iid=f['id'], values=(f.get('name'), f.get('mimeType'), f.get('modifiedTime')))

        def on_load():
            sel = tree.selection()
            if not sel:
                messagebox.showwarning("Google Drive", "Selecciona un modelo de la lista.")
                return
            file_id = sel[0]
            meta = next((x for x in files if x['id'] == file_id), None)
            if not meta:
                messagebox.showerror("Google Drive", "No se encontraron metadatos del archivo seleccionado.")
                return
            content, mime = self.drive_manager.download_drive_file_bytes(file_id)
            if content is None:
                return
            
            import io, pickle
            try:
                model = pickle.load(io.BytesIO(content))
                self.rbf.centers = model.get('centers')
                self.rbf.weights = model.get('weights')
                self.rbf.num_centers = model.get('num_centers')
                self.rbf.error_optimo = model.get('error_optimo', self.error_optimo_var.get())
                self.model_feature_names = model.get('feature_names')
                self.model_is_normalized = model.get('is_normalized', True)
                self.model_data_min = model.get('data_min')
                self.model_data_max = model.get('data_max')
                self.model_y_min = model.get('y_min')
                self.model_y_max = model.get('y_max')
                self.model_target_mapping = model.get('target_mapping')
                picker.destroy()
                messagebox.showinfo("Éxito", f"Modelo '{meta.get('name')}' cargado correctamente desde Drive.")
            except Exception as e:
                messagebox.showerror("Google Drive", f"Error al cargar el modelo:\n{e}")

        btns = tk.Frame(picker, bg='#f0f0f0')
        btns.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Button(btns, text="Cancelar", command=picker.destroy,
                  bg='#7f8c8d', fg='white').pack(side=tk.RIGHT, padx=5)
        tk.Button(btns, text="Cargar Modelo", command=on_load,
                  bg='#2ecc71', fg='white').pack(side=tk.RIGHT, padx=5)

    def open_preprocessing_window(self):
        X = np.array([
            [1.0, 2.0],
            [1.5, 2.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ])
        Y = np.array([1.0, 1.0, 0.0, 0.0])
        self.X_data_original = X.copy()
        self.Y_data_original = Y.copy()
        self.feature_names = ["X1", "X2"]
        self.target_mapping = None
        # Los datos se cargan sin normalizar, usar preprocesamiento para escalar
        self.X_data = X
        self.Y_data = Y
        self.is_normalized = False
        self.update_table()
        min_centers = int(self.X_data.shape[1])
        max_centers = max(1, len(self.X_data))
        try:
            self.centers_spin.config(from_=min_centers, to=max_centers)
        except Exception:
            pass
        current = self.num_centers_var.get()
        self.num_centers_var.set(min(max_centers, max(current, min_centers)))
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "✓ Datos de ejemplo cargados exitosamente\n")
        self.results_text.insert(tk.END, f"  Patrones: {len(self.X_data)}\n")
        self.results_text.insert(tk.END, f"  Entradas: {self.X_data.shape[1]}\n")
        self.results_text.insert(tk.END, f"  Salidas: 1\n")
        self.results_text.insert(tk.END, f"  Normalizado: {'SÍ' if self.is_normalized else 'NO'}\n\n")

    def edit_data(self):
        edit_window = tk.Toplevel(self.root)
        edit_window.title("Editar Datos de Entrenamiento")
        edit_window.geometry("500x400")
        edit_window.configure(bg='#f0f0f0')
        tk.Label(edit_window, text="Editar Patrones de Entrenamiento", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack(pady=10)
        text_area = scrolledtext.ScrolledText(edit_window, font=('Courier', 10), height=15, width=50)
        text_area.pack(padx=10, pady=10)
        data_to_show = self.X_data_original if self.X_data_original is not None else self.X_data
        y_to_show = self.Y_data_original if self.Y_data_original is not None else self.Y_data
        if data_to_show is not None:
            text_area.insert(tk.END, "# Formato: X1 X2 ... Xn YD (un patrón por línea)\n")
            text_area.insert(tk.END, "# Los datos se muestran sin normalizar\n")
            for i in range(len(data_to_show)):
                xs = " ".join(str(v) for v in data_to_show[i])
                text_area.insert(tk.END, f"{xs} {y_to_show[i]}\n")
        def save_data():
            try:
                text = text_area.get(1.0, tk.END)
                lines = [l.strip() for l in text.split('\n') if l.strip() and not l.strip().startswith('#')]
                X_list = []
                Y_list = []
                for line in lines:
                    values = [float(v) for v in line.split()]
                    if len(values) >= 2:
                        X_list.append(values[:-1])
                        Y_list.append(values[-1])
                X = np.array(X_list)
                Y = np.array(Y_list)
                self.X_data_original = X
                self.Y_data_original = Y
                # Los datos se cargan sin normalizar, usar preprocesamiento para escalar
                self.X_data = X
                self.Y_data = Y
                self.is_normalized = False
                self.update_table()
                try:
                    min_centers = int(self.X_data.shape[1])
                    max_centers = max(1, len(self.X_data))
                    self.centers_spin.config(from_=min_centers, to=max_centers)
                    current = self.num_centers_var.get()
                    self.num_centers_var.set(min(max_centers, max(current, min_centers)))
                except Exception:
                    pass
                edit_window.destroy()
                messagebox.showinfo("Éxito", f"Datos actualizados: {len(self.X_data)} patrones")
            except Exception as e:
                messagebox.showerror("Error", f"Error al procesar datos:\n{str(e)}")
        tk.Button(edit_window, text="Guardar", command=save_data,
                 bg='#27ae60', fg='white', font=('Arial', 10, 'bold')).pack(pady=5)

    def open_preprocessing_window(self):
        """Ventana de preprocesamiento de datos"""
        from rbf.utils import handle_missing_values, get_basic_statistics, standardize_data
        import pandas as pd
        
        if self.X_data is None or self.Y_data is None:
            messagebox.showerror("Error", "Primero carga un dataset")
            return
        
        prep_window = tk.Toplevel(self.root)
        prep_window.title("Preprocesamiento de Datos")
        prep_window.geometry("900x700")
        prep_window.configure(bg='#f0f0f0')
        
        # Título
        tk.Label(prep_window, text="⚙️ PREPROCESAMIENTO DE DATOS",
                font=('Arial', 14, 'bold'), bg='#f0f0f0').pack(pady=10)
        
        # Notebook para diferentes secciones
        prep_notebook = ttk.Notebook(prep_window)
        prep_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # ==== TAB 1: Estadísticas Básicas ====
        stats_tab = tk.Frame(prep_notebook, bg='#f0f0f0')
        prep_notebook.add(stats_tab, text="📊 Estadísticas")
        
        stats_text = scrolledtext.ScrolledText(stats_tab, wrap=tk.WORD, font=('Courier', 9), height=30)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        def show_statistics():
            # Crear DataFrame temporal con todos los datos
            if hasattr(self, 'X_processed_df') and self.X_processed_df is not None:
                df_temp = self.X_processed_df.copy()
                df_temp['Target'] = self.y_series_loaded if hasattr(self, 'y_series_loaded') else self.Y_data_original
            else:
                cols = self.feature_names if self.feature_names else [f"X{i+1}" for i in range(self.X_data_original.shape[1])]
                df_temp = pd.DataFrame(self.X_data_original, columns=cols)
                df_temp['Target'] = self.Y_data_original
            
            stats = get_basic_statistics(df_temp)
            
            stats_text.delete(1.0, tk.END)
            stats_text.insert(tk.END, "="*70 + "\n")
            stats_text.insert(tk.END, "ESTADÍSTICAS BÁSICAS DEL DATASET\n")
            stats_text.insert(tk.END, "="*70 + "\n\n")
            stats_text.insert(tk.END, f"Dimensiones: {stats['shape'][0]} filas x {stats['shape'][1]} columnas\n\n")
            
            stats_text.insert(tk.END, "VALORES FALTANTES:\n")
            stats_text.insert(tk.END, "-"*70 + "\n")
            for col, missing in stats['missing_values'].items():
                pct = (missing / stats['shape'][0]) * 100 if stats['shape'][0] > 0 else 0
                stats_text.insert(tk.END, f"  {col}: {missing} ({pct:.2f}%)\n")
            
            stats_text.insert(tk.END, "\n" + "="*70 + "\n")
            stats_text.insert(tk.END, "ESTADÍSTICAS NUMÉRICAS:\n")
            stats_text.insert(tk.END, "="*70 + "\n")
            for col, col_stats in stats['numeric_stats'].items():
                stats_text.insert(tk.END, f"\n{col}:\n")
                stats_text.insert(tk.END, f"  Media:    {col_stats['mean']:.4f}\n")
                stats_text.insert(tk.END, f"  Std Dev:  {col_stats['std']:.4f}\n")
                stats_text.insert(tk.END, f"  Min:      {col_stats['min']:.4f}\n")
                stats_text.insert(tk.END, f"  Q25:      {col_stats['q25']:.4f}\n")
                stats_text.insert(tk.END, f"  Mediana:  {col_stats['median']:.4f}\n")
                stats_text.insert(tk.END, f"  Q75:      {col_stats['q75']:.4f}\n")
                stats_text.insert(tk.END, f"  Max:      {col_stats['max']:.4f}\n")
            
            if stats.get('categorical_info'):
                stats_text.insert(tk.END, "\n" + "="*70 + "\n")
                stats_text.insert(tk.END, "VARIABLES CATEGÓRICAS:\n")
                stats_text.insert(tk.END, "="*70 + "\n")
                for col, info in stats['categorical_info'].items():
                    stats_text.insert(tk.END, f"\n{col}:\n")
                    stats_text.insert(tk.END, f"  Valores únicos: {info['unique_values']}\n")
                    stats_text.insert(tk.END, f"  Top 5 valores:\n")
                    for val, count in info['top_values'].items():
                        stats_text.insert(tk.END, f"    {val}: {count}\n")
        
        tk.Button(stats_tab, text="📊 Calcular Estadísticas", command=show_statistics,
                 bg='#3498db', fg='white', font=('Arial', 10, 'bold')).pack(pady=5)
        
        # ==== TAB 2: Valores Faltantes ====
        missing_tab = tk.Frame(prep_notebook, bg='#f0f0f0')
        prep_notebook.add(missing_tab, text="🔍 Valores Faltantes")
        
        tk.Label(missing_tab, text="Estrategia para manejar valores faltantes:",
                font=('Arial', 10, 'bold'), bg='#f0f0f0').pack(pady=10)
        
        missing_strategy_var = tk.StringVar(value='mean')
        strategies = [
            ('Media (numéricos)', 'mean'),
            ('Mediana (numéricos)', 'median'),
            ('Moda (todos)', 'mode'),
            ('Eliminar filas con NaN', 'drop_rows'),
            ('Eliminar columnas con >50% NaN', 'drop_cols')
        ]
        
        for text, value in strategies:
            tk.Radiobutton(missing_tab, text=text, variable=missing_strategy_var,
                          value=value, bg='#f0f0f0', font=('Arial', 9)).pack(anchor='w', padx=30)
        
        missing_result_text = scrolledtext.ScrolledText(missing_tab, wrap=tk.WORD, font=('Courier', 9), height=15)
        missing_result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        def apply_missing_strategy():
            strategy = missing_strategy_var.get()
            # Crear DataFrame temporal
            if hasattr(self, 'X_processed_df') and self.X_processed_df is not None:
                df_temp = self.X_processed_df.copy()
                df_temp['Target'] = self.y_series_loaded if hasattr(self, 'y_series_loaded') else self.Y_data_original
            else:
                cols = self.feature_names if self.feature_names else [f"X{i+1}" for i in range(self.X_data_original.shape[1])]
                df_temp = pd.DataFrame(self.X_data_original, columns=cols)
                df_temp['Target'] = self.Y_data_original
            
            df_clean, missing_stats = handle_missing_values(df_temp, strategy=strategy)
            
            missing_result_text.delete(1.0, tk.END)
            missing_result_text.insert(tk.END, f"Estrategia aplicada: {strategy}\n\n")
            missing_result_text.insert(tk.END, f"Valores faltantes totales: {missing_stats['total_missing']}\n\n")
            
            if 'rows_dropped' in missing_stats:
                missing_result_text.insert(tk.END, f"Filas eliminadas: {missing_stats['rows_dropped']}\n")
            if 'columns_dropped' in missing_stats:
                missing_result_text.insert(tk.END, f"Columnas eliminadas: {missing_stats['columns_dropped']}\n")
            
            # Actualizar datos - convertir a float explícitamente
            if len(df_clean) > 0:
                y_col = 'Target'
                X_cols = [c for c in df_clean.columns if c != y_col]
                
                # Convertir a numérico asegurando tipo float
                self.X_data_original = df_clean[X_cols].astype(float).to_numpy()
                self.Y_data_original = df_clean[y_col].astype(float).to_numpy()
                self.X_data = self.X_data_original.copy()
                self.Y_data = self.Y_data_original.copy()
                self.update_table()
                missing_result_text.insert(tk.END, f"\n✓ Datos actualizados: {len(self.X_data)} patrones\n")
            else:
                messagebox.showwarning("Advertencia", "No quedaron datos después de aplicar la estrategia.")
        
        tk.Button(missing_tab, text="Aplicar Estrategia", command=apply_missing_strategy,
                 bg='#27ae60', fg='white', font=('Arial', 10, 'bold')).pack(pady=5)
        
        # ==== TAB 3: Escalado de Datos ====
        scale_tab = tk.Frame(prep_notebook, bg='#f0f0f0')
        prep_notebook.add(scale_tab, text="📐 Escalado")
        
        tk.Label(scale_tab, text="Método de escalado:",
                font=('Arial', 10, 'bold'), bg='#f0f0f0').pack(pady=10)
        
        scale_method_var = tk.StringVar(value='normalize')
        
        tk.Radiobutton(scale_tab, text="Normalización Min-Max (0-1)",
                      variable=scale_method_var, value='normalize',
                      bg='#f0f0f0', font=('Arial', 9)).pack(anchor='w', padx=30)
        tk.Radiobutton(scale_tab, text="Estandarización Z-score (media=0, std=1)",
                      variable=scale_method_var, value='standardize',
                      bg='#f0f0f0', font=('Arial', 9)).pack(anchor='w', padx=30)
        tk.Radiobutton(scale_tab, text="Sin escalado (datos originales)",
                      variable=scale_method_var, value='none',
                      bg='#f0f0f0', font=('Arial', 9)).pack(anchor='w', padx=30)
        
        scale_result_text = scrolledtext.ScrolledText(scale_tab, wrap=tk.WORD, font=('Courier', 9), height=15)
        scale_result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        def apply_scaling():
            method = scale_method_var.get()
            scale_result_text.delete(1.0, tk.END)
            
            if method == 'normalize':
                self.X_data, self.Y_data, self.data_min, self.data_max, self.y_min, self.y_max = normalize_data(
                    self.X_data_original, self.Y_data_original)
                self.is_normalized = True
                scale_result_text.insert(tk.END, "✓ Normalización Min-Max aplicada\n\n")
                scale_result_text.insert(tk.END, f"Rango X: [0, 1]\n")
                scale_result_text.insert(tk.END, f"Rango Y: [0, 1]\n")
            elif method == 'standardize':
                self.X_data, self.Y_data, self.data_min, self.data_max, self.y_min, self.y_max = standardize_data(
                    self.X_data_original, self.Y_data_original)
                self.is_normalized = True
                scale_result_text.insert(tk.END, "✓ Estandarización Z-score aplicada\n\n")
                scale_result_text.insert(tk.END, f"Media X ≈ 0, Std X ≈ 1\n")
                scale_result_text.insert(tk.END, f"Media Y ≈ 0, Std Y ≈ 1\n")
            else:
                self.X_data = self.X_data_original.copy()
                self.Y_data = self.Y_data_original.copy()
                self.is_normalized = False
                scale_result_text.insert(tk.END, "✓ Usando datos originales (sin escalado)\n")
            
            self.update_table()
            scale_result_text.insert(tk.END, f"\n✓ Datos actualizados en la tabla principal\n")
        
        tk.Button(scale_tab, text="Aplicar Escalado", command=apply_scaling,
                 bg='#9b59b6', fg='white', font=('Arial', 10, 'bold')).pack(pady=5)
        
        # Botón cerrar
        tk.Button(prep_window, text="Cerrar", command=prep_window.destroy,
                 bg='#7f8c8d', fg='white', font=('Arial', 10, 'bold')).pack(pady=10)

    def google_sign_in(self):
        if self.drive_manager.ensure_drive_service():
            messagebox.showinfo("Google Drive", "Sesión iniciada correctamente. Ya puedes abrir archivos desde Drive.")

    def open_from_drive(self):
        if not self.drive_manager.ensure_drive_service():
            return
        picker = tk.Toplevel(self.root)
        picker.title("Seleccionar archivo de Google Drive")
        picker.geometry("700x400")
        picker.configure(bg='#f0f0f0')
        tk.Label(picker, text="Archivos recientes en Drive (csv/xlsx/xls/json/txt)", font=('Arial', 11, 'bold'), bg='#f0f0f0').pack(pady=(10, 5))
        frame = tk.Frame(picker, bg='#f0f0f0')
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        cols = ('Nombre', 'MIME', 'Modificado')
        tree = ttk.Treeview(frame, columns=cols, show='headings')
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=200 if c == 'Nombre' else 160, anchor='w')
        vs = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=vs.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        files = self.drive_manager.list_drive_files()
        for f in files:
            tree.insert('', 'end', iid=f['id'], values=(f.get('name'), f.get('mimeType'), f.get('modifiedTime')))
        def on_open():
            sel = tree.selection()
            if not sel:
                messagebox.showwarning("Google Drive", "Selecciona un archivo de la lista.")
                return
            file_id = sel[0]
            meta = next((x for x in files if x['id'] == file_id), None)
            if not meta:
                messagebox.showerror("Google Drive", "No se encontraron metadatos del archivo seleccionado.")
                return
            content, mime = self.drive_manager.download_drive_file_bytes(file_id)
            if content is None:
                return
            name = meta.get('name', '')
            import io
            import pandas as pd
            try:
                df = None
                lower = name.lower()
                if lower.endswith('.csv') or (mime and 'csv' in mime):
                    df = pd.read_csv(io.BytesIO(content))
                elif lower.endswith(('.xlsx', '.xls')) or (mime and 'spreadsheet' in mime):
                    df = pd.read_excel(io.BytesIO(content))
                elif lower.endswith('.json') or (mime and 'json' in mime):
                    try:
                        df = pd.read_json(io.BytesIO(content))
                    except ValueError:
                        df = pd.read_json(io.BytesIO(content), lines=True)
                elif lower.endswith('.txt') or (mime and 'text/plain' in mime):
                    try:
                        df = pd.read_csv(io.BytesIO(content))
                    except Exception:
                        df = pd.read_csv(io.BytesIO(content), delim_whitespace=True, header=None)
                else:
                    messagebox.showerror("Google Drive", f"Tipo de archivo no soportado: {name}")
                    return
                self._process_loaded_dataframe(df, source_hint=f"Drive: {name}")
                picker.destroy()
            except Exception as e:
                messagebox.showerror("Google Drive", f"Error al leer el archivo seleccionado:\n{e}")
        btns = tk.Frame(picker, bg='#f0f0f0')
        btns.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Button(btns, text="Cancelar", command=picker.destroy,
                  bg='#7f8c8d', fg='white').pack(side=tk.RIGHT, padx=5)
        tk.Button(btns, text="Descargar y Cargar", command=on_open,
                  bg='#2ecc71', fg='white').pack(side=tk.RIGHT, padx=5)

    def _process_loaded_dataframe(self, df, source_hint=""):
        if df.shape[1] < 2:
            messagebox.showerror("Error", "El dataset debe tener al menos 2 columnas (entradas y salida).")
            return
        X_df = df.iloc[:, :-1]
        y_series = df.iloc[:, -1]
        X_processed, self.feature_names = one_hot_encode_X(X_df)
        self.X_processed_df = X_processed.copy()
        X = X_processed.to_numpy(dtype=float)
        Y, self.target_mapping = label_encode_Y(y_series)
        self.y_series_loaded = y_series.copy()
        self.X_data_original = X.copy()
        self.Y_data_original = Y.copy()
        # Los datos se cargan sin normalizar, usar preprocesamiento para escalar
        self.X_data = X
        self.Y_data = Y
        self.is_normalized = False
        min_centers = int(self.X_data.shape[1])
        max_centers = max(1, len(self.X_data))
        try:
            self.centers_spin.config(from_=min_centers, to=max_centers)
        except Exception:
            pass
        current = self.num_centers_var.get()
        self.num_centers_var.set(min(max_centers, max(current, min_centers)))
        self.update_table()
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"✓ Dataset cargado exitosamente ({source_hint})\n")
        self.results_text.insert(tk.END, f"  Patrones: {len(self.X_data)}\n")
        self.results_text.insert(tk.END, f"  Entradas (tras encoding): {self.X_data.shape[1]}\n")
        self.results_text.insert(tk.END, f"  Salidas: 1\n")
        self.results_text.insert(tk.END, f"  Normalizado: {'SÍ' if self.is_normalized else 'NO'}\n")
        if self.target_mapping is not None:
            self.results_text.insert(tk.END, "  Mapeo de clases Y (categórica):\n")
            for k, v in self.target_mapping.items():
                self.results_text.insert(tk.END, f"    {k} -> {v}\n")
        self.results_text.insert(tk.END, "\n")
    def load_dataset(self):
        import pandas as pd
        from data.loader import load_local_dataset
        df = load_local_dataset()
        if df is None:
            return
        if df.shape[1] < 2:
            messagebox.showerror("Error", "El dataset debe tener al menos 2 columnas (entradas y salida)")
            return
        X_df = df.iloc[:, :-1]
        y_series = df.iloc[:, -1]
        X_processed, self.feature_names = one_hot_encode_X(X_df)
        self.X_processed_df = X_processed.copy()
        X = X_processed.to_numpy(dtype=float)
        Y, self.target_mapping = label_encode_Y(y_series)
        self.y_series_loaded = y_series.copy()
        self.X_data_original = X.copy()
        self.Y_data_original = Y.copy()
        # Los datos se cargan sin normalizar, usar preprocesamiento para escalar
        self.X_data = X
        self.Y_data = Y
        self.is_normalized = False
        min_centers = int(self.X_data.shape[1])
        max_centers = max(1, len(self.X_data))
        try:
            self.centers_spin.config(from_=min_centers, to=max_centers)
        except Exception:
            pass
        current = self.num_centers_var.get()
        self.num_centers_var.set(min(max_centers, max(current, min_centers)))
        self.update_table()
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"✓ Dataset cargado exitosamente\n")
        self.results_text.insert(tk.END, f"  Patrones: {len(self.X_data)}\n")
        self.results_text.insert(tk.END, f"  Entradas (tras encoding): {self.X_data.shape[1]}\n")
        self.results_text.insert(tk.END, f"  Salidas: 1\n")
        self.results_text.insert(tk.END, f"  Normalizado: {'SÍ' if self.is_normalized else 'NO'}\n")
        if self.target_mapping is not None:
            self.results_text.insert(tk.END, "  Mapeo de clases Y (categórica):\n")
            for k, v in self.target_mapping.items():
                self.results_text.insert(tk.END, f"    {k} -> {v}\n")
        self.results_text.insert(tk.END, "\n")

    def save_model(self):
        if self.rbf.weights is None or self.rbf.centers is None:
            messagebox.showerror("Error", "No hay modelo entrenado para guardar.")
            return
        model = {
            'centers': self.rbf.centers,
            'weights': self.rbf.weights,
            'num_centers': self.rbf.num_centers,
            'error_optimo': self.rbf.error_optimo,
            'feature_names': self.feature_names,
            'is_normalized': self.is_normalized,
            'data_min': self.data_min,
            'data_max': self.data_max,
            'y_min': self.y_min,
            'y_max': self.y_max,
            'target_mapping': self.target_mapping
        }
        from data.save_load import save_model as save_model_fn
        save_model_fn(model)

    def load_model(self):
        from data.save_load import load_model as load_model_fn
        model = load_model_fn()
        if not model:
            return
        self.rbf.centers = model.get('centers')
        self.rbf.weights = model.get('weights')
        self.rbf.num_centers = model.get('num_centers')
        self.rbf.error_optimo = model.get('error_optimo', self.error_optimo_var.get())
        self.model_feature_names = model.get('feature_names')
        self.model_is_normalized = model.get('is_normalized', True)
        self.model_data_min = model.get('data_min')
        self.model_data_max = model.get('data_max')
        self.model_y_min = model.get('y_min')
        self.model_y_max = model.get('y_max')
        self.model_target_mapping = model.get('target_mapping')
        messagebox.showinfo("Éxito", "Modelo cargado correctamente. Ya puedes predecir sobre el dataset cargado.")

    def update_table(self):
        for col in self.tree['columns']:
            self.tree.heading(col, text='')
        self.tree['columns'] = ()
        self.tree.delete(*self.tree.get_children())
        if self.X_data is None or self.Y_data is None:
            return
        n_inputs = self.X_data.shape[1]
        if self.feature_names is not None and len(self.feature_names) == n_inputs:
            columns = list(self.feature_names) + ["YD"]
        else:
            columns = [f"X{i+1}" for i in range(n_inputs)] + ["YD"]
        self.tree['columns'] = columns
        for col in columns:
            if col == "YD":
                self.tree.heading(col, text="YD (Deseada)")
            else:
                self.tree.heading(col, text=col)
            self.tree.column(col, width=80, minwidth=60, stretch=False, anchor='center')  # Ancho fijo sin stretch
        for i in range(len(self.X_data)):
            row = [f"{self.X_data[i, j]:.4f}" for j in range(n_inputs)] + [f"{self.Y_data[i]:.4f}"]
            self.tree.insert('', 'end', values=row)

    def train_network(self):
        if self.X_data is None or self.Y_data is None:
            messagebox.showerror("Error", "No hay datos cargados")
            return
        
        # Si ya hay un modelo entrenado, preguntar si desea reentrenar
        if self.rbf.centers is not None and self.rbf.weights is not None:
            response = messagebox.askyesno("Reentrenar", 
                "Ya existe un modelo entrenado. ¿Desea reentrenarlo desde cero?")
            if not response:
                return
        
        num_centers = self.num_centers_var.get()
        num_inputs = int(self.X_data.shape[1])
        if num_centers < num_inputs:
            messagebox.showerror("Error", f"El número de centros radiales no puede ser menor al número de entradas ({num_inputs}).")
            return
        n = len(self.X_data)
        idx = np.arange(n)
        np.random.shuffle(idx)
        train_pct = self.train_pct_var.get()
        split = int(n * (train_pct / 100.0))
        train_idx, test_idx = idx[:split], idx[split:]
        X_train, Y_train = self.X_data[train_idx], self.Y_data[train_idx]
        X_test, Y_test = self.X_data[test_idx], self.Y_data[test_idx]
        self.results_text.delete(1.0, tk.END)
        error_optimo = self.error_optimo_var.get()
        self.results_text.insert(tk.END, "="*60 + "\n")
        self.results_text.insert(tk.END, f"ENTRENAMIENTO DE RED NEURONAL RBF ({train_pct}% train, {100-train_pct}% test)\n")
        self.results_text.insert(tk.END, "="*60 + "\n\n")
        self.results_text.insert(tk.END, f"DATOS:\n")
        self.results_text.insert(tk.END, f"  • Patrones totales: {n}\n")
        self.results_text.insert(tk.END, f"  • Patrones de entrenamiento: {len(X_train)}\n")
        self.results_text.insert(tk.END, f"  • Patrones de prueba: {len(X_test)}\n")
        self.results_text.insert(tk.END, f"  • Datos normalizados: {'SÍ (Min-Max 0-1)' if self.is_normalized else 'NO (datos originales)'}\n\n")
        self.results_text.insert(tk.END, f"CONFIGURACIÓN:\n")
        self.results_text.insert(tk.END, f"  • Número de centros radiales: {num_centers}\n")
        self.results_text.insert(tk.END, f"  • Error de aproximación óptimo: {error_optimo}\n")
        self.results_text.insert(tk.END, f"  • Función de activación: FA = Ω² * ln(Ω)\n\n")
        Y_pred_train, errors_train, error_general_train, A_matrix = self.rbf.train(
            X_train, Y_train, num_centers, error_optimo
        )
        self.results_text.insert(tk.END, f"CENTROS RADIALES INICIALIZADOS:\n")
        for i, center in enumerate(self.rbf.centers):
            values = ", ".join([f"{c:.4f}" for c in center])
            self.results_text.insert(tk.END, f"  R{i+1} = ({values})\n")
        self.results_text.insert(tk.END, "\n")
        self.results_text.insert(tk.END, f"PESOS CALCULADOS (W = A\\Y):\n")
        for i, weight in enumerate(self.rbf.weights):
            self.results_text.insert(tk.END, f"  W{i} = {weight:.4f}\n")
        self.results_text.insert(tk.END, "\n")
        self.results_text.insert(tk.END, f"RESULTADOS SOBRE DATOS DE ENTRENAMIENTO:\n")
        self.results_text.insert(tk.END, "-"*60 + "\n")
        self.results_text.insert(tk.END, "Patrón    YD       YR       Error    |Error|\n")
        self.results_text.insert(tk.END, "-"*60 + "\n")
        for i in range(len(Y_train)):
            self.results_text.insert(tk.END, 
                f"  {i+1}      {Y_train[i]:.4f}   {Y_pred_train[i]:.4f}   {errors_train[i]:7.4f}  {abs(errors_train[i]):.4f}\n")
        self.results_text.insert(tk.END, "\n")
        self.results_text.insert(tk.END, f"ERROR GENERAL (EG) TRAIN: {error_general_train:.4f}\n")
        self.results_text.insert(tk.END, f"ERROR ÓPTIMO:             {error_optimo:.4f}\n\n")
        if error_general_train <= error_optimo:
            self.results_text.insert(tk.END, "✓ LA RED CONVERGE (EG ≤ Error Óptimo)\n", 'success')
            self.results_text.tag_config('success', foreground='green', font=('Courier', 9, 'bold'))
            if len(X_test) > 0:
                Y_pred_test = self.rbf.predict(X_test)
                errors_test = Y_test - Y_pred_test
                error_general_test = np.mean(np.abs(errors_test))
                threshold = 0.5
                error_rate = np.mean(np.abs(errors_test) > threshold) * 100 if len(errors_test) > 0 else 0
                self.results_text.insert(tk.END, f"RESULTADOS SOBRE DATOS DE PRUEBA:\n")
                self.results_text.insert(tk.END, "-"*60 + "\n")
                self.results_text.insert(tk.END, "Patrón    YD       YR       Error    |Error|\n")
                self.results_text.insert(tk.END, "-"*60 + "\n")
                for i in range(len(Y_test)):
                    self.results_text.insert(tk.END, 
                        f"  {i+1}      {Y_test[i]:.4f}   {Y_pred_test[i]:.4f}   {errors_test[i]:7.4f}  {abs(errors_test[i]):.4f}\n")
                self.results_text.insert(tk.END, "\n")
                self.results_text.insert(tk.END, f"ERROR GENERAL (EG) TEST: {error_general_test:.4f}\n")
                self.results_text.insert(tk.END, f"TASA DE ERROR (>|{threshold}|): {error_rate:.2f}%\n")
            else:
                self.results_text.insert(tk.END, "No hay suficientes datos para prueba (test set vacío).\n")
        else:
            self.results_text.insert(tk.END, "✗ LA RED NO CONVERGE (EG > Error Óptimo)\n", 'warning')
            self.results_text.insert(tk.END, "  Sugerencia: Aumentar el número de centros radiales\n", 'warning')
            self.results_text.tag_config('warning', foreground='red', font=('Courier', 9, 'bold'))
            self.results_text.insert(tk.END, "\nNo se realiza prueba sobre datos de test porque la red no converge.\n")
        self.update_graphs(Y_pred_train, errors_train, error_general_train, error_optimo)

    def update_graphs(self, Y_pred, errors, error_general, error_optimo):
        plot_train_results(self.ax1, self.ax2, self.rbf.Y_train, Y_pred, errors, error_general, error_optimo)
    def __init__(self, root):
        self.root = root
        self.root.title("Red Neuronal de Funciones de Base Radial (RBF)")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        self.rbf = RBF_Network()
        # Variables de estado
        self.X_data = None
        self.Y_data = None
        self.X_data_original = None
        self.Y_data_original = None
        self.is_normalized = False
        self.data_min = None
        self.data_max = None
        self.feature_names = None
        self.target_mapping = None
        self.y_min = None
        self.y_max = None
        self.X_processed_df = None
        self.y_series_loaded = None
        self.model_feature_names = None
        self.model_is_normalized = None
        self.model_data_min = None
        self.model_data_max = None
        self.model_y_min = None
        self.model_y_max = None
        self.model_target_mapping = None
        # Estado de Google Drive
        self.google_client_secrets = os.path.join(os.path.dirname(__file__), '../SecretJsonGoogle.json')
        self.google_token_path = os.path.join(os.path.dirname(__file__), '../token.json')
        self.drive_manager = GoogleDriveManager(self.google_client_secrets, self.google_token_path)
        self.create_widgets()

    def create_widgets(self):
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # --- Tab 1: Google Drive (Solo Login/Logout) ---
        page_drive = tk.Frame(notebook, bg='#f0f0f0')
        notebook.add(page_drive, text="1. Google Drive")
        
        drive_label = tk.Label(page_drive, text="Gestión de Sesión de Google Drive",
                              font=('Arial', 14, 'bold'), bg='#f0f0f0')
        drive_label.pack(pady=20)
        
        tk.Label(page_drive, text="Inicia sesión para acceder a tus archivos en Google Drive",
                font=('Arial', 10), bg='#f0f0f0', fg='#555').pack(pady=5)
        
        btns_drive = tk.Frame(page_drive, bg='#f0f0f0')
        btns_drive.pack(pady=20)
        
        tk.Button(btns_drive, text="🔐 Iniciar Sesión Drive", command=self.google_sign_in,
                  bg='#1abc9c', fg='white', font=('Arial', 12, 'bold'), 
                  width=20, height=2).pack(pady=10)
        tk.Button(btns_drive, text="� Cerrar Sesión Drive", command=self.drive_logout,
                  bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'),
                  width=20, height=2).pack(pady=10)

        # --- Tab 2: Modelo y Entrenamiento ---
        page_train = tk.Frame(notebook, bg='#f0f0f0')
        notebook.add(page_train, text="2. Modelo y Entrenamiento")

        # Contenedor principal con ancho fijo para ambas columnas
        main_container = tk.Frame(page_train, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Dividir en 2 columnas con ancho relativo fijo (50% cada una)
        main_container.grid_columnconfigure(0, weight=1, uniform="group1")  # Columna izquierda
        main_container.grid_columnconfigure(1, weight=1, uniform="group1")  # Columna derecha
        main_container.grid_rowconfigure(0, weight=1)
        
        left_column = tk.Frame(main_container, bg='#f0f0f0')
        left_column.grid(row=0, column=0, sticky='nsew', padx=(0, 2))
        
        right_column = tk.Frame(main_container, bg='#f0f0f0')
        right_column.grid(row=0, column=1, sticky='nsew', padx=(2, 0))

        # ========== COLUMNA IZQUIERDA: Carga, Config y Entrenamiento ==========
        
        # Gestión de modelos
        model_label = tk.Label(left_column, text="Gestión de Modelos",
                              font=('Arial', 11, 'bold'), bg='#f0f0f0')
        model_label.pack(pady=5)
        btns_model = tk.Frame(left_column, bg='#f0f0f0')
        btns_model.pack(pady=3)
        tk.Button(btns_model, text="☁️ Cargar (Drive)", command=self.load_model_drive,
                  bg='#16a085', fg='white', font=('Arial', 8, 'bold')).pack(side=tk.LEFT, padx=2)
        tk.Button(btns_model, text="☁️ Guardar (Drive)", command=self.save_model_drive,
                  bg='#2980b9', fg='white', font=('Arial', 8, 'bold')).pack(side=tk.LEFT, padx=2)
        tk.Button(btns_model, text="📥 Cargar", command=self.load_model,
                  bg='#34495e', fg='white', font=('Arial', 8, 'bold')).pack(side=tk.LEFT, padx=2)
        tk.Button(btns_model, text="💾 Guardar", command=self.save_model,
                  bg='#8e44ad', fg='white', font=('Arial', 8, 'bold')).pack(side=tk.LEFT, padx=2)
        tk.Button(btns_model, text="🔮 Predecir", command=self.apply_loaded_model,
                  bg='#d35400', fg='white', font=('Arial', 8, 'bold')).pack(side=tk.LEFT, padx=2)

        # Separador
        ttk.Separator(left_column, orient='horizontal').pack(fill=tk.X, pady=8)

        # Dataset y edición
        data_label = tk.Label(left_column, text="Datos",
                              font=('Arial', 11, 'bold'), bg='#f0f0f0')
        data_label.pack(pady=3)
        btns = tk.Frame(left_column, bg='#f0f0f0')
        btns.pack(pady=3)
        tk.Button(btns, text="📂 Local", command=self.load_dataset,
                  bg='#2ecc71', fg='white', font=('Arial', 8, 'bold')).pack(side=tk.LEFT, padx=2)
        tk.Button(btns, text="☁️ Drive", command=self.open_from_drive,
                  bg='#16a085', fg='white', font=('Arial', 8, 'bold')).pack(side=tk.LEFT, padx=2)
        tk.Button(btns, text="Editar", command=self.edit_data,
                  bg='#9b59b6', fg='white', font=('Arial', 8)).pack(side=tk.LEFT, padx=2)
        
        # Botón de preprocesamiento
        tk.Button(left_column, text="⚙️ PREPROCESAMIENTO", command=self.open_preprocessing_window,
                  bg='#3498db', fg='white', font=('Arial', 9, 'bold'), height=1).pack(fill=tk.X, pady=5)

        # Tabla de datos
        tree_container = tk.Frame(left_column, bg='#f0f0f0', height=150)
        tree_container.pack(fill=tk.X, pady=5)
        tree_container.pack_propagate(False)  # Evita que el frame crezca en altura
        self.tree = ttk.Treeview(tree_container, show='headings', height=6)
        vsb = ttk.Scrollbar(tree_container, orient='vertical', command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_container, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky='ew')  # Cambio: sticky 'ew' en vez de 'nsew'
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        tree_container.rowconfigure(0, weight=0)  # Cambio: weight=0
        tree_container.columnconfigure(0, weight=1)

        # Configuración de red
        config_frame = tk.LabelFrame(left_column, text="⚙ Configuración",
                                     font=('Arial', 9, 'bold'), bg='#f0f0f0', padx=5, pady=5)
        config_frame.pack(fill=tk.X, pady=5)

        tk.Label(config_frame, text="Centros:",
                 bg='#f0f0f0', font=('Arial', 8)).grid(row=0, column=0, sticky='w', pady=2)
        self.num_centers_var = tk.IntVar(value=2)
        self.centers_spin = tk.Spinbox(
            config_frame, from_=1, to=10, textvariable=self.num_centers_var,
            width=10, font=('Arial', 8))
        self.centers_spin.grid(row=0, column=1, sticky='w', padx=5, pady=2)

        tk.Label(config_frame, text="Error Óptimo:",
                 bg='#f0f0f0', font=('Arial', 8)).grid(row=1, column=0, sticky='w', pady=2)
        self.error_optimo_var = tk.DoubleVar(value=0.1)
        tk.Entry(config_frame, textvariable=self.error_optimo_var,
                 width=10, font=('Arial', 8)).grid(row=1, column=1, sticky='w', padx=5, pady=2)

        tk.Label(config_frame, text="% Train:",
                 bg='#f0f0f0', font=('Arial', 8)).grid(row=2, column=0, sticky='w', pady=2)
        self.train_pct_var = tk.IntVar(value=80)
        self.train_pct_spin = tk.Spinbox(
            config_frame, from_=70, to=95, textvariable=self.train_pct_var,
            width=10, font=('Arial', 8))
        self.train_pct_spin.grid(row=2, column=1, sticky='w', padx=5, pady=2)

        train_btn = tk.Button(left_column, text="🚀 ENTRENAR RED RBF",
            command=self.train_network,
            bg='#27ae60', fg='white', font=('Arial', 11, 'bold'),
            height=2, cursor='hand2')
        train_btn.pack(fill=tk.X, pady=8)

        # Botón para probar modelo entrenado
        test_btn = tk.Button(left_column, text="🧪 PROBAR MODELO ENTRENADO",
            command=self.test_trained_model,
            bg='#e67e22', fg='white', font=('Arial', 10, 'bold'),
            height=1, cursor='hand2')
        test_btn.pack(fill=tk.X, pady=(0, 8))

        # Resultados de texto
        results_frame = tk.LabelFrame(left_column, text="Resultados",
                                      font=('Arial', 9, 'bold'), bg='#f0f0f0', padx=5, pady=5)
        results_frame.pack(fill=tk.BOTH, expand=True)
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD,
                                                      font=('Courier', 7), height=12)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # ========== COLUMNA DERECHA: Gráficas ==========
        
        tk.Label(right_column, text="Gráficas de Entrenamiento",
                font=('Arial', 11, 'bold'), bg='#f0f0f0').pack(pady=5)

        graph1_frame = tk.LabelFrame(right_column, text="Salidas Deseadas vs Salidas de la Red",
                                     font=('Arial', 9, 'bold'), bg='#f0f0f0', padx=5, pady=5)
        graph1_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.fig1 = Figure(figsize=(6, 4), dpi=90)
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=graph1_frame)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        graph2_frame = tk.LabelFrame(right_column, text="Análisis de Errores",
                                     font=('Arial', 9, 'bold'), bg='#f0f0f0', padx=5, pady=5)
        graph2_frame.pack(fill=tk.BOTH, expand=True)
        self.fig2 = Figure(figsize=(6, 4), dpi=90)
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=graph2_frame)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
