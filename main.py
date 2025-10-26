import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class RBF_Network:
    """Red Neuronal de Funciones de Base Radial (RBF)"""
    
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.num_centers = 2
        self.centers = None
        self.weights = None
        self.error_optimo = 0.1
        self.training_history = []
        
    def rbf_activation(self, distance):
        """Función de Activación RBF: FA = Ω² * ln(Ω)"""
        if distance == 0:
            return 0
        return (distance ** 2) * np.log(distance)
    
    def euclidean_distance(self, x, center):
        """Calcula la distancia euclidiana entre un vector y un centro"""
        return np.sqrt(np.sum((x - center) ** 2))
    
    def initialize_centers(self, X):
        """Inicializa los centros radiales aleatoriamente entre min y max de los datos"""
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        
        self.centers = np.random.uniform(min_vals, max_vals, (self.num_centers, X.shape[1]))
        return self.centers
    
    def train(self, X, Y, num_centers, error_optimo):
        """Entrena la red RBF"""
        self.X_train = X
        self.Y_train = Y
        self.num_centers = num_centers
        self.error_optimo = error_optimo
        
        # Inicializar centros radiales
        self.initialize_centers(X)
        
        # Construir matriz de interpolación A
        num_patterns = X.shape[0]
        A = np.ones((num_patterns, num_centers + 1))  # +1 para el bias W0
        
        # Calcular distancias y aplicar función de activación
        for i in range(num_patterns):
            for j in range(num_centers):
                distance = self.euclidean_distance(X[i], self.centers[j])
                A[i, j + 1] = self.rbf_activation(distance)
        
        # Resolver el sistema A * W = Y usando mínimos cuadrados
        self.weights = np.linalg.lstsq(A, Y, rcond=None)[0]
        
        # Calcular salidas de la red
        Y_pred = A.dot(self.weights)
        
        # Calcular errores
        errors = Y - Y_pred
        error_general = np.mean(np.abs(errors))
        
        # Guardar historial
        self.training_history.append({
            'num_centers': num_centers,
            'centers': self.centers.copy(),
            'weights': self.weights.copy(),
            'Y_pred': Y_pred.copy(),
            'errors': errors.copy(),
            'error_general': error_general,
            'A_matrix': A.copy()
        })
        
        return Y_pred, errors, error_general, A
    
    def predict(self, X):
        """Realiza predicciones con la red entrenada"""
        if self.weights is None:
            raise ValueError("La red no ha sido entrenada")
        
        num_patterns = X.shape[0]
        A = np.ones((num_patterns, self.num_centers + 1))
        
        for i in range(num_patterns):
            for j in range(self.num_centers):
                distance = self.euclidean_distance(X[i], self.centers[j])
                A[i, j + 1] = self.rbf_activation(distance)
        
        return A.dot(self.weights)


class RBF_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Red Neuronal de Funciones de Base Radial (RBF)")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Crear red neuronal
        self.rbf = RBF_Network()
        
        # Variables
        self.X_data = None
        self.Y_data = None
        self.X_data_original = None  # Datos sin normalizar
        self.Y_data_original = None
        self.is_normalized = False
        self.data_min = None
        self.data_max = None
        
        # Crear interfaz
        self.create_widgets()
        
        # Cargar datos de ejemplo
        self.load_example_data()
    
    def create_widgets(self):
        # Frame principal dividido en izquierda y derecha
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel izquierdo - Controles y datos
        left_frame = tk.Frame(main_frame, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Panel derecho - Gráficas
        right_frame = tk.Frame(main_frame, bg='#f0f0f0')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # ===== PANEL IZQUIERDO =====
        
        # Título
        title_label = tk.Label(left_frame, text="RED NEURONAL RBF", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=(0, 10))
        
        # Frame de datos de entrada
        data_frame = tk.LabelFrame(left_frame, text="Datos de Entrada", 
                                   font=('Arial', 10, 'bold'), bg='#f0f0f0', padx=10, pady=10)
        data_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))
        
        # Tabla de datos
        self.tree = ttk.Treeview(data_frame, columns=('X1', 'X2', 'YD'), 
                                show='headings', height=5)
        self.tree.heading('X1', text='X1')
        self.tree.heading('X2', text='X2')
        self.tree.heading('YD', text='YD (Deseada)')
        
        for col in ('X1', 'X2', 'YD'):
            self.tree.column(col, width=100, anchor='center')
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Botones de datos
        btn_frame = tk.Frame(data_frame, bg='#f0f0f0')
        btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        tk.Button(btn_frame, text="📂 Cargar Dataset", command=self.load_dataset,
                 bg='#2ecc71', fg='white', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Cargar Ejemplo", command=self.load_example_data,
                 bg='#3498db', fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Editar Datos", command=self.edit_data,
                 bg='#9b59b6', fg='white', font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        
        # Frame de configuración
        config_frame = tk.LabelFrame(left_frame, text="Configuración de la Red", 
                                    font=('Arial', 10, 'bold'), bg='#f0f0f0', padx=10, pady=10)
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Normalización
        self.normalize_var = tk.BooleanVar(value=True)
        tk.Checkbutton(config_frame, text="Normalizar Datos (Min-Max 0-1)", 
                      variable=self.normalize_var, bg='#f0f0f0', 
                      font=('Arial', 9, 'bold'), command=self.toggle_normalization).grid(
                      row=0, column=0, columnspan=2, sticky='w', pady=5)
        
        # Número de centros radiales
        tk.Label(config_frame, text="Número de Centros Radiales:", 
                bg='#f0f0f0', font=('Arial', 9)).grid(row=1, column=0, sticky='w', pady=5)
        self.num_centers_var = tk.IntVar(value=2)
        tk.Spinbox(config_frame, from_=1, to=10, textvariable=self.num_centers_var,
                  width=10, font=('Arial', 9)).grid(row=1, column=1, sticky='w', padx=(10, 0))
        
        # Error de aproximación óptimo
        tk.Label(config_frame, text="Error de Aproximación Óptimo:", 
                bg='#f0f0f0', font=('Arial', 9)).grid(row=2, column=0, sticky='w', pady=5)
        self.error_optimo_var = tk.DoubleVar(value=0.1)
        tk.Entry(config_frame, textvariable=self.error_optimo_var, 
                width=10, font=('Arial', 9)).grid(row=2, column=1, sticky='w', padx=(10, 0))
        
        # Botón de entrenamiento
        train_btn = tk.Button(left_frame, text="🚀 ENTRENAR RED RBF", 
                             command=self.train_network,
                             bg='#27ae60', fg='white', font=('Arial', 12, 'bold'),
                             height=2, cursor='hand2')
        train_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Frame de resultados
        results_frame = tk.LabelFrame(left_frame, text="Resultados del Entrenamiento", 
                                     font=('Arial', 10, 'bold'), bg='#f0f0f0', padx=10, pady=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Área de texto para resultados
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, 
                                                     font=('Courier', 9), height=15)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # ===== PANEL DERECHO - GRÁFICAS =====
        
        # Frame superior para gráfica YD vs YR
        graph1_frame = tk.LabelFrame(right_frame, text="Salidas Deseadas vs Salidas de la Red", 
                                    font=('Arial', 10, 'bold'), bg='#f0f0f0', padx=5, pady=5)
        graph1_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.fig1 = Figure(figsize=(6, 4), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=graph1_frame)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Frame inferior para gráfica de errores
        graph2_frame = tk.LabelFrame(right_frame, text="Error General vs Error de Aproximación Óptimo", 
                                    font=('Arial', 10, 'bold'), bg='#f0f0f0', padx=5, pady=5)
        graph2_frame.pack(fill=tk.BOTH, expand=True)
        
        self.fig2 = Figure(figsize=(6, 4), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=graph2_frame)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def normalize_data(self, X, Y):
        """Normaliza los datos usando Min-Max scaling (0-1) solo para columnas numéricas"""
        import pandas as pd
        # Guardar datos originales
        self.X_data_original = X.copy()
        self.Y_data_original = Y.copy()

        # Si X es DataFrame, convertir a numpy solo con columnas numéricas
        if isinstance(X, pd.DataFrame):
            X_num = X.select_dtypes(include=[float, int]).to_numpy()
        else:
            X_num = X

        self.data_min = np.min(X_num, axis=0)
        self.data_max = np.max(X_num, axis=0)

        # Normalizar X
        X_normalized = np.zeros_like(X_num, dtype=float)
        for i in range(X_num.shape[1]):
            min_val = self.data_min[i]
            max_val = self.data_max[i]
            if max_val - min_val != 0:
                X_normalized[:, i] = (X_num[:, i] - min_val) / (max_val - min_val)
            else:
                X_normalized[:, i] = 0.0

        # Normalizar Y si tiene rango
        y_min = np.min(Y)
        y_max = np.max(Y)
        if y_max - y_min != 0:
            Y_normalized = (Y - y_min) / (y_max - y_min)
        else:
            Y_normalized = np.zeros_like(Y, dtype=float)

        self.is_normalized = True
        return X_normalized, Y_normalized
    
    def toggle_normalization(self):
        """Activa/desactiva la normalización"""
        if self.X_data_original is not None:
            if self.normalize_var.get():
                self.X_data, self.Y_data = self.normalize_data(self.X_data_original, self.Y_data_original)
                self.update_table()
                self.results_text.insert(tk.END, "\n✓ Datos normalizados (Min-Max 0-1)\n")
            else:
                self.X_data = self.X_data_original.copy()
                self.Y_data = self.Y_data_original.copy()
                self.is_normalized = False
                self.update_table()
                self.results_text.insert(tk.END, "\n✓ Usando datos originales (sin normalizar)\n")
    
    def update_table(self):
        """Actualiza la tabla con los datos actuales"""
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for i in range(len(self.X_data)):
            self.tree.insert('', 'end', values=(f"{self.X_data[i, 0]:.4f}", 
                                               f"{self.X_data[i, 1]:.4f}", 
                                               f"{self.Y_data[i]:.4f}"))
    
    def load_dataset(self):
        """Carga un dataset desde archivo"""
        import pandas as pd
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
            return
        try:
            # Leer archivo según extensión
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
                return

            # Validar que tenga al menos 2 columnas (X y Y)
            if df.shape[1] < 2:
                messagebox.showerror("Error", "El dataset debe tener al menos 2 columnas (entradas y salida)")
                return

            # Seleccionar solo columnas numéricas para X
            X = df.select_dtypes(include=[float, int]).to_numpy()
            # Si la última columna es la salida y es no numérica, convertirla a códigos
            if not pd.api.types.is_numeric_dtype(df.iloc[:, -1]):
                Y = df.iloc[:, -1].astype('category').cat.codes.to_numpy()
            else:
                Y = df.iloc[:, -1].to_numpy(dtype=float)

            # Guardar originales
            self.X_data_original = X.copy()
            self.Y_data_original = Y.copy()

            # Normalizar si corresponde
            if self.normalize_var.get():
                self.X_data, self.Y_data = self.normalize_data(X, Y)
            else:
                self.X_data = X
                self.Y_data = Y
                self.is_normalized = False

            # Ajustar el mínimo de centros radiales
            min_centers = self.X_data.shape[1]
            self.num_centers_var.set(min_centers)
            # Actualizar el Spinbox para reflejar el nuevo mínimo
            for widget in self.root.winfo_children():
                for child in widget.winfo_children():
                    if isinstance(child, tk.Spinbox):
                        child.config(from_=min_centers)

            self.update_table()
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"✓ Dataset cargado exitosamente\n")
            self.results_text.insert(tk.END, f"  Patrones: {len(self.X_data)}\n")
            self.results_text.insert(tk.END, f"  Entradas: {self.X_data.shape[1]}\n")
            self.results_text.insert(tk.END, f"  Salidas: 1\n")
            self.results_text.insert(tk.END, f"  Normalizado: {'SÍ' if self.is_normalized else 'NO'}\n\n")
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar el archivo:\n{str(e)}")
    
    def load_example_data(self):
        """Carga los datos de ejemplo del problema"""
        X = np.array([
            [1.0, 2.0],
            [1.5, 2.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ])
        
        Y = np.array([1.0, 1.0, 0.0, 0.0])
        
        # Guardar como originales
        self.X_data_original = X.copy()
        self.Y_data_original = Y.copy()
        
        # Aplicar normalización si está activada
        if self.normalize_var.get():
            self.X_data, self.Y_data = self.normalize_data(X, Y)
        else:
            self.X_data = X
            self.Y_data = Y
            self.is_normalized = False
        
        # Actualizar tabla
        self.update_table()
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "✓ Datos de ejemplo cargados exitosamente\n")
        self.results_text.insert(tk.END, f"  Patrones: {len(self.X_data)}\n")
        self.results_text.insert(tk.END, f"  Entradas: {self.X_data.shape[1]}\n")
        self.results_text.insert(tk.END, f"  Salidas: 1\n")
        self.results_text.insert(tk.END, f"  Normalizado: {'SÍ' if self.is_normalized else 'NO'}\n\n")
    
    def edit_data(self):
        """Abre ventana para editar datos manualmente"""
        edit_window = tk.Toplevel(self.root)
        edit_window.title("Editar Datos de Entrenamiento")
        edit_window.geometry("500x400")
        edit_window.configure(bg='#f0f0f0')
        
        tk.Label(edit_window, text="Editar Patrones de Entrenamiento", 
                font=('Arial', 12, 'bold'), bg='#f0f0f0').pack(pady=10)
        
        text_area = scrolledtext.ScrolledText(edit_window, font=('Courier', 10), 
                                             height=15, width=50)
        text_area.pack(padx=10, pady=10)
        
        # Cargar datos ORIGINALES (sin normalizar)
        data_to_show = self.X_data_original if self.X_data_original is not None else self.X_data
        y_to_show = self.Y_data_original if self.Y_data_original is not None else self.Y_data
        
        if data_to_show is not None:
            text_area.insert(tk.END, "# Formato: X1 X2 YD (un patrón por línea)\n")
            text_area.insert(tk.END, "# Los datos se muestran sin normalizar\n")
            for i in range(len(data_to_show)):
                text_area.insert(tk.END, f"{data_to_show[i, 0]} {data_to_show[i, 1]} {y_to_show[i]}\n")
        
        def save_data():
            try:
                text = text_area.get(1.0, tk.END)
                lines = [l.strip() for l in text.split('\n') if l.strip() and not l.strip().startswith('#')]
                
                X_list = []
                Y_list = []
                
                for line in lines:
                    values = [float(v) for v in line.split()]
                    if len(values) >= 3:
                        X_list.append(values[:-1])  # Todas menos la última
                        Y_list.append(values[-1])    # La última es Y
                
                X = np.array(X_list)
                Y = np.array(Y_list)
                
                # Guardar como originales
                self.X_data_original = X
                self.Y_data_original = Y
                
                # Aplicar normalización si está activada
                if self.normalize_var.get():
                    self.X_data, self.Y_data = self.normalize_data(X, Y)
                else:
                    self.X_data = X
                    self.Y_data = Y
                    self.is_normalized = False
                
                self.update_table()
                edit_window.destroy()
                messagebox.showinfo("Éxito", f"Datos actualizados: {len(self.X_data)} patrones")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al procesar datos:\n{str(e)}")
        
        tk.Button(edit_window, text="Guardar", command=save_data,
                 bg='#27ae60', fg='white', font=('Arial', 10, 'bold')).pack(pady=5)
    
    def train_network(self):
        """Entrena la red neuronal RBF"""
        if self.X_data is None or self.Y_data is None:
            messagebox.showerror("Error", "No hay datos cargados")
            return
        num_inputs = self.X_data.shape[1]
        num_centers = self.num_centers_var.get()
        if num_centers < num_inputs:
            messagebox.showerror("Error", f"El número de centros radiales no puede ser menor al número de entradas ({num_inputs}).")
            return
        # Limpiar resultados anteriores
        self.results_text.delete(1.0, tk.END)
        # Obtener parámetros
        error_optimo = self.error_optimo_var.get()
        # Entrenar
        self.results_text.insert(tk.END, "="*60 + "\n")
        self.results_text.insert(tk.END, "ENTRENAMIENTO DE RED NEURONAL RBF\n")
        self.results_text.insert(tk.END, "="*60 + "\n\n")
        self.results_text.insert(tk.END, f"DATOS:\n")
        self.results_text.insert(tk.END, f"  • Patrones de entrenamiento: {len(self.X_data)}\n")
        self.results_text.insert(tk.END, f"  • Datos normalizados: {'SÍ (Min-Max 0-1)' if self.is_normalized else 'NO (datos originales)'}\n\n")
        self.results_text.insert(tk.END, f"CONFIGURACIÓN:\n")
        self.results_text.insert(tk.END, f"  • Número de centros radiales: {num_centers}\n")
        self.results_text.insert(tk.END, f"  • Error de aproximación óptimo: {error_optimo}\n")
        self.results_text.insert(tk.END, f"  • Función de activación: FA = Ω² * ln(Ω)\n\n")
        # Entrenar la red
        Y_pred, errors, error_general, A_matrix = self.rbf.train(
            self.X_data, self.Y_data, num_centers, error_optimo
        )
        
        # Mostrar centros radiales
        self.results_text.insert(tk.END, f"CENTROS RADIALES INICIALIZADOS:\n")
        for i, center in enumerate(self.rbf.centers):
            self.results_text.insert(tk.END, f"  R{i+1} = ({center[0]:.4f}, {center[1]:.4f})\n")
        self.results_text.insert(tk.END, "\n")
        
        # Mostrar cálculo de distancias y activaciones
        self.results_text.insert(tk.END, f"CÁLCULO DE DISTANCIAS EUCLIDIANAS Y ACTIVACIONES:\n")
        self.results_text.insert(tk.END, "-"*60 + "\n")
        
        for i in range(len(self.X_data)):
            self.results_text.insert(tk.END, f"Patrón {i+1}: X = ({self.X_data[i, 0]:.2f}, {self.X_data[i, 1]:.2f}), YD = {self.Y_data[i]:.2f}\n")
            for j in range(num_centers):
                dist = self.rbf.euclidean_distance(self.X_data[i], self.rbf.centers[j])
                activation = self.rbf.rbf_activation(dist)
                self.results_text.insert(tk.END, 
                    f"  D{j+1}{i+1} = {dist:.4f} → FA(D{j+1}{i+1}) = {activation:.4f}\n")
            self.results_text.insert(tk.END, "\n")
        
        # Mostrar matriz de interpolación
        self.results_text.insert(tk.END, f"MATRIZ DE INTERPOLACIÓN A:\n")
        self.results_text.insert(tk.END, "-"*60 + "\n")
        for row in A_matrix:
            self.results_text.insert(tk.END, "  " + "  ".join([f"{val:8.4f}" for val in row]) + "\n")
        self.results_text.insert(tk.END, "\n")
        
        # Mostrar pesos
        self.results_text.insert(tk.END, f"PESOS CALCULADOS (W = A\\Y):\n")
        for i, weight in enumerate(self.rbf.weights):
            self.results_text.insert(tk.END, f"  W{i} = {weight:.4f}\n")
        self.results_text.insert(tk.END, "\n")
        
        # Mostrar resultados
        self.results_text.insert(tk.END, f"SIMULACIÓN - SALIDAS DE LA RED:\n")
        self.results_text.insert(tk.END, "-"*60 + "\n")
        self.results_text.insert(tk.END, "Patrón    YD       YR       Error    |Error|\n")
        self.results_text.insert(tk.END, "-"*60 + "\n")
        
        for i in range(len(self.Y_data)):
            self.results_text.insert(tk.END, 
                f"  {i+1}      {self.Y_data[i]:.4f}   {Y_pred[i]:.4f}   {errors[i]:7.4f}  {abs(errors[i]):.4f}\n")
        
        self.results_text.insert(tk.END, "\n")
        self.results_text.insert(tk.END, f"ERROR GENERAL (EG): {error_general:.4f}\n")
        self.results_text.insert(tk.END, f"ERROR ÓPTIMO:       {error_optimo:.4f}\n\n")
        
        # Verificar convergencia
        if error_general <= error_optimo:
            self.results_text.insert(tk.END, "✓ LA RED CONVERGE (EG ≤ Error Óptimo)\n", 'success')
            self.results_text.tag_config('success', foreground='green', font=('Courier', 9, 'bold'))
        else:
            self.results_text.insert(tk.END, "✗ LA RED NO CONVERGE (EG > Error Óptimo)\n", 'warning')
            self.results_text.insert(tk.END, "  Sugerencia: Aumentar el número de centros radiales\n", 'warning')
            self.results_text.tag_config('warning', foreground='red', font=('Courier', 9, 'bold'))
        
        # Actualizar gráficas
        self.update_graphs(Y_pred, errors, error_general, error_optimo)
    
    def update_graphs(self, Y_pred, errors, error_general, error_optimo):
        """Actualiza las gráficas con los resultados"""
        
        # Gráfica 1: YD vs YR
        self.ax1.clear()
        patterns = np.arange(1, len(self.Y_data) + 1)
        
        self.ax1.plot(patterns, self.Y_data, 'o-', label='YD (Deseada)', 
                     color='#3498db', linewidth=2, markersize=8)
        self.ax1.plot(patterns, Y_pred, 's-', label='YR (Red)', 
                     color='#e74c3c', linewidth=2, markersize=8)
        
        self.ax1.set_xlabel('Patrón', fontsize=10, fontweight='bold')
        self.ax1.set_ylabel('Salida', fontsize=10, fontweight='bold')
        self.ax1.set_title('Salidas Deseadas vs Salidas de la Red', 
                          fontsize=11, fontweight='bold')
        self.ax1.legend(loc='best', fontsize=9)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_xticks(patterns)
        self.fig1.tight_layout()
        self.canvas1.draw()
        
        # Gráfica 2: Errores
        self.ax2.clear()
        
        # Barras de error por patrón
        colors = ['#e74c3c' if abs(e) > error_optimo/len(self.Y_data) else '#27ae60' 
                 for e in errors]
        self.ax2.bar(patterns, np.abs(errors), color=colors, alpha=0.7, 
                    label='|Error| por patrón')
        
        # Línea de error general
        self.ax2.axhline(y=error_general, color='#3498db', linestyle='--', 
                        linewidth=2, label=f'EG = {error_general:.4f}')
        
        # Línea de error óptimo
        self.ax2.axhline(y=error_optimo, color='#f39c12', linestyle='--', 
                        linewidth=2, label=f'Error Óptimo = {error_optimo:.4f}')
        
        self.ax2.set_xlabel('Patrón', fontsize=10, fontweight='bold')
        self.ax2.set_ylabel('Error', fontsize=10, fontweight='bold')
        self.ax2.set_title('Análisis de Errores', fontsize=11, fontweight='bold')
        self.ax2.legend(loc='best', fontsize=9)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_xticks(patterns)
        self.fig2.tight_layout()
        self.canvas2.draw()


if __name__ == '__main__':
    root = tk.Tk()
    app = RBF_GUI(root)
    root.mainloop()
