import numpy as np
from matplotlib.figure import Figure

# Funciones para graficar resultados y errores

def plot_train_results(ax1, ax2, Y_train, Y_pred, errors, error_general, error_optimo):
    n = len(Y_pred)
    patterns = np.arange(1, n + 1)
    ax1.clear()
    ax1.plot(patterns, Y_train, 'o-', label='YD (Deseada)', color='#3498db', linewidth=2, markersize=8)
    ax1.plot(patterns, Y_pred, 's-', label='YR (Red)', color='#e74c3c', linewidth=2, markersize=8)
    ax1.set_xlabel('Patrón', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Salida', fontsize=10, fontweight='bold')
    ax1.set_title('Salidas Deseadas vs Salidas de la Red (Train)', fontsize=11, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(patterns)
    ax1.figure.tight_layout()
    ax1.figure.canvas.draw()
    # Gráfica 2: Errores
    ax2.clear()
    colors = ['#e74c3c' if abs(e) > error_optimo/n else '#27ae60' for e in errors]
    ax2.bar(patterns, np.abs(errors), color=colors, alpha=0.7, label='|Error| por patrón')
    ax2.axhline(y=error_general, color='#3498db', linestyle='--', linewidth=2, label=f'EG = {error_general:.4f}')
    ax2.axhline(y=error_optimo, color='#f39c12', linestyle='--', linewidth=2, label=f'Error Óptimo = {error_optimo:.4f}')
    ax2.set_xlabel('Patrón', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Error', fontsize=10, fontweight='bold')
    ax2.set_title('Análisis de Errores (Train)', fontsize=11, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(patterns)
    ax2.figure.tight_layout()
    ax2.figure.canvas.draw()
