import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


def mapa_potencial_y_campo(cargas, x_lim=(-1, 1), y_lim=(-1, 1), res=200, titulo="", graficar=True):
    """
    Calcula el potencial y campo eléctrico para una distribución de cargas.
    
    Parámetros:
    -----------
    cargas : lista de tuplas [(q1, x1, y1), ...]
    x_lim, y_lim : tuplas con límites del plano.
    res : resolución de la malla.
    titulo : título de la gráfica.
    graficar : bool (True para mostrar el gráfico, False para solo devolver los datos calculados).
    """
    K = 8.98755e9  # Constante de Coulomb (N·m²/C²)
    ruido_relativo=0.5 #porcentaje de ruido en E (desv tipica)
    # Crear malla 2D
    x = np.linspace(x_lim[0], x_lim[1], res)
    y = np.linspace(y_lim[0], y_lim[1], res)
    X, Y = np.meshgrid(x, y)
    
    V = np.zeros_like(X, dtype=float)
    Ex = np.zeros_like(X, dtype=float)
    Ey = np.zeros_like(Y, dtype=float)
    
    # Superposición de contribuciones
    for q, cx, cy in cargas:
        rx = X - cx
        ry = Y - cy
        r = np.hypot(rx, ry)

        # Evitar división por cero
        r_safe = np.where(r == 0, np.nan, r)

        # Potencial eléctrico
        V += K * q / r_safe

        # Campo eléctrico sin ruido
        Ex_contrib = K * q * rx / (r_safe**3)
        Ey_contrib = K * q * ry / (r_safe**3)

        # Ruido gaussiano proporcional al campo
        Ex += Ex_contrib + ruido_relativo * np.abs(Ex_contrib) * np.random.randn(*X.shape)
        Ey += Ey_contrib + ruido_relativo * np.abs(Ey_contrib) * np.random.randn(*Y.shape)
    # Si no se requiere gráfica, devolvemos las matrices calculadas
    if not graficar:
        return {"X": X, "Y": Y, "V": V, "Ex": Ex, "Ey": Ey}
    
    # Generación de la gráfica
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 1. Mapa de calor del Potencial Eléctrico
    v_max = np.nanpercentile(np.abs(V), 95)
    pcm = ax.pcolormesh(X, Y, V, cmap='RdBu_r', vmin=-v_max, vmax=v_max, shading='auto', alpha=0.7)
    fig.colorbar(pcm, ax=ax, label='Potencial Eléctrico $V$ (V)')
    
    # 2. Líneas equipotenciales
    ax.contour(X, Y, V, levels=15, colors='black', linewidths=0.5, alpha=0.5)
    
    # 3. Líneas de campo eléctrico
    ax.streamplot(X, Y, Ex, Ey, color='black', density=1.1, linewidth=0.8, arrowsize=1.0)
    
    # 4. Posición de las cargas
    for q, cx, cy in cargas:
        color = 'red' if q > 0 else 'blue'
        signo = '+' if q > 0 else '-'
        ax.plot(cx, cy, 'o', color=color, markersize=10, markeredgecolor='k', zorder=5)
        ax.text(cx, cy, signo, color='white', weight='bold', fontsize=10,
                ha='center', va='center', zorder=6)
    
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(titulo)
    ax.set_aspect('equal')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return {"X": X, "Y": Y, "V": V, "Ex": Ex, "Ey": Ey}

def generar_y_graficar_sets_aleatorios(num_sets, n_cargas, x_lim=(-1, 1), y_lim=(-1, 1), q_max=2e-9, graficar=True):
    """
    Genera `num_sets` distribuciones aleatorias.
    
    Parámetros:
    -----------
    graficar : bool (Si es True gráfica cada set. Si es False, retorna una lista con los datos de todos los sets).
    """
    resultados = []
    
    for i in range(num_sets):
        cargas_aleatorias = []
        for _ in range(n_cargas):
            cx = np.random.uniform(x_lim[0] * 0.8, x_lim[1] * 0.8)
            cy = np.random.uniform(y_lim[0] * 0.8, y_lim[1] * 0.8)
            
            signo = np.random.choice([-1, 1])
            q_magnitude = np.random.uniform(0.2 * q_max, q_max)
            q = signo * q_magnitude
            
            cargas_aleatorias.append((q, cx, cy))
        
        titulo_set = f'Set {i+1} de {num_sets} ({n_cargas} cargas aleatorias)'
        datos = mapa_potencial_y_campo(cargas_aleatorias, x_lim=x_lim, y_lim=y_lim, titulo=titulo_set, graficar=graficar)
        
        resultados.append({
            "cargas": cargas_aleatorias,
            "datos_malla": datos
        })
        
    return resultados

def calcular_estadisticas_globales(datos_generados, x_lim, y_lim):
    """
    Recorre el dataset generado para encontrar los valores máximos absolutos
    de la carga (Q) y del campo eléctrico logarítmico.
    Esto permite normalizar dinámicamente sin importar la escala física.
    """
    max_q_global = 0.0
    max_log_e_global = 0.0
    
    for item in datos_generados:
        # 1. Buscar la carga máxima en este set
        cargas = item["cargas"]
        for q, cx, cy in cargas:
            if abs(q) > max_q_global:
                max_q_global = abs(q)
                
        # 2. Buscar el valor máximo del campo tras el logaritmo
        Ex = np.nan_to_num(item["datos_malla"]["Ex"], nan=0.0)
        Ey = np.nan_to_num(item["datos_malla"]["Ey"], nan=0.0)
        
        Ex_log = np.log1p(np.abs(Ex))
        Ey_log = np.log1p(np.abs(Ey))
        
        max_log_local = max(np.max(Ex_log), np.max(Ey_log))
        if max_log_local > max_log_e_global:
            max_log_e_global = max_log_local

    # 3. Calcular la distancia máxima según los límites del plano
    max_dist_x = max(abs(x_lim[0]), abs(x_lim[1]))
    max_dist_y = max(abs(y_lim[0]), abs(y_lim[1]))
    max_dist_global = max(max_dist_x, max_dist_y)
    
    # Evitar divisiones por cero en caso de arrays vacíos
    if max_q_global == 0: max_q_global = 1.0
    if max_log_e_global == 0: max_log_e_global = 1.0
    
    print(f"Estadísticas calculadas -> Max Q: {max_q_global:.2e} C | Max Dist: {max_dist_global} m | Max Log(E): {max_log_e_global:.2f}")
    
    return {
        "max_q": max_q_global,
        "max_dist": max_dist_global,
        "max_log_e": max_log_e_global
    }


class CargasDataset(Dataset):
    def __init__(self, datos_generados, stats):
        """
        Dataset que utiliza estadísticas globales calculadas dinámicamente.
        """
        self.datos = datos_generados
        self.stats = stats

    def __len__(self):
        return len(self.datos)

    def __getitem__(self, idx):
        item = self.datos[idx]
        
        # --- 1. ENTRADAS (E_x, E_y) ---
        Ex = item["datos_malla"]["Ex"]
        Ey = item["datos_malla"]["Ey"]
        
        Ex = np.nan_to_num(Ex, nan=0.0)
        Ey = np.nan_to_num(Ey, nan=0.0)
        
        # Transformación logarítmica y normalización dinámica al rango [-1, 1]
        Ex_norm = (np.sign(Ex) * np.log1p(np.abs(Ex))) / self.stats["max_log_e"]
        Ey_norm = (np.sign(Ey) * np.log1p(np.abs(Ey))) / self.stats["max_log_e"]
        
        X_tensor = torch.tensor(np.stack([Ex_norm, Ey_norm], axis=0), dtype=torch.float32)
        
        # --- 2. TARGETS (Q, X, Y) ---
        cargas = item["cargas"]
        cargas_ordenadas = sorted(cargas, key=lambda c: c[1])
        Y_array = np.array(cargas_ordenadas, dtype=np.float32)
        
        # Normalizar la carga Q dividiendo por el Q máximo global de todo el dataset
        Y_array[:, 0] = Y_array[:, 0] / self.stats["max_q"]
        
        # Normalizar coordenadas X, Y dividiendo por la distancia máxima del plano
        Y_array[:, 1] = Y_array[:, 1] / self.stats["max_dist"]
        Y_array[:, 2] = Y_array[:, 2] / self.stats["max_dist"]
        
        Y_tensor = torch.tensor(Y_array, dtype=torch.float32)
        
        return X_tensor, Y_tensor
        
class CNNExtractorCargas(nn.Module):
    def __init__(self, max_cargas=10, input_channels=2):
        super(CNNExtractorCargas, self).__init__()
        self.max_cargas = max_cargas
        
        # Extractor de características (CNN)
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Red neuronal completamente conectada (MLP)
        self.regressor = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, max_cargas * 3)
        )

    # El cambio principal está aquí: se elimina "n_cargas" de los argumentos
    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. Extracción de características
        x = self.features(x)
        
        # 2. Aplanar
        x = x.view(batch_size, -1)
        
        # 3. Predicción
        out = self.regressor(x)
        
        # 4. Redimensionar para devolver siempre TODAS las max_cargas
        out = out.view(batch_size, self.max_cargas, 3)
        
        return out

    
# ==========================================
# 1. DEFINICIÓN DEL DATASET
# ==========================================
def calcular_estadisticas_globales(datos_generados):
    """
    Recorre el dataset generado para encontrar los valores máximos absolutos
    de la carga (Q), campo eléctrico logarítmico y dimensiones espaciales,
    sin requerir límites predefinidos.
    """
    max_q_global = 0.0
    max_log_e_global = 0.0
    max_dist_global = 0.0
    
    for item in datos_generados:
        # 1. Buscar la carga máxima real en este set
        for q, cx, cy in item["cargas"]:
            if abs(q) > max_q_global:
                max_q_global = abs(q)
                
        # 2. Buscar el valor máximo del campo tras el logaritmo
        Ex = np.nan_to_num(item["datos_malla"]["Ex"], nan=0.0)
        Ey = np.nan_to_num(item["datos_malla"]["Ey"], nan=0.0)
        
        Ex_log = np.log1p(np.abs(Ex))
        Ey_log = np.log1p(np.abs(Ey))
        
        max_log_local = max(np.max(Ex_log), np.max(Ey_log))
        if max_log_local > max_log_e_global:
            max_log_e_global = max_log_local

        # 3. Buscar la distancia máxima directamente de las matrices generadas
        X_malla = item["datos_malla"]["X"]
        Y_malla = item["datos_malla"]["Y"]
        max_dist_local = max(np.max(np.abs(X_malla)), np.max(np.abs(Y_malla)))
        
        if max_dist_local > max_dist_global:
            max_dist_global = max_dist_local

    # Prevención de divisiones por cero por si se pasa un dataset vacío o sin cargas
    if max_q_global == 0: max_q_global = 1.0
    if max_log_e_global == 0: max_log_e_global = 1.0
    if max_dist_global == 0: max_dist_global = 1.0
    
    print(f"Estadísticas 100% dinámicas -> Max Q: {max_q_global:.2e} C | Max Dist: {max_dist_global} m | Max Log(E): {max_log_e_global:.2f}")
    
    return {
        "max_q": max_q_global,
        "max_dist": max_dist_global,
        "max_log_e": max_log_e_global
    }

class CargasDataset(Dataset):
    def __init__(self, datos_generados, stats):
        """
        Dataset que utiliza estadísticas globales calculadas dinámicamente.
        """
        self.datos = datos_generados
        self.stats = stats

    def __len__(self):
        return len(self.datos)

    def __getitem__(self, idx):
        item = self.datos[idx]
        
        # --- 1. ENTRADAS (E_x, E_y) ---
        Ex = item["datos_malla"]["Ex"]
        Ey = item["datos_malla"]["Ey"]
        
        Ex = np.nan_to_num(Ex, nan=0.0)
        Ey = np.nan_to_num(Ey, nan=0.0)
        
        # Transformación logarítmica y normalización dinámica al rango [-1, 1]
        Ex_norm = (np.sign(Ex) * np.log1p(np.abs(Ex))) / self.stats["max_log_e"]
        Ey_norm = (np.sign(Ey) * np.log1p(np.abs(Ey))) / self.stats["max_log_e"]
        
        X_tensor = torch.tensor(np.stack([Ex_norm, Ey_norm], axis=0), dtype=torch.float32)
        
        # --- 2. TARGETS (Q, X, Y) ---
        cargas = item["cargas"]
        cargas_ordenadas = sorted(cargas, key=lambda c: c[1])
        Y_array = np.array(cargas_ordenadas, dtype=np.float32)
        
        # Normalizar la carga Q dividiendo por el Q máximo global de todo el dataset
        Y_array[:, 0] = Y_array[:, 0] / self.stats["max_q"]
        
        # Normalizar coordenadas X, Y dividiendo por la distancia máxima del plano
        Y_array[:, 1] = Y_array[:, 1] / self.stats["max_dist"]
        Y_array[:, 2] = Y_array[:, 2] / self.stats["max_dist"]
        
        Y_tensor = torch.tensor(Y_array, dtype=torch.float32)
        
        return X_tensor, Y_tensor
    
def entrenar_modelo_cargas(num_sets=200, n_cargas=3, batch_size=16, epochs=100, learning_rate=1e-3, plot_curves=True):
    """
    Entrena el modelo CNNExtractorCargas generando datos aleatorios.
    
    Parámetros:
    -----------
    num_sets : int
        Número total de muestras a generar para el dataset.
    n_cargas : int
        Cargas por plano (coincidirá con max_cargas del modelo).
    batch_size : int
        Tamaño del lote para los DataLoaders.
    epochs : int
        Número de épocas de entrenamiento.
    learning_rate : float
        Tasa de aprendizaje para el optimizador Adam.
    plot_curves : bool
        Si es True, grafica las curvas de pérdida al finalizar.
        
    Retorna:
    --------
    modelo : nn.Module
        El modelo entrenado.
    hist_train_loss : list
        Historial de pérdida de entrenamiento.
    hist_test_loss : list
        Historial de pérdida de validación.
    """
    print(f"Generando dataset con {num_sets} ejemplos...")
    # Llamamos a la función de tu código base[cite: 1]
    datos_crudos = generar_y_graficar_sets_aleatorios(
        num_sets=num_sets, 
        n_cargas=n_cargas, 
        graficar=False 
    )

# Calcular dinámicamente toda la escala física leyendo las matrices
    stats_dinamicas = calcular_estadisticas_globales(datos_crudos)

    # Pasar las estadísticas al dataset
    dataset = CargasDataset(datos_crudos, stats=stats_dinamicas)

    
    # Split 80% Train / 20% Test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Datos preparados: {train_size} para Train, {test_size} para Test.")

    # ----------------------------------
    # Inicialización del Modelo
    # ----------------------------------
    modelo = CNNExtractorCargas(max_cargas=n_cargas, input_channels=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(modelo.parameters(), lr=learning_rate)

    hist_train_loss = []
    hist_test_loss = []

    # ----------------------------------
    # Bucle de Entrenamiento
    # ----------------------------------
    print("Iniciando entrenamiento...")
    for epoch in range(epochs):
        
        # --- FASE DE ENTRENAMIENTO ---
        modelo.train()
        train_loss_acum = 0.0
        
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            predicciones = modelo(batch_X)
            loss = criterion(predicciones, batch_Y)
            loss.backward()
            optimizer.step()
            
            train_loss_acum += loss.item() * batch_X.size(0)
            
        epoch_train_loss = train_loss_acum / len(train_loader.dataset)
        hist_train_loss.append(epoch_train_loss)
        
        # --- FASE DE EVALUACIÓN (TEST) ---
        modelo.eval()
        test_loss_acum = 0.0
        
        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                predicciones = modelo(batch_X)
                loss = criterion(predicciones, batch_Y)
                test_loss_acum += loss.item() * batch_X.size(0)
                
        epoch_test_loss = test_loss_acum / len(test_loader.dataset)
        hist_test_loss.append(epoch_test_loss)
        
        # Feedback visual
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Test Loss: {epoch_test_loss:.4f}")

    print("¡Entrenamiento finalizado!")

    # ----------------------------------
    # GRAFICAR LAS CURVAS
    # ----------------------------------
    if plot_curves:
        plt.figure(figsize=(10, 6))
       # En lugar de graficar desde la época 1, omitimos el primer valor [1:]
        plt.plot(range(2, epochs + 1), hist_train_loss[1:], marker='o', label='Training Loss', color='blue')
        plt.plot(range(2, epochs + 1), hist_test_loss[1:], marker='s', label='Test Loss', color='red')
        
        plt.title('Curvas de Aprendizaje (Training vs Test Loss)')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida (MSE)')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # al final de entrenar_modelo_cargas
    return modelo, hist_train_loss, hist_test_loss, test_dataset, stats_dinamicas
import math

def comparar_predicciones(modelo, test_dataset, stats, n_ejemplos=30, por_pagina=9):
    modelo.eval()
    d, q_max = stats["max_dist"], stats["max_q"]
    n_total = min(n_ejemplos, len(test_dataset))
    n_paginas = math.ceil(n_total / por_pagina)

    for pag in range(n_paginas):
        ini = pag * por_pagina
        fin = min(ini + por_pagina, n_total)
        n = fin - ini

        cols = min(4, math.ceil(math.sqrt(n)))
        filas = math.ceil(n / cols)
        fig, axes = plt.subplots(filas, cols, figsize=(4 * cols, 4 * filas), squeeze=False)
        axes = axes.ravel()

        for ax, idx in zip(axes, range(ini, fin)):
            X_t, Y_t = test_dataset[idx]
            with torch.no_grad():
                Y_p = modelo(X_t.unsqueeze(0))[0]

            real = Y_t.numpy() * [q_max, d, d]
            pred = Y_p.numpy() * [q_max, d, d]

            for r, p in zip(real, pred):
                ax.plot([r[1], p[1]], [r[2], p[2]], '--', color='gray', lw=1, zorder=1)
            ax.scatter(real[:, 1], real[:, 2], c=np.sign(real[:, 0]), cmap='bwr', vmin=-1, vmax=1,
                       s=90, marker='o', edgecolors='k', zorder=2, label='Real')
            ax.scatter(pred[:, 1], pred[:, 2], c=np.sign(pred[:, 0]), cmap='bwr', vmin=-1, vmax=1,
                       s=90, marker='x', linewidths=2, zorder=3, label='Predicha')

            ax.set_xlim(-d, d); ax.set_ylim(-d, d); ax.set_aspect('equal')
            ax.set_title(f"Test #{idx}", fontsize=10)
            ax.grid(alpha=0.3, ls='--')

        for ax in axes[n:]:
            ax.axis('off')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
        fig.suptitle(f"Real (○) vs Predicha (✕) | rojo: +, azul: − | Página {pag + 1}/{n_paginas}", y=1.0)
        plt.tight_layout(rect=(0, 0, 1, 0.94))
        plt.show()
#==========================
# EJEMPLO DE USO
# ==========================================
RUTA = "modelo_cargas.pt"
start=1
if start == 1:
    try:
        ckpt = torch.load(RUTA, weights_only=False)
        modelo = CNNExtractorCargas(max_cargas=3, input_channels=2)
        modelo.load_state_dict(ckpt["pesos"])
        stats = ckpt["stats"]
        print("Modelo cargado.")
    except FileNotFoundError:
        modelo, _, _, _, stats = entrenar_modelo_cargas(
            num_sets=500, n_cargas=3, batch_size=16, epochs=100, learning_rate=0.01
        )
        torch.save({"pesos": modelo.state_dict(), "stats": stats}, RUTA)
        print("Modelo entrenado y guardado.")

    datos = generar_y_graficar_sets_aleatorios(num_sets=10, n_cargas=3, graficar=False)
    comparar_predicciones(modelo, CargasDataset(datos, stats), stats)