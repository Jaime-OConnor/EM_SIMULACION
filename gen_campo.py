"""
Reconstrucción de distribuciones de cargas puntuales a partir del campo eléctrico
(con ruido) mediante una CNN.

Flujo:
    1. Generar sets aleatorios de cargas y calcular V, Ex, Ey en una malla 2D.
    2. Normalizar con estadísticas globales del dataset.
    3. Entrenar una CNN que predice (Q, x, y) de cada carga.
    4. Comparar predicciones con valores reales.
"""

import math
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ==========================================
# CONSTANTES
# ==========================================
K_COULOMB = 8.98755e9   # N·m²/C²
RUIDO_RELATIVO = 0.5    # Desviación típica del ruido, relativa a |E|
RUTA_MODELO = "modelo_cargas.pt"  # nuevo nombre: no reutiliza el modelo antiguo


# ==========================================
# 1. FÍSICA: POTENCIAL Y CAMPO
# ==========================================
@lru_cache(maxsize=8)
def _malla(x_lim, y_lim, res):
    """Malla 2D cacheada: todos los sets con los mismos límites comparten X, Y (ahorra memoria)."""
    x = np.linspace(*x_lim, res)
    y = np.linspace(*y_lim, res)
    X, Y = np.meshgrid(x, y)
    X.setflags(write=False)
    Y.setflags(write=False)
    return X, Y


def calcular_potencial_y_campo(cargas, x_lim=(-1, 1), y_lim=(-1, 1), res=200,
                               ruido_relativo=RUIDO_RELATIVO):
    """
    Calcula V (sin ruido) y E (con ruido gaussiano proporcional) para una lista de cargas.

    cargas : [(q, x, y), ...]
    Devuelve dict con X, Y, V, Ex, Ey. En la posición exacta de una carga el valor es NaN.
    """
    X, Y = _malla(tuple(x_lim), tuple(y_lim), res)
    V = np.zeros_like(X)
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(X)

    for q, cx, cy in cargas:
        rx, ry = X - cx, Y - cy
        r = np.hypot(rx, ry)
        r = np.where(r == 0, np.nan, r)  # evita división por cero
        r3 = r ** 3

        V += K_COULOMB * q / r

        ex = K_COULOMB * q * rx / r3
        ey = K_COULOMB * q * ry / r3
        Ex += ex + ruido_relativo * np.abs(ex) * np.random.randn(*X.shape)
        Ey += ey + ruido_relativo * np.abs(ey) * np.random.randn(*X.shape)

    return {"X": X, "Y": Y, "V": V, "Ex": Ex, "Ey": Ey}


def graficar_potencial_y_campo(datos, cargas, titulo=""):
    """Mapa de potencial, equipotenciales, líneas de campo y posición de las cargas."""
    X, Y, V, Ex, Ey = (datos[k] for k in ("X", "Y", "V", "Ex", "Ey"))

    fig, ax = plt.subplots(figsize=(8, 7))

    v_max = np.nanpercentile(np.abs(V), 95)
    pcm = ax.pcolormesh(X, Y, V, cmap="RdBu_r", vmin=-v_max, vmax=v_max,
                        shading="auto", alpha=0.7)
    fig.colorbar(pcm, ax=ax, label="Potencial Eléctrico $V$ (V)")

    ax.contour(X, Y, V, levels=15, colors="black", linewidths=0.5, alpha=0.5)
    ax.streamplot(X, Y, Ex, Ey, color="black", density=1.1, linewidth=0.8, arrowsize=1.0)

    for q, cx, cy in cargas:
        ax.plot(cx, cy, "o", color="red" if q > 0 else "blue",
                markersize=10, markeredgecolor="k", zorder=5)
        ax.text(cx, cy, "+" if q > 0 else "-", color="white", weight="bold",
                fontsize=10, ha="center", va="center", zorder=6)

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(titulo)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    plt.show()


def mapa_potencial_y_campo(cargas, x_lim=(-1, 1), y_lim=(-1, 1), res=200,
                           titulo="", graficar=True):
    """Calcula (y opcionalmente grafica) V y E. Misma interfaz que la versión original."""
    datos = calcular_potencial_y_campo(cargas, x_lim, y_lim, res)
    if graficar:
        graficar_potencial_y_campo(datos, cargas, titulo)
    return datos


# ==========================================
# 2. GENERACIÓN DE DATOS
# ==========================================
def generar_cargas_aleatorias(n_cargas, x_lim=(-1, 1), y_lim=(-1, 1), q_max=2e-9):
    """Cargas en el 80 % central del plano, |q| ∈ [0.2·q_max, q_max], signo aleatorio."""
    cx = np.random.uniform(x_lim[0] * 0.8, x_lim[1] * 0.8, n_cargas)
    cy = np.random.uniform(y_lim[0] * 0.8, y_lim[1] * 0.8, n_cargas)
    q = np.random.choice([-1, 1], n_cargas) * np.random.uniform(0.2 * q_max, q_max, n_cargas)
    return [(float(qi), float(xi), float(yi)) for qi, xi, yi in zip(q, cx, cy)]


def generar_y_graficar_sets_aleatorios(num_sets, n_cargas, x_lim=(-1, 1), y_lim=(-1, 1),
                                       q_max=2e-9, res=200, graficar=True):
    """Genera `num_sets` distribuciones aleatorias. Devuelve [{'cargas', 'datos_malla'}, ...]."""
    resultados = []
    for i in range(num_sets):
        cargas = generar_cargas_aleatorias(n_cargas, x_lim, y_lim, q_max)
        datos = mapa_potencial_y_campo(
            cargas, x_lim, y_lim, res,
            titulo=f"Set {i + 1} de {num_sets} ({n_cargas} cargas aleatorias)",
            graficar=graficar,
        )
        resultados.append({"cargas": cargas, "datos_malla": datos})
    return resultados


# ==========================================
# 3. NORMALIZACIÓN Y DATASET
# ==========================================
def _log_campo(E):
    """Transformación logarítmica con signo: sign(E)·log(1+|E|). NaN → 0."""
    E = np.nan_to_num(E, nan=0.0)
    return np.sign(E) * np.log1p(np.abs(E))


def calcular_estadisticas_globales(datos_generados):
    """Máximos globales de |Q|, |log E| y distancia, para normalizar a [-1, 1]."""
    max_q = max_log_e = max_dist = 0.0

    for item in datos_generados:
        malla = item["datos_malla"]
        max_q = max(max_q, max(abs(q) for q, _, _ in item["cargas"]))
        max_log_e = max(max_log_e,
                        np.abs(_log_campo(malla["Ex"])).max(),
                        np.abs(_log_campo(malla["Ey"])).max())
        max_dist = max(max_dist, np.abs(malla["X"]).max(), np.abs(malla["Y"]).max())

    stats = {
        "max_q": float(max_q) or 1.0,          # evita divisiones por cero
        "max_dist": float(max_dist) or 1.0,
        "max_log_e": float(max_log_e) or 1.0,
    }
    print(f"Estadísticas -> Max Q: {stats['max_q']:.2e} C | "
          f"Max Dist: {stats['max_dist']} m | Max Log(E): {stats['max_log_e']:.2f}")
    return stats


class CargasDataset(Dataset):
    """
    Entrada: tensor (2, H, W) con Ex, Ey en log y normalizados.
    Target:  tensor (n_cargas, 3) con (Q, x, y) normalizados, ordenados por x.
    """

    def __init__(self, datos_generados, stats):
        self.datos = datos_generados
        self.stats = stats

    def __len__(self):
        return len(self.datos)

    def __getitem__(self, idx):
        item = self.datos[idx]
        malla, s = item["datos_malla"], self.stats

        entrada = np.stack([_log_campo(malla["Ex"]), _log_campo(malla["Ey"])]) / s["max_log_e"]

        target = np.array(sorted(item["cargas"], key=lambda c: c[1]), dtype=np.float32)
        target /= np.array([s["max_q"], s["max_dist"], s["max_dist"]], dtype=np.float32)

        return (torch.tensor(entrada, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32))


class CargasAlVuelo(Dataset):
    """
    Genera un set nuevo en cada acceso: datos ilimitados sin ocupar RAM
    y sin que el modelo pueda memorizarlos. `n_muestras` = muestras por época.
    """

    def __init__(self, n_muestras, n_cargas, stats, x_lim=(-1, 1), y_lim=(-1, 1),
                 q_max=2e-9, res=200):
        self.n, self.n_cargas, self.stats = n_muestras, n_cargas, stats
        self.x_lim, self.y_lim, self.q_max, self.res = x_lim, y_lim, q_max, res

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        cargas = generar_cargas_aleatorias(self.n_cargas, self.x_lim, self.y_lim, self.q_max)
        malla = calcular_potencial_y_campo(cargas, self.x_lim, self.y_lim, self.res)
        return CargasDataset([{"cargas": cargas, "datos_malla": malla}], self.stats)[0]


# ==========================================
# 4. MODELO
# ==========================================
def _bloque_conv(c_in, c_out):
    """Conv → BN → ReLU → MaxPool. Se devuelve como lista para mantener los nombres
    de capa del modelo original (compatibilidad con checkpoints ya guardados)."""
    return [
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    ]


class CNNExtractorCargas(nn.Module):
    """CNN + MLP que predice (Q, x, y) para `max_cargas` cargas."""

    def __init__(self, max_cargas=10, input_channels=2):
        super().__init__()
        self.max_cargas = max_cargas

        self.features = nn.Sequential(
            *_bloque_conv(input_channels, 16),
            *_bloque_conv(16, 32),
            *_bloque_conv(32, 64),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.regressor = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, max_cargas * 3),
        )

    def forward(self, x):
        out = self.regressor(torch.flatten(self.features(x), 1))
        return out.view(x.size(0), self.max_cargas, 3)


# ==========================================
# 5. ENTRENAMIENTO
# ==========================================
def _dispositivo():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _semilla_worker(_):
    """Cada proceso del DataLoader genera datos distintos."""
    np.random.seed(torch.initial_seed() % 2**32)


def _epoca(modelo, loader, criterion, device, optimizer=None):
    """Una pasada por `loader`. Si hay optimizer entrena; si no, evalúa. Devuelve la pérdida media."""
    entrenando = optimizer is not None
    modelo.train(entrenando)
    perdida_total = 0.0

    with torch.set_grad_enabled(entrenando):
        for batch_X, batch_Y in loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            loss = criterion(modelo(batch_X), batch_Y)
            if entrenando:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            perdida_total += loss.item() * batch_X.size(0)

    return perdida_total / len(loader.dataset)


def graficar_curvas(hist_train, hist_test):
    """Curvas de pérdida en escala log, omitiendo la primera época."""
    if len(hist_train) < 2:
        return
    epocas = range(2, len(hist_train) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epocas, hist_train[1:], marker="o", label="Training Loss", color="blue")
    plt.plot(epocas, hist_test[1:], marker="s", label="Test Loss", color="red")
    plt.title("Curvas de Aprendizaje (Training vs Test Loss)")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida (MSE)")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def entrenar_modelo_cargas(num_sets=3000, n_cargas=3, batch_size=64, epochs=100,
                           learning_rate=1e-3, plot_curves=True, semilla=None,
                           n_test=1000, num_workers=4, weight_decay=1e-4):
    """
    Entrena la CNN con datos generados al vuelo.

    num_sets    : muestras de entrenamiento NUEVAS por época (no se guardan en RAM).
    n_test      : tamaño del set de test fijo (se genera una vez).
    num_workers : procesos que generan datos en paralelo (0 = sin paralelizar).

    Devuelve (modelo, hist_train_loss, hist_test_loss, test_dataset, stats).
    """
    device = _dispositivo()
    if semilla is not None:
        np.random.seed(semilla)
        torch.manual_seed(semilla)

    # Test fijo: sirve para comparar épocas y para fijar la normalización
    print(f"Generando set de test fijo ({n_test} ejemplos)...")
    datos_test = generar_y_graficar_sets_aleatorios(n_test, n_cargas, graficar=False)
    stats = calcular_estadisticas_globales(datos_test)
    test_ds = CargasDataset(datos_test, stats)

    # Train: datos nuevos en cada época
    train_ds = CargasAlVuelo(num_sets, n_cargas, stats)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        worker_init_fn=_semilla_worker if num_workers > 0 else None,
        persistent_workers=num_workers > 0, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"Train al vuelo: {num_sets}/época | Test fijo: {n_test} | Dispositivo: {device}")

    modelo = CNNExtractorCargas(max_cargas=n_cargas, input_channels=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(modelo.parameters(), lr=learning_rate, weight_decay=weight_decay)

    hist_train, hist_test = [], []
    print("Iniciando entrenamiento...")
    for epoch in range(1, epochs + 1):
        hist_train.append(_epoca(modelo, train_loader, criterion, device, optimizer))
        hist_test.append(_epoca(modelo, test_loader, criterion, device))
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch [{epoch:02d}/{epochs}] | Train Loss: {hist_train[-1]:.4f} "
                  f"| Test Loss: {hist_test[-1]:.4f}")
    print("¡Entrenamiento finalizado!")

    if plot_curves:
        graficar_curvas(hist_train, hist_test)

    return modelo, hist_train, hist_test, test_ds, stats


# ==========================================
# 6. EVALUACIÓN VISUAL
# ==========================================
def comparar_predicciones(modelo, test_dataset, stats, n_ejemplos=30, por_pagina=9):
    """Cuadrícula Real (○) vs Predicha (✕) por páginas. Rojo: +, azul: −."""
    modelo.eval()
    device = next(modelo.parameters()).device
    d, q_max = stats["max_dist"], stats["max_q"]
    escala = np.array([q_max, d, d])

    n_total = min(n_ejemplos, len(test_dataset))
    n_paginas = math.ceil(n_total / por_pagina)

    for pag in range(n_paginas):
        ini, fin = pag * por_pagina, min((pag + 1) * por_pagina, n_total)
        n = fin - ini
        cols = min(4, math.ceil(math.sqrt(n)))
        filas = math.ceil(n / cols)
        fig, axes = plt.subplots(filas, cols, figsize=(4 * cols, 4 * filas), squeeze=False)
        axes = axes.ravel()

        for ax, idx in zip(axes, range(ini, fin)):
            X_t, Y_t = test_dataset[idx]
            with torch.no_grad():
                Y_p = modelo(X_t.unsqueeze(0).to(device))[0].cpu()

            real = Y_t.numpy() * escala
            pred = Y_p.numpy() * escala

            for r, p in zip(real, pred):
                ax.plot([r[1], p[1]], [r[2], p[2]], "--", color="gray", lw=1, zorder=1)
            ax.scatter(real[:, 1], real[:, 2], c=np.sign(real[:, 0]), cmap="bwr", vmin=-1, vmax=1,
                       s=90, marker="o", edgecolors="k", zorder=2, label="Real")
            ax.scatter(pred[:, 1], pred[:, 2], c=np.sign(pred[:, 0]), cmap="bwr", vmin=-1, vmax=1,
                       s=90, marker="x", linewidths=2, zorder=3, label="Predicha")

            ax.set_xlim(-d, d)
            ax.set_ylim(-d, d)
            ax.set_aspect("equal")
            ax.set_title(f"Test #{idx}", fontsize=10)
            ax.grid(alpha=0.3, ls="--")

        for ax in axes[n:]:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.suptitle(f"Real (○) vs Predicha (✕) | rojo: +, azul: − | Página {pag + 1}/{n_paginas}", y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        plt.show()


# ==========================================
# 7. PERSISTENCIA
# ==========================================
def cargar_o_entrenar(ruta=RUTA_MODELO, n_cargas=3, **kwargs_entrenamiento):
    """Carga el modelo si existe el checkpoint; si no, lo entrena y lo guarda."""
    device = _dispositivo()
    try:
        ckpt = torch.load(ruta, map_location=device, weights_only=False)
        n_ckpt = ckpt.get("n_cargas", n_cargas)  # compatible con checkpoints antiguos
        modelo = CNNExtractorCargas(max_cargas=n_ckpt, input_channels=2).to(device)
        modelo.load_state_dict(ckpt["pesos"])
        stats = ckpt["stats"]
        print("Modelo cargado.")
    except FileNotFoundError:
        modelo, _, _, _, stats = entrenar_modelo_cargas(n_cargas=n_cargas, **kwargs_entrenamiento)
        torch.save({"pesos": modelo.state_dict(), "stats": stats, "n_cargas": n_cargas}, ruta)
        print("Modelo entrenado y guardado.")
    return modelo, stats


# ==========================================
# EJEMPLO DE USO
# ==========================================
if __name__ == "__main__":
    N_CARGAS = 4
    modelo, stats = cargar_o_entrenar(
        n_cargas=N_CARGAS,
        num_sets=3000,     # muestras NUEVAS por época (no se guardan en RAM)
        epochs=60,          
        batch_size=64,
        learning_rate=1e-3,
        n_test=1000,        # test fijo
        num_workers=4,      # núcleos de CPU generando datos (bájalo si va mal)
    )
    datos_nuevos = generar_y_graficar_sets_aleatorios(num_sets=18, n_cargas=N_CARGAS, graficar=False)
    comparar_predicciones(modelo, CargasDataset(datos_nuevos, stats), stats)