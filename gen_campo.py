"""
Detección de un número VARIABLE de cargas puntuales a partir del campo eléctrico
(con ruido) mediante una red convolucional que predice mapas de calor (estilo CenterNet).

Flujo:
    1. Generar sets aleatorios con un número aleatorio de cargas y calcular V, Ex, Ey.
    2. Construir los mapas objetivo: calor (dónde hay carga), q y desplazamiento subpíxel.
    3. Entrenar una U-Net que predice esos mapas a partir de (Ex, Ey).
    4. Detectar picos en el mapa de calor: cada pico es una carga (el número sale solo).
    5. Evaluar (precisión, recall, error de posición, signo, |q|) y comparar visualmente.
"""

import math
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

# ==========================================
# CONSTANTES
# ==========================================
K_COULOMB = 8.98755e9   # N·m²/C²
RUIDO_RELATIVO = 0.5    # Desviación típica del ruido, relativa a |E|
RUTA_MODELO = "modelo_cargas_v3.pt"
SIGMA_PX = 2.0          # Anchura (píxeles) de la gaussiana que marca cada carga en el mapa de calor
UMBRAL = 0.5            # Probabilidad mínima para aceptar un pico como carga
DIST_ACIERTO = 0.05     # (m) Distancia máxima para considerar una carga bien detectada


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
    """Calcula (y opcionalmente grafica) V y E."""
    datos = calcular_potencial_y_campo(cargas, x_lim, y_lim, res)
    if graficar:
        graficar_potencial_y_campo(datos, cargas, titulo)
    return datos


# ==========================================
# 2. GENERACIÓN DE DATOS
# ==========================================
def _rango(n_cargas):
    """Acepta un entero (número fijo) o una tupla (n_min, n_max)."""
    return (n_cargas, n_cargas) if isinstance(n_cargas, int) else tuple(n_cargas)


def generar_cargas_aleatorias(n_cargas, x_lim=(-1, 1), y_lim=(-1, 1), q_max=2e-9, min_sep=0.1):
    """
    Cargas en el 80 % central del plano, |q| ∈ [0.2·q_max, q_max], signo aleatorio.

    n_cargas : int (fijo) o (n_min, n_max) → número aleatorio en ese rango (ambos incluidos).
    min_sep  : (m) separación mínima entre cargas, para que no se fundan en el mapa de calor.
    """
    n_min, n_max = _rango(n_cargas)
    n = np.random.randint(n_min, n_max + 1)
    cargas = []
    for _ in range(n):
        for _intento in range(100):
            cx = np.random.uniform(x_lim[0] * 0.8, x_lim[1] * 0.8)
            cy = np.random.uniform(y_lim[0] * 0.8, y_lim[1] * 0.8)
            if all(math.hypot(cx - x, cy - y) >= min_sep for _, x, y in cargas):
                q = np.random.choice([-1, 1]) * np.random.uniform(0.2 * q_max, q_max)
                cargas.append((float(q), float(cx), float(cy)))
                break
    return cargas


def generar_y_graficar_sets_aleatorios(num_sets, n_cargas, x_lim=(-1, 1), y_lim=(-1, 1),
                                       q_max=2e-9, res=200, graficar=True, min_sep=0.1):
    """Genera `num_sets` distribuciones aleatorias. Devuelve [{'cargas', 'datos_malla'}, ...]."""
    resultados = []
    for i in range(num_sets):
        cargas = generar_cargas_aleatorias(n_cargas, x_lim, y_lim, q_max, min_sep)
        datos = mapa_potencial_y_campo(
            cargas, x_lim, y_lim, res,
            titulo=f"Set {i + 1} de {num_sets} ({len(cargas)} cargas aleatorias)",
            graficar=graficar,
        )
        resultados.append({"cargas": cargas, "datos_malla": datos})
    return resultados


# ==========================================
# 3. NORMALIZACIÓN, MAPAS OBJETIVO Y DATASET
# ==========================================
def _log_campo(E):
    """Transformación logarítmica con signo: sign(E)·log(1+|E|). NaN → 0."""
    E = np.nan_to_num(E, nan=0.0)
    return np.sign(E) * np.log1p(np.abs(E))


def _geometria(malla):
    """Origen y paso de la malla: lo necesario para pasar de metros a píxeles y viceversa."""
    X, Y = malla["X"], malla["Y"]
    return {"x0": float(X[0, 0]), "dx": float(X[0, 1] - X[0, 0]),
            "y0": float(Y[0, 0]), "dy": float(Y[1, 0] - Y[0, 0]),
            "H": int(X.shape[0]), "W": int(X.shape[1])}


def calcular_estadisticas_globales(datos_generados):
    """Máximos globales de |Q|, |log E| y distancia, más la geometría de la malla."""
    max_q = max_log_e = max_dist = 0.0

    for item in datos_generados:
        malla = item["datos_malla"]
        max_q = max(max_q, max((abs(q) for q, _, _ in item["cargas"]), default=0.0))
        max_log_e = max(max_log_e,
                        np.abs(_log_campo(malla["Ex"])).max(),
                        np.abs(_log_campo(malla["Ey"])).max())
        max_dist = max(max_dist, np.abs(malla["X"]).max(), np.abs(malla["Y"]).max())

    stats = {
        "max_q": float(max_q) or 1.0,          # evita divisiones por cero
        "max_dist": float(max_dist) or 1.0,
        "max_log_e": float(max_log_e) or 1.0,
        "geometria": _geometria(datos_generados[0]["datos_malla"]),
    }
    print(f"Estadísticas -> Max Q: {stats['max_q']:.2e} C | "
          f"Max Dist: {stats['max_dist']} m | Max Log(E): {stats['max_log_e']:.2f}")
    return stats


def mapas_objetivo(cargas, malla, max_q, sigma=SIGMA_PX):
    """
    Construye el target (4, H, W):
        [0] calor : gaussiana centrada en el píxel de cada carga (vale exactamente 1 en el centro)
        [1] q     : q / max_q en el píxel central de cada carga (0 en el resto)
        [2] off_x : desplazamiento subpíxel en x, en ese píxel central
        [3] off_y : desplazamiento subpíxel en y
    """
    g = _geometria(malla)
    H, W = g["H"], g["W"]
    obj = np.zeros((4, H, W), dtype=np.float32)
    filas = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]

    for q, cx, cy in cargas:
        fx = (cx - g["x0"]) / g["dx"]   # posición en píxeles (con decimales)
        fy = (cy - g["y0"]) / g["dy"]
        ix, iy = int(round(fx)), int(round(fy))
        if not (0 <= ix < W and 0 <= iy < H):
            continue
        gauss = np.exp(-((cols - ix) ** 2 + (filas - iy) ** 2) / (2 * sigma ** 2))
        np.maximum(obj[0], gauss, out=obj[0])
        obj[1, iy, ix] = q / max_q
        obj[2, iy, ix] = fx - ix
        obj[3, iy, ix] = fy - iy
    return obj


class CargasDataset(Dataset):
    """
    Entrada: tensor (2, H, W) con Ex, Ey en log y normalizados.
    Target:  tensor (4, H, W) con los mapas de calor, q y desplazamiento (ver `mapas_objetivo`).
    Las cargas reales siguen accesibles en `self.datos[idx]["cargas"]`.
    """

    def __init__(self, datos_generados, stats, sigma=SIGMA_PX):
        self.datos = datos_generados
        self.stats = stats
        self.sigma = sigma

    def __len__(self):
        return len(self.datos)

    def __getitem__(self, idx):
        item = self.datos[idx]
        malla, s = item["datos_malla"], self.stats

        entrada = np.stack([_log_campo(malla["Ex"]), _log_campo(malla["Ey"])]) / s["max_log_e"]
        objetivo = mapas_objetivo(item["cargas"], malla, s["max_q"], self.sigma)

        return (torch.tensor(entrada, dtype=torch.float32),
                torch.from_numpy(objetivo))


class CargasAlVuelo(Dataset):
    """
    Genera un set nuevo en cada acceso: datos ilimitados sin ocupar RAM
    y sin que el modelo pueda memorizarlos. `n_muestras` = muestras por época.
    """

    def __init__(self, n_muestras, n_cargas, stats, x_lim=(-1, 1), y_lim=(-1, 1),
                 q_max=2e-9, res=200, min_sep=0.1, sigma=SIGMA_PX):
        self.n, self.n_cargas, self.stats = n_muestras, n_cargas, stats
        self.x_lim, self.y_lim, self.q_max, self.res = x_lim, y_lim, q_max, res
        self.min_sep, self.sigma = min_sep, sigma

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        cargas = generar_cargas_aleatorias(self.n_cargas, self.x_lim, self.y_lim,
                                           self.q_max, self.min_sep)
        malla = calcular_potencial_y_campo(cargas, self.x_lim, self.y_lim, self.res)
        item = {"cargas": cargas, "datos_malla": malla}
        return CargasDataset([item], self.stats, self.sigma)[0]


# ==========================================
# 4. MODELO (U-Net)
# ==========================================
def _bloque_conv(c_in, c_out):
    """Conv → BN → ReLU, dos veces. Mismo tipo de bloque que la CNN anterior."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
    )


class CNNDetectorCargas(nn.Module):
    """
    U-Net: el codificador es el de la CNN anterior (16 → 32 → 64 canales con MaxPool);
    el decodificador vuelve a la resolución original usando conexiones de salto.
    Salida (B, 4, H, W): [logit calor, q, off_x, off_y]. Funciona con cualquier tamaño de malla.
    """

    def __init__(self, input_channels=2, base=16):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        # Codificador
        self.enc1 = _bloque_conv(input_channels, c1)   # H
        self.enc2 = _bloque_conv(c1, c2)               # H/2
        self.enc3 = _bloque_conv(c2, c3)               # H/4
        self.fondo = _bloque_conv(c3, c4)              # H/8
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decodificador (sube, concatena con el nivel del codificador y refina)
        self.dec3 = _bloque_conv(c4 + c3, c3)
        self.dec2 = _bloque_conv(c3 + c2, c2)
        self.dec1 = _bloque_conv(c2 + c1, c1)

        # Cabeza: 4 mapas de salida
        self.cabeza = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, 4, kernel_size=1),
        )
        # Inicialización estándar de CenterNet: al principio predice "casi nada de carga"
        nn.init.constant_(self.cabeza[-1].bias, 0.0)
        with torch.no_grad():
            self.cabeza[-1].bias[0] = -2.19   # sigmoid(-2.19) ≈ 0.1

    @staticmethod
    def _subir(x, referencia):
        return F.interpolate(x, size=referencia.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        f = self.fondo(self.pool(e3))

        d3 = self.dec3(torch.cat([self._subir(f, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._subir(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._subir(d2, e1), e1], dim=1))
        return self.cabeza(d1)


class PerdidaDetector(nn.Module):
    """
    Pérdida total = focal (mapa de calor) + peso_q·L1(q) + peso_off·L1(desplazamiento).
    La focal es la de CenterNet: penaliza poco los píxeles cercanos a una carga
    (no son "fallos graves") y se centra en los difíciles.
    q y desplazamiento solo se evalúan en los píxeles centrales de cada carga.
    """

    def __init__(self, peso_q=1.0, peso_off=1.0, alpha=2, beta=4):
        super().__init__()
        self.peso_q, self.peso_off, self.alpha, self.beta = peso_q, peso_off, alpha, beta

    def forward(self, salida, objetivo):
        p = torch.sigmoid(salida[:, 0]).clamp(1e-4, 1 - 1e-4)
        gt = objetivo[:, 0]
        centro = gt.eq(1).float()
        n_pos = centro.sum().clamp(min=1.0)

        pos = torch.log(p) * (1 - p) ** self.alpha * centro
        neg = torch.log(1 - p) * p ** self.alpha * (1 - gt) ** self.beta * (1 - centro)
        focal = -(pos.sum() + neg.sum()) / n_pos

        mascara = centro.unsqueeze(1)
        l1 = (salida[:, 1:] - objetivo[:, 1:]).abs() * mascara
        l1_q = l1[:, 0].sum() / n_pos
        l1_off = l1[:, 1:].sum() / n_pos

        return focal + self.peso_q * l1_q + self.peso_off * l1_off


# ==========================================
# 5. DECODIFICACIÓN Y MÉTRICAS
# ==========================================
def decodificar(salida, stats, umbral=UMBRAL):
    """
    Convierte la salida de la red (4, H, W) para UNA muestra en un array (n, 3) de (q, x, y).
    Una carga = máximo local del mapa de calor (vecindario 3×3) por encima de `umbral`.
    """
    g = stats["geometria"]
    calor = torch.sigmoid(salida[0])
    maximos = F.max_pool2d(calor[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    iy, ix = torch.nonzero((calor == maximos) & (calor > umbral), as_tuple=True)

    q = salida[1, iy, ix] * stats["max_q"]
    x = g["x0"] + (ix.float() + salida[2, iy, ix]) * g["dx"]
    y = g["y0"] + (iy.float() + salida[3, iy, ix]) * g["dy"]
    return torch.stack([q, x, y], dim=1).cpu().numpy()


def emparejar(real, pred, dist_max=DIST_ACIERTO):
    """Empareja cargas reales y predichas (algoritmo húngaro). Devuelve pares (i_real, j_pred)."""
    if len(real) == 0 or len(pred) == 0:
        return []
    d = np.hypot(real[:, None, 1] - pred[None, :, 1], real[:, None, 2] - pred[None, :, 2])
    filas, cols = linear_sum_assignment(d)
    return [(i, j) for i, j in zip(filas, cols) if d[i, j] <= dist_max]


def evaluar_detector(modelo, dataset, stats, umbral=UMBRAL, dist_max=DIST_ACIERTO, batch_size=32):
    """Métricas sobre un CargasDataset: precisión, recall, número exacto, posición, signo y |q|."""
    modelo.eval()
    device = next(modelo.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    tp = fp = fn = n_exacto = 0
    err_pos, signo_ok, err_q = [], [], []
    idx = 0
    with torch.no_grad():
        for batch_X, _ in loader:
            salidas = modelo(batch_X.to(device))
            for salida in salidas:
                real = np.array(dataset.datos[idx]["cargas"], dtype=float).reshape(-1, 3)
                pred = decodificar(salida, stats, umbral)
                pares = emparejar(real, pred, dist_max)

                tp += len(pares)
                fp += len(pred) - len(pares)
                fn += len(real) - len(pares)
                n_exacto += len(pred) == len(real)
                for i, j in pares:
                    err_pos.append(math.hypot(real[i, 1] - pred[j, 1], real[i, 2] - pred[j, 2]))
                    signo_ok.append(np.sign(real[i, 0]) == np.sign(pred[j, 0]))
                    err_q.append(abs(abs(pred[j, 0]) - abs(real[i, 0])) / abs(real[i, 0]))
                idx += 1

    m = {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "n_exacto": n_exacto / max(idx, 1),
        "error_pos_m": float(np.mean(err_pos)) if err_pos else float("nan"),
        "acierto_signo": float(np.mean(signo_ok)) if signo_ok else float("nan"),
        "error_rel_q": float(np.mean(err_q)) if err_q else float("nan"),
    }
    print(f"Precisión: {m['precision']:.1%} | Recall: {m['recall']:.1%} | "
          f"Nº cargas exacto: {m['n_exacto']:.1%}\n"
          f"Error posición: {m['error_pos_m'] * 100:.2f} cm | Signo: {m['acierto_signo']:.1%} | "
          f"Error |q|: {m['error_rel_q']:.1%}")
    return m


# ==========================================
# 6. ENTRENAMIENTO
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
    plt.ylabel("Pérdida (focal + L1)")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def entrenar_modelo_cargas(num_sets=3000, n_cargas=(1, 6), batch_size=32, epochs=100,
                           learning_rate=1e-3, plot_curves=True, semilla=None,
                           n_test=1000, num_workers=4, weight_decay=1e-4,
                           min_sep=0.1, sigma=SIGMA_PX):
    """
    Entrena el detector con datos generados al vuelo.

    num_sets    : muestras de entrenamiento NUEVAS por época (no se guardan en RAM).
    n_cargas    : (n_min, n_max) cargas por set, o un entero si quieres un número fijo.
    n_test      : tamaño del set de test fijo (se genera una vez).
    num_workers : procesos que generan datos en paralelo (0 = sin paralelizar).
    min_sep     : separación mínima (m) entre cargas de un mismo set.

    Devuelve (modelo, hist_train_loss, hist_test_loss, test_dataset, stats).
    """
    device = _dispositivo()
    if semilla is not None:
        np.random.seed(semilla)
        torch.manual_seed(semilla)

    # Test fijo: sirve para comparar épocas y para fijar la normalización
    print(f"Generando set de test fijo ({n_test} ejemplos)...")
    datos_test = generar_y_graficar_sets_aleatorios(n_test, n_cargas, graficar=False, min_sep=min_sep)
    stats = calcular_estadisticas_globales(datos_test)
    test_ds = CargasDataset(datos_test, stats, sigma)

    # Train: datos nuevos en cada época
    train_ds = CargasAlVuelo(num_sets, n_cargas, stats, min_sep=min_sep, sigma=sigma)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        worker_init_fn=_semilla_worker if num_workers > 0 else None,
        persistent_workers=num_workers > 0, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"Train al vuelo: {num_sets}/época | Test fijo: {n_test} | "
          f"Cargas por set: {_rango(n_cargas)} | Dispositivo: {device}")

    modelo = CNNDetectorCargas(input_channels=2).to(device)
    criterion = PerdidaDetector()
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

    print("\nMétricas sobre el test:")
    evaluar_detector(modelo, test_ds, stats)

    if plot_curves:
        graficar_curvas(hist_train, hist_test)

    return modelo, hist_train, hist_test, test_ds, stats


# ==========================================
# 7. EVALUACIÓN VISUAL
# ==========================================
def comparar_predicciones(modelo, test_dataset, stats, n_ejemplos=30, por_pagina=9, umbral=UMBRAL):
    """
    Cuadrícula Real (○) vs Predicha (✕) por páginas, con el mapa de calor predicho de fondo.
    Rojo: +, azul: −. Línea gris: emparejamiento real ↔ predicha.
    """
    modelo.eval()
    device = next(modelo.parameters()).device
    g = stats["geometria"]
    x_max, y_max = g["x0"] + g["dx"] * (g["W"] - 1), g["y0"] + g["dy"] * (g["H"] - 1)
    extension = (g["x0"], x_max, g["y0"], y_max)

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
            X_t, _ = test_dataset[idx]
            with torch.no_grad():
                salida = modelo(X_t.unsqueeze(0).to(device))[0]
            pred = decodificar(salida, stats, umbral)
            real = np.array(test_dataset.datos[idx]["cargas"], dtype=float).reshape(-1, 3)

            calor = torch.sigmoid(salida[0]).cpu().numpy()
            ax.imshow(calor, extent=extension, origin="lower", cmap="Greys", vmin=0, vmax=1, alpha=0.5)

            for i, j in emparejar(real, pred):
                ax.plot([real[i, 1], pred[j, 1]], [real[i, 2], pred[j, 2]],
                        "--", color="gray", lw=1, zorder=1)
            ax.scatter(real[:, 1], real[:, 2], c=np.sign(real[:, 0]), cmap="bwr", vmin=-1, vmax=1,
                       s=90, marker="o", edgecolors="k", zorder=2, label="Real")
            ax.scatter(pred[:, 1], pred[:, 2], c=np.sign(pred[:, 0]), cmap="bwr", vmin=-1, vmax=1,
                       s=90, marker="x", linewidths=2, zorder=3, label="Predicha")

            ax.set_xlim(g["x0"], x_max)
            ax.set_ylim(g["y0"], y_max)
            ax.set_aspect("equal")
            ax.set_title(f"Test #{idx} | real: {len(real)} · pred: {len(pred)}", fontsize=10)
            ax.grid(alpha=0.3, ls="--")

        for ax in axes[n:]:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.suptitle(f"Real (○) vs Predicha (✕) | rojo: +, azul: − | Página {pag + 1}/{n_paginas}", y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        plt.show()


# ==========================================
# 8. PERSISTENCIA
# ==========================================
def cargar_o_entrenar(ruta=RUTA_MODELO, **kwargs_entrenamiento):
    """Carga el modelo si existe el checkpoint; si no, lo entrena y lo guarda."""
    device = _dispositivo()
    try:
        ckpt = torch.load(ruta, map_location=device, weights_only=False)
    except FileNotFoundError:
        modelo, _, _, _, stats = entrenar_modelo_cargas(**kwargs_entrenamiento)
        torch.save({"tipo": "detector_mapa_calor", "pesos": modelo.state_dict(), "stats": stats}, ruta)
        print("Modelo entrenado y guardado.")
        return modelo, stats

    if ckpt.get("tipo") != "detector_mapa_calor":
        raise ValueError(f"'{ruta}' es un modelo antiguo (número fijo de cargas). "
                         f"Usa otra ruta para entrenar el detector nuevo.")
    modelo = CNNDetectorCargas(input_channels=2).to(device)
    modelo.load_state_dict(ckpt["pesos"])
    print("Modelo cargado.")
    return modelo, ckpt["stats"]


# ==========================================
# EJEMPLO DE USO
# ==========================================
if __name__ == "__main__":
    N_CARGAS = (5, 10)   # cada set tiene entre 1 y 6 cargas; la red NO sabe cuántas

    modelo, stats = cargar_o_entrenar(
        n_cargas=N_CARGAS,
        num_sets=800,     # muestras NUEVAS por época (no se guardan en RAM)
        epochs=100,
        batch_size=32,      # la U-Net usa más memoria de GPU que la CNN anterior
        learning_rate=1e-3,
        n_test=1000,        # test fijo
        num_workers=4,      # núcleos de CPU generando datos (bájalo si va mal)
        min_sep=0,        # separación mínima entre cargas (m)
    )

    datos_nuevos = generar_y_graficar_sets_aleatorios(num_sets=18, n_cargas=N_CARGAS, graficar=False)
    ds_nuevo = CargasDataset(datos_nuevos, stats)
    print("\nMétricas sobre datos nuevos:")
    evaluar_detector(modelo, ds_nuevo, stats)
    comparar_predicciones(modelo, ds_nuevo, stats)