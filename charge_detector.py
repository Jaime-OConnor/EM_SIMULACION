"""
Detection of a VARIABLE number of point charges from the (noisy) electric field
using a convolutional network that predicts heatmaps (CenterNet style).

Pipeline:
    1. Generate random sets with a random number of charges and compute V, Ex, Ey.
    2. Build the target maps: heat (where a charge is), q and sub-pixel offset.
    3. Train a U-Net that predicts those maps from (Ex, Ey).
    4. Detect peaks in the heatmap: each peak is a charge (the count comes out on its own).
    5. Evaluate (precision, recall, position error, sign, |q|) and compare visually.
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
# CONSTANTS
# ==========================================
K_COULOMB = 8.98755e9   # N·m²/C²
RELATIVE_NOISE = 0.5    # Noise standard deviation, relative to |E|
MODEL_PATH = "english_model_new_v3.pt"
SIGMA_PX = 2.0          # Width (pixels) of the Gaussian that marks each charge in the heatmap
THRESHOLD = 0.5         # Minimum probability to accept a peak as a charge
MATCH_DIST = 0.05       # (m) Maximum distance to consider a charge correctly detected


# ==========================================
# 1. PHYSICS: POTENTIAL AND FIELD
# ==========================================
@lru_cache(maxsize=8)
def _grid(x_lim, y_lim, res):
    """Cached 2D grid: all sets with the same limits share X, Y (saves memory)."""
    x = np.linspace(*x_lim, res)
    y = np.linspace(*y_lim, res)
    X, Y = np.meshgrid(x, y)
    X.setflags(write=False)
    Y.setflags(write=False)
    return X, Y


def compute_potential_and_field(charges, x_lim=(-1, 1), y_lim=(-1, 1), res=200,
                                relative_noise=RELATIVE_NOISE):
    """
    Computes V (noise-free) and E (with proportional Gaussian noise) for a list of charges.

    charges : [(q, x, y), ...]
    Returns a dict with X, Y, V, Ex, Ey. At the exact position of a charge the value is NaN.
    """
    X, Y = _grid(tuple(x_lim), tuple(y_lim), res)
    V = np.zeros_like(X)
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(X)

    for q, cx, cy in charges:
        rx, ry = X - cx, Y - cy
        r = np.hypot(rx, ry)
        r = np.where(r == 0, np.nan, r)  # avoid division by zero
        r3 = r ** 3

        V += K_COULOMB * q / r

        ex = K_COULOMB * q * rx / r3
        ey = K_COULOMB * q * ry / r3
        Ex += ex + relative_noise * np.abs(ex) * np.random.randn(*X.shape)
        Ey += ey + relative_noise * np.abs(ey) * np.random.randn(*X.shape)

    return {"X": X, "Y": Y, "V": V, "Ex": Ex, "Ey": Ey}


def plot_potential_and_field(data, charges, title=""):
    """Potential map, equipotentials, field lines and charge positions."""
    X, Y, V, Ex, Ey = (data[k] for k in ("X", "Y", "V", "Ex", "Ey"))

    fig, ax = plt.subplots(figsize=(8, 7))

    v_max = np.nanpercentile(np.abs(V), 95)
    pcm = ax.pcolormesh(X, Y, V, cmap="RdBu_r", vmin=-v_max, vmax=v_max,
                        shading="auto", alpha=0.7)
    fig.colorbar(pcm, ax=ax, label="Electric potential $V$ (V)")

    ax.contour(X, Y, V, levels=15, colors="black", linewidths=0.5, alpha=0.5)
    ax.streamplot(X, Y, Ex, Ey, color="black", density=1.1, linewidth=0.8, arrowsize=1.0)

    for q, cx, cy in charges:
        ax.plot(cx, cy, "o", color="red" if q > 0 else "blue",
                markersize=10, markeredgecolor="k", zorder=5)
        ax.text(cx, cy, "+" if q > 0 else "-", color="white", weight="bold",
                fontsize=10, ha="center", va="center", zorder=6)

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    plt.show()


def potential_and_field_map(charges, x_lim=(-1, 1), y_lim=(-1, 1), res=200,
                            title="", plot=True):
    """Computes (and optionally plots) V and E."""
    data = compute_potential_and_field(charges, x_lim, y_lim, res)
    if plot:
        plot_potential_and_field(data, charges, title)
    return data


# ==========================================
# 2. DATA GENERATION
# ==========================================
def _count_range(n_charges):
    """Accepts an integer (fixed count) or a tuple (n_min, n_max)."""
    return (n_charges, n_charges) if isinstance(n_charges, int) else tuple(n_charges)


def generate_random_charges(n_charges, x_lim=(-1, 1), y_lim=(-1, 1), q_max=2e-9, min_sep=0.1):
    """
    Charges in the central 80 % of the plane, |q| ∈ [0.2·q_max, q_max], random sign.

    n_charges : int (fixed) or (n_min, n_max) → random count in that range (both inclusive).
    min_sep   : (m) minimum separation between charges, so they don't merge in the heatmap.
    """
    n_min, n_max = _count_range(n_charges)
    n = np.random.randint(n_min, n_max + 1)
    charges = []
    for _ in range(n):
        for _attempt in range(100):
            cx = np.random.uniform(x_lim[0] * 0.8, x_lim[1] * 0.8)
            cy = np.random.uniform(y_lim[0] * 0.8, y_lim[1] * 0.8)
            if all(math.hypot(cx - x, cy - y) >= min_sep for _, x, y in charges):
                q = np.random.choice([-1, 1]) * np.random.uniform(0.2 * q_max, q_max)
                charges.append((float(q), float(cx), float(cy)))
                break
    return charges


def generate_random_sets(num_sets, n_charges, x_lim=(-1, 1), y_lim=(-1, 1),
                         q_max=2e-9, res=200, plot=True, min_sep=0.1):
    """Generates `num_sets` random distributions. Returns [{'charges', 'grid_data'}, ...]."""
    results = []
    for i in range(num_sets):
        charges = generate_random_charges(n_charges, x_lim, y_lim, q_max, min_sep)
        data = potential_and_field_map(
            charges, x_lim, y_lim, res,
            title=f"Set {i + 1} of {num_sets} ({len(charges)} random charges)",
            plot=plot,
        )
        results.append({"charges": charges, "grid_data": data})
    return results


# ==========================================
# 3. NORMALIZATION, TARGET MAPS AND DATASET
# ==========================================
def _log_field(E):
    """Signed logarithmic transform: sign(E)·log(1+|E|). NaN → 0."""
    E = np.nan_to_num(E, nan=0.0)
    return np.sign(E) * np.log1p(np.abs(E))


def _geometry(grid):
    """Grid origin and step: what is needed to convert meters to pixels and back."""
    X, Y = grid["X"], grid["Y"]
    return {"x0": float(X[0, 0]), "dx": float(X[0, 1] - X[0, 0]),
            "y0": float(Y[0, 0]), "dy": float(Y[1, 0] - Y[0, 0]),
            "H": int(X.shape[0]), "W": int(X.shape[1])}


def compute_global_stats(generated_data):
    """Global maxima of |Q|, |log E| and distance, plus the grid geometry."""
    max_q = max_log_e = max_dist = 0.0

    for item in generated_data:
        grid = item["grid_data"]
        max_q = max(max_q, max((abs(q) for q, _, _ in item["charges"]), default=0.0))
        max_log_e = max(max_log_e,
                        np.abs(_log_field(grid["Ex"])).max(),
                        np.abs(_log_field(grid["Ey"])).max())
        max_dist = max(max_dist, np.abs(grid["X"]).max(), np.abs(grid["Y"]).max())

    stats = {
        "max_q": float(max_q) or 1.0,          # avoid division by zero
        "max_dist": float(max_dist) or 1.0,
        "max_log_e": float(max_log_e) or 1.0,
        "geometry": _geometry(generated_data[0]["grid_data"]),
    }
    print(f"Stats -> Max Q: {stats['max_q']:.2e} C | "
          f"Max Dist: {stats['max_dist']} m | Max Log(E): {stats['max_log_e']:.2f}")
    return stats


def target_maps(charges, grid, max_q, sigma=SIGMA_PX):
    """
    Builds the target (4, H, W):
        [0] heat  : Gaussian centred on each charge's pixel (exactly 1 at the centre)
        [1] q     : q / max_q at each charge's centre pixel (0 elsewhere)
        [2] off_x : sub-pixel offset in x, at that centre pixel
        [3] off_y : sub-pixel offset in y
    """
    g = _geometry(grid)
    H, W = g["H"], g["W"]
    target = np.zeros((4, H, W), dtype=np.float32)
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]

    for q, cx, cy in charges:
        fx = (cx - g["x0"]) / g["dx"]   # position in pixels (fractional)
        fy = (cy - g["y0"]) / g["dy"]
        ix, iy = int(round(fx)), int(round(fy))
        if not (0 <= ix < W and 0 <= iy < H):
            continue
        gauss = np.exp(-((cols - ix) ** 2 + (rows - iy) ** 2) / (2 * sigma ** 2))
        np.maximum(target[0], gauss, out=target[0])
        target[1, iy, ix] = q / max_q
        target[2, iy, ix] = fx - ix
        target[3, iy, ix] = fy - iy
    return target


class ChargeDataset(Dataset):
    """
    Input:  tensor (2, H, W) with Ex, Ey log-transformed and normalized.
    Target: tensor (4, H, W) with the heat, q and offset maps (see `target_maps`).
    The ground-truth charges remain accessible in `self.data[idx]["charges"]`.
    """

    def __init__(self, generated_data, stats, sigma=SIGMA_PX):
        self.data = generated_data
        self.stats = stats
        self.sigma = sigma

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        grid, s = item["grid_data"], self.stats

        inputs = np.stack([_log_field(grid["Ex"]), _log_field(grid["Ey"])]) / s["max_log_e"]
        target = target_maps(item["charges"], grid, s["max_q"], self.sigma)

        return (torch.tensor(inputs, dtype=torch.float32),
                torch.from_numpy(target))


class OnTheFlyChargeDataset(Dataset):
    """
    Generates a new set on every access: unlimited data without using RAM
    and without the model being able to memorize it. `n_samples` = samples per epoch.
    """

    def __init__(self, n_samples, n_charges, stats, x_lim=(-1, 1), y_lim=(-1, 1),
                 q_max=2e-9, res=200, min_sep=0.1, sigma=SIGMA_PX):
        self.n, self.n_charges, self.stats = n_samples, n_charges, stats
        self.x_lim, self.y_lim, self.q_max, self.res = x_lim, y_lim, q_max, res
        self.min_sep, self.sigma = min_sep, sigma

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        charges = generate_random_charges(self.n_charges, self.x_lim, self.y_lim,
                                          self.q_max, self.min_sep)
        grid = compute_potential_and_field(charges, self.x_lim, self.y_lim, self.res)
        item = {"charges": charges, "grid_data": grid}
        return ChargeDataset([item], self.stats, self.sigma)[0]


# ==========================================
# 4. MODEL (U-Net)
# ==========================================
def _conv_block(c_in, c_out):
    """Conv → BN → ReLU, twice. Same kind of block as the previous CNN."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
    )


class ChargeDetectorCNN(nn.Module):
    """
    U-Net: the encoder is the one from the previous CNN (16 → 32 → 64 channels with MaxPool);
    the decoder goes back to the original resolution using skip connections.
    Output (B, 4, H, W): [heat logit, q, off_x, off_y]. Works with any grid size.
    """

    def __init__(self, input_channels=2, base=16):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        # Encoder
        self.enc1 = _conv_block(input_channels, c1)   # H
        self.enc2 = _conv_block(c1, c2)               # H/2
        self.enc3 = _conv_block(c2, c3)               # H/4
        self.bottleneck = _conv_block(c3, c4)         # H/8
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder (upsample, concatenate with the encoder level and refine)
        self.dec3 = _conv_block(c4 + c3, c3)
        self.dec2 = _conv_block(c3 + c2, c2)
        self.dec1 = _conv_block(c2 + c1, c1)

        # Head: 4 output maps
        self.head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, 4, kernel_size=1),
        )
        # Standard CenterNet initialization: at the start it predicts "almost no charge"
        nn.init.constant_(self.head[-1].bias, 0.0)
        with torch.no_grad():
            self.head[-1].bias[0] = -2.19   # sigmoid(-2.19) ≈ 0.1

    @staticmethod
    def _upsample(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self._upsample(b, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._upsample(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._upsample(d2, e1), e1], dim=1))
        return self.head(d1)


class DetectorLoss(nn.Module):
    """
    Total loss = focal (heatmap) + weight_q·L1(q) + weight_off·L1(offset).
    The focal loss is CenterNet's: it barely penalizes pixels close to a charge
    (they are not "serious mistakes") and focuses on the hard ones.
    q and offset are only evaluated at each charge's centre pixel.
    """

    def __init__(self, weight_q=1.0, weight_off=1.0, alpha=2, beta=4):
        super().__init__()
        self.weight_q, self.weight_off, self.alpha, self.beta = weight_q, weight_off, alpha, beta

    def forward(self, output, target):
        p = torch.sigmoid(output[:, 0]).clamp(1e-4, 1 - 1e-4)
        gt = target[:, 0]
        center = gt.eq(1).float()
        n_pos = center.sum().clamp(min=1.0)

        pos = torch.log(p) * (1 - p) ** self.alpha * center
        neg = torch.log(1 - p) * p ** self.alpha * (1 - gt) ** self.beta * (1 - center)
        focal = -(pos.sum() + neg.sum()) / n_pos

        mask = center.unsqueeze(1)
        l1 = (output[:, 1:] - target[:, 1:]).abs() * mask
        l1_q = l1[:, 0].sum() / n_pos
        l1_off = l1[:, 1:].sum() / n_pos

        return focal + self.weight_q * l1_q + self.weight_off * l1_off


# ==========================================
# 5. DECODING AND METRICS
# ==========================================
def decode(output, stats, threshold=THRESHOLD):
    """
    Converts the network output (4, H, W) for ONE sample into an (n, 3) array of (q, x, y).
    A charge = local maximum of the heatmap (3×3 neighbourhood) above `threshold`.
    """
    g = stats["geometry"]
    heat = torch.sigmoid(output[0])
    peaks = F.max_pool2d(heat[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    iy, ix = torch.nonzero((heat == peaks) & (heat > threshold), as_tuple=True)

    q = output[1, iy, ix] * stats["max_q"]
    x = g["x0"] + (ix.float() + output[2, iy, ix]) * g["dx"]
    y = g["y0"] + (iy.float() + output[3, iy, ix]) * g["dy"]
    return torch.stack([q, x, y], dim=1).cpu().numpy()


def match(true, pred, max_dist=MATCH_DIST):
    """Matches true and predicted charges (Hungarian algorithm). Returns pairs (i_true, j_pred)."""
    if len(true) == 0 or len(pred) == 0:
        return []
    d = np.hypot(true[:, None, 1] - pred[None, :, 1], true[:, None, 2] - pred[None, :, 2])
    rows, cols = linear_sum_assignment(d)
    return [(i, j) for i, j in zip(rows, cols) if d[i, j] <= max_dist]


def evaluate_detector(model, dataset, stats, threshold=THRESHOLD, max_dist=MATCH_DIST, batch_size=32):
    """Metrics on a ChargeDataset: precision, recall, exact count, position, sign and |q|."""
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    tp = fp = fn = n_exact = 0
    pos_err, sign_ok, q_err = [], [], []
    idx = 0
    with torch.no_grad():
        for batch_X, _ in loader:
            outputs = model(batch_X.to(device))
            for output in outputs:
                true = np.array(dataset.data[idx]["charges"], dtype=float).reshape(-1, 3)
                pred = decode(output, stats, threshold)
                pairs = match(true, pred, max_dist)

                tp += len(pairs)
                fp += len(pred) - len(pairs)
                fn += len(true) - len(pairs)
                n_exact += len(pred) == len(true)
                for i, j in pairs:
                    pos_err.append(math.hypot(true[i, 1] - pred[j, 1], true[i, 2] - pred[j, 2]))
                    sign_ok.append(np.sign(true[i, 0]) == np.sign(pred[j, 0]))
                    q_err.append(abs(abs(pred[j, 0]) - abs(true[i, 0])) / abs(true[i, 0]))
                idx += 1

    m = {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "exact_count": n_exact / max(idx, 1),
        "pos_error_m": float(np.mean(pos_err)) if pos_err else float("nan"),
        "sign_accuracy": float(np.mean(sign_ok)) if sign_ok else float("nan"),
        "rel_q_error": float(np.mean(q_err)) if q_err else float("nan"),
    }
    print(f"Precision: {m['precision']:.1%} | Recall: {m['recall']:.1%} | "
          f"Exact charge count: {m['exact_count']:.1%}\n"
          f"Position error: {m['pos_error_m'] * 100:.2f} cm | Sign: {m['sign_accuracy']:.1%} | "
          f"|q| error: {m['rel_q_error']:.1%}")
    return m


# ==========================================
# 6. TRAINING
# ==========================================
def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_worker(_):
    """Each DataLoader process generates different data."""
    np.random.seed(torch.initial_seed() % 2**32)


def _run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`. With an optimizer it trains; otherwise it evaluates. Returns the mean loss."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for batch_X, batch_Y in loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            loss = criterion(model(batch_X), batch_Y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch_X.size(0)

    return total_loss / len(loader.dataset)


def plot_learning_curves(train_hist, test_hist):
    """Loss curves on a log scale, skipping the first epoch."""
    if len(train_hist) < 2:
        return
    epochs = range(2, len(train_hist) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_hist[1:], marker="o", label="Training Loss", color="blue")
    plt.plot(epochs, test_hist[1:], marker="s", label="Test Loss", color="red")
    plt.title("Learning Curves (Training vs Test Loss)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (focal + L1)")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def train_charge_model(num_sets=3000, n_charges=(1, 6), batch_size=32, epochs=100,
                       learning_rate=1e-3, plot_curves=True, seed=None,
                       n_test=1000, num_workers=4, weight_decay=1e-4,
                       min_sep=0.1, sigma=SIGMA_PX):
    """
    Trains the detector with data generated on the fly.

    num_sets    : NEW training samples per epoch (not stored in RAM).
    n_charges   : (n_min, n_max) charges per set, or an integer for a fixed count.
    n_test      : size of the fixed test set (generated once).
    num_workers : processes generating data in parallel (0 = no parallelism).
    min_sep     : minimum separation (m) between charges in the same set.

    Returns (model, train_loss_hist, test_loss_hist, test_dataset, stats).
    """
    device = _device()
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Fixed test set: used to compare epochs and to fix the normalization
    print(f"Generating fixed test set ({n_test} examples)...")
    test_data = generate_random_sets(n_test, n_charges, plot=False, min_sep=min_sep)
    stats = compute_global_stats(test_data)
    test_ds = ChargeDataset(test_data, stats, sigma)

    # Train: new data every epoch
    train_ds = OnTheFlyChargeDataset(num_sets, n_charges, stats, min_sep=min_sep, sigma=sigma)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        persistent_workers=num_workers > 0, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"On-the-fly train: {num_sets}/epoch | Fixed test: {n_test} | "
          f"Charges per set: {_count_range(n_charges)} | Device: {device}")

    model = ChargeDetectorCNN(input_channels=2).to(device)
    criterion = DetectorLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_hist, test_hist = [], []
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        train_hist.append(_run_epoch(model, train_loader, criterion, device, optimizer))
        test_hist.append(_run_epoch(model, test_loader, criterion, device))
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch [{epoch:02d}/{epochs}] | Train Loss: {train_hist[-1]:.4f} "
                  f"| Test Loss: {test_hist[-1]:.4f}")
    print("Training finished!")

    print("\nTest set metrics:")
    evaluate_detector(model, test_ds, stats)

    if plot_curves:
        plot_learning_curves(train_hist, test_hist)

    return model, train_hist, test_hist, test_ds, stats


# ==========================================
# 7. VISUAL EVALUATION
# ==========================================
def compare_predictions(model, test_dataset, stats, n_examples=30, per_page=9, threshold=THRESHOLD):
    """
    Paged grid of True (○) vs Predicted (✕), with the predicted heatmap in the background.
    Red: +, blue: −. Grey line: true ↔ predicted match.
    """
    model.eval()
    device = next(model.parameters()).device
    g = stats["geometry"]
    x_max, y_max = g["x0"] + g["dx"] * (g["W"] - 1), g["y0"] + g["dy"] * (g["H"] - 1)
    extent = (g["x0"], x_max, g["y0"], y_max)

    n_total = min(n_examples, len(test_dataset))
    n_pages = math.ceil(n_total / per_page)

    for page in range(n_pages):
        start, end = page * per_page, min((page + 1) * per_page, n_total)
        n = end - start
        cols = min(4, math.ceil(math.sqrt(n)))
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
        axes = axes.ravel()

        for ax, idx in zip(axes, range(start, end)):
            X_t, _ = test_dataset[idx]
            with torch.no_grad():
                output = model(X_t.unsqueeze(0).to(device))[0]
            pred = decode(output, stats, threshold)
            true = np.array(test_dataset.data[idx]["charges"], dtype=float).reshape(-1, 3)

            heat = torch.sigmoid(output[0]).cpu().numpy()
            ax.imshow(heat, extent=extent, origin="lower", cmap="Greys", vmin=0, vmax=1, alpha=0.5)

            for i, j in match(true, pred):
                ax.plot([true[i, 1], pred[j, 1]], [true[i, 2], pred[j, 2]],
                        "--", color="gray", lw=1, zorder=1)
            ax.scatter(true[:, 1], true[:, 2], c=np.sign(true[:, 0]), cmap="bwr", vmin=-1, vmax=1,
                       s=90, marker="o", edgecolors="k", zorder=2, label="True")
            ax.scatter(pred[:, 1], pred[:, 2], c=np.sign(pred[:, 0]), cmap="bwr", vmin=-1, vmax=1,
                       s=90, marker="x", linewidths=2, zorder=3, label="Predicted")

            ax.set_xlim(g["x0"], x_max)
            ax.set_ylim(g["y0"], y_max)
            ax.set_aspect("equal")
            ax.set_title(f"Test #{idx} | true: {len(true)} · pred: {len(pred)}", fontsize=10)
            ax.grid(alpha=0.3, ls="--")

        for ax in axes[n:]:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.suptitle(f"True (○) vs Predicted (✕) | red: +, blue: − | Page {page + 1}/{n_pages}", y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        plt.show()


# ==========================================
# 8. PERSISTENCE
# ==========================================
def load_or_train(path=MODEL_PATH, **train_kwargs):
    """Loads the model if the checkpoint exists; otherwise trains it and saves it."""
    device = _device()
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except FileNotFoundError:
        model, _, _, _, stats = train_charge_model(**train_kwargs)
        torch.save({"type": "heatmap_detector", "weights": model.state_dict(), "stats": stats}, path)
        print("Model trained and saved.")
        return model, stats

    if ckpt.get("type") != "heatmap_detector":
        raise ValueError(f"'{path}' is an old model (fixed number of charges or old format). "
                         f"Use another path to train the new detector.")
    model = ChargeDetectorCNN(input_channels=2).to(device)
    model.load_state_dict(ckpt["weights"])
    print("Model loaded.")
    return model, ckpt["stats"]


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    N_CHARGES = (5, 10)   # each set has between 5 and 10 charges; the network does NOT know how many

    model, stats = load_or_train(
        n_charges=N_CHARGES,
        num_sets=800,       # NEW samples per epoch (not stored in RAM)
        epochs=20,
        batch_size=32,      # the U-Net uses more GPU memory than the previous CNN
        learning_rate=1e-3,
        n_test=1000,        # fixed test set
        num_workers=8,      # CPU cores generating data (lower it if things go wrong)
        min_sep=0,          # minimum separation between charges (m)
    )

    new_data = generate_random_sets(num_sets=18, n_charges=N_CHARGES, plot=False)
    new_ds = ChargeDataset(new_data, stats)
    print("\nMetrics on new data:")
    evaluate_detector(model, new_ds, stats)
    compare_predictions(model, new_ds, stats)
