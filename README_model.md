## How the model works

The network detects a **variable, unknown number of point charges** from a noisy 2D electric field. Instead of regressing a fixed list of `(q, x, y)` values, it predicts **dense maps** over the grid (CenterNet-style), so the number of charges falls out of the output naturally.

### Input
A tensor of shape `(2, H, W)`: the `Ex` and `Ey` components on a 200×200 grid, with multiplicative Gaussian noise (50 % of |E| by default). Each component is passed through a signed log transform `sign(E)·log(1+|E|)` to compress the huge dynamic range near the charges, then divided by a global maximum.

### Architecture: U-Net
- **Encoder:** three `Conv-BN-ReLU ×2` blocks (16 → 32 → 64 channels), each followed by 2×2 max-pooling, down to a 128-channel bottleneck at H/8. This captures the global structure of the field (how the field lines of several charges interact).
- **Decoder:** bilinear upsampling back to full resolution. At each level the upsampled features are concatenated with the encoder features of the same resolution (**skip connections**), which recovers the fine spatial detail needed to place charges accurately.
- **Head:** a small conv head outputs 4 maps, `(4, H, W)`:
  1. **Heatmap** (logit) — probability that a charge is centred at each pixel.
  2. **q** — normalized charge value (`q / max_q`), sign included.
  3. **off_x**, 4. **off_y** — sub-pixel offset, so positions are not limited to the grid resolution.

The heatmap bias is initialised to −2.19 (sigmoid ≈ 0.1), the standard CenterNet trick so training starts by predicting "almost no charges" and stays stable.

### Training targets
For each charge, the target heatmap gets a Gaussian blob (σ = 2 px) centred on its nearest pixel, with value exactly 1 at the centre. The `q` and offset targets are written only at that centre pixel.

### Loss
- **Focal loss** on the heatmap (CenterNet variant, α = 2, β = 4): pixels close to a true charge are only lightly penalised, and the loss concentrates on hard examples. This handles the extreme imbalance (a few charge pixels vs. 40,000 background pixels).
- **L1 loss** on `q` and on the offsets, evaluated **only at charge centre pixels**.

Total = focal + L1(q) + L1(offset), all normalised by the number of charges.

### Data
Training data is generated **on the fly**: each epoch sees brand-new random configurations, so the model cannot memorise them and no dataset is stored in RAM. A fixed test set is generated once and also defines the normalisation constants.

### Inference (decoding)
1. Apply sigmoid to the heatmap.
2. Keep pixels that are a local maximum in a 3×3 neighbourhood (max-pool NMS) **and** above the threshold (0.5).
3. Each surviving peak is one charge: position = pixel + predicted offset (converted to metres), charge = predicted `q × max_q`.

### Evaluation
Predicted and true charges are paired with the **Hungarian algorithm**; a pair counts as a hit if they are within 5 cm. Reported metrics: precision, recall, % of samples with the exact charge count, mean position error, sign accuracy and relative |q| error.
