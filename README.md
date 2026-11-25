# EM_SIMULACION
 
Work in progress.
 
A PyTorch neural network that detects point charges (number, position and value) from a noisy 2D electric field. The number of charges is not known in advance.


 ### HOW TO USE
 
 1. Clone the repository locally
 2. If you want to train your own model, delete the model file. If not, go to step 3.
 3. Run the .py file. It will generate a charge dataset and plot some examples. If you decided to train, train/test loss will be outputted.
 4. Enjoy


### DESCRIPTION OF THE MODEL

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
