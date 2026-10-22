# Design

## Goal

Decode four classes of motor imagery (left hand, right hand, feet, tongue) from
22-channel scalp EEG in real time, and characterise where and why decoding fails.

Dataset: BCI Competition IV 2a (9 subjects, 250 Hz, two sessions recorded on
different days, 288 trials per session).

## Pipeline

1. **Classical baseline.** Causal bandpass, epoching, CSP + LDA.
2. **Deep decoder.** EEGNet, within-subject and leave-one-subject-out (LOSO).
3. **Real-time engine.** ONNX export; Rust service (`ort`) that replays test data
   at 250 Hz, reimplements preprocessing natively, runs sliding-window inference,
   and reports latency per stage (preprocessing / inference / overhead).
4. **Demo.** Decoded classes drive a directional speller grid.
5. **Robustness.** Cross-subject failure analysis, artifact/noise injection,
   window length vs. accuracy and latency.

## Phase 1 in detail

The classical baseline establishes what the signal is before a neural network
abstracts it away.

**Background.** Imagining a movement suppresses 8–30 Hz (mu/beta) oscillations
over the corresponding motor cortex, known as event-related desynchronization.
Right-hand imagery suppresses power over the left hemisphere and vice versa; feet
and tongue produce different spatial patterns. Classes are therefore separable by
*where* band power drops.

1. **Load.** MOABB downloads the dataset. Each run is a continuous recording of
   22 EEG channels at 250 Hz, with event markers for cue onset and class.
2. **Filter.** Causal 8–30 Hz bandpass on each continuous run, isolating the
   mu/beta band and removing drift, muscle noise and line interference.
3. **Epoch.** Slice cue + 0.5 s to cue + 2.5 s at each marker. Each trial becomes a
   22 × 500 array with a label; 288 trials per subject per session.
4. **Spatial filtering (CSP).** Single electrodes are noisy; weighted combinations
   of electrodes give cleaner estimates of activity over a region. Common Spatial
   Patterns learns the weightings whose output power differs most between classes.
   Eight components are kept.
5. **Features.** Log-variance of each component over the window, i.e. band power.
   Each trial reduces from 11,000 values to 8.
6. **Classify.** Linear Discriminant Analysis fits linear boundaries between the
   four classes in that 8-dimensional feature space.
7. **Evaluate.** Fit on session T, score on session E (a different day). Report
   accuracy, kappa and ITR.

## Decisions

### Causal filtering only
Butterworth bandpass as second-order sections, applied with `scipy.signal.sosfilt`,
never `filtfilt`. Zero-phase filtering uses future samples, which a live system
cannot; training on it would create training/serving skew. Filtering is applied to
each continuous run *before* epoching, matching what a stream sees and avoiding
per-epoch edge transients. Initial filter state is `sosfilt_zi(sos)` scaled by each
channel's first sample, to suppress the startup transient from DC offsets.

Tests assert that filtering in small chunks with carried state equals filtering the
whole signal, and that the filter is causal (an impulse at t leaves output before t
unchanged).

### Filter spec as a contract
SOS coefficients, initial-state rule, band, order, sampling rate, channel order,
units and window definition are exported to `artifacts/filters/<name>.json`. The
Rust engine loads this file rather than hard-coding values, so both sides share one
definition.

### Frequency bands
8–30 Hz (mu/beta) for CSP. 4–40 Hz for EEGNet, whose first layer learns its own
temporal filters. The deployed model's band is the one the engine implements.

### EEGNet architecture
EEGNet-8,2 (Lawhern et al., 2018): 8 temporal filters of 64 samples, 2 spatial
(depthwise) filters per temporal filter for 16 virtual channels, a separable
convolution of 16 samples with 16 output maps, average pooling ÷4 then ÷8, and
dropout 0.5. For a 22 × 500 input this is 2,420 parameters. The 64-sample kernel
spans 0.26 s at 250 Hz; the paper's 64 samples were 0.5 s at 128 Hz. Time-axis
padding is explicit (zero-pad, then convolve) so lengths stay 500 and 125. The
model outputs logits; softmax is applied by the loss and at prediction time. The
paper's max-norm weight constraints are omitted.

### EEGNet training
Cross-entropy loss, Adam with learning rate 1e-3, batches of 32. The number of
training epochs is chosen per subject on session T alone: 4-fold stratified
cross-validation, each fold trained for 500 epochs, and the epoch with the best
validation accuracy averaged over folds is kept. A fresh model is then trained on
all 288 trials of session T for that many epochs. Session E is never used for
training or for any choice; it is scored once at the end.

Why not a single validation split with early stopping: the first version held
out 20% of T (58 trials) and stopped after 50 epochs without improvement (max
300). Validation accuracy swung by ±10 points between epochs, so the stopping
point was largely noise; some subjects stopped at epoch 0 and others hit the cap,
and mean kappa was 0.363, below CSP+LDA. Averaging the curves over folds puts all
288 trials into validation once, and retraining on all of T uses the held-out
data too, in the spirit of Schirrmeister et al.'s two-phase training.

Inputs are the filtered µV trials with no further scaling; the BatchNorm after
the first layer handles scale. Seeds fix weight initialisation, fold assignment,
batch order and dropout. Training runs on the Apple GPU (MPS, about 27× faster
than CPU here); MPS results are seeded but not guaranteed bit-identical, so tests
run on CPU.

### Epoch window
MOABB events mark trial start; the cue appears at +2.0 s. The decode window is
cue + 0.5 s to cue + 2.5 s (500 samples). Data are converted to µV.

### Evaluation protocol
Train on session T, test on session E, as in the competition. This measures
session-to-session transfer, not just within-session fit. MOABB session keys vary
by version ("0train"/"1test" vs "session_T"/"session_E") and are mapped
explicitly rather than by sort order.

### Metrics
Accuracy, Cohen's kappa (the official 2a metric), confusion matrix, and Wolpaw
information transfer rate in bits/min. ITR uses the full trial duration (~8 s,
configurable) rather than the 2 s decode window, which would inflate it roughly
fourfold. The trial time used is always reported alongside ITR.

### Reference points
- FBCSP (competition winner): mean kappa 0.569
- EEGNet-class models: roughly 74–80% mean accuracy

## References
- Tangermann et al., *Review of the BCI Competition IV*, Frontiers in Neuroscience, 2012
- Ang et al., *Filter Bank Common Spatial Pattern Algorithm on BCI Competition IV Datasets 2a and 2b*, Frontiers in Neuroscience, 2012
- Blankertz et al., *Optimizing Spatial Filters for Robust EEG Single-Trial Analysis*, IEEE Signal Processing Magazine, 2008
- Lawhern et al., *EEGNet*, Journal of Neural Engineering, 2018
- Schirrmeister et al., *Deep learning with convolutional neural networks for EEG decoding and visualization*, Human Brain Mapping, 2017
- Wolpaw et al., *Brain–computer interfaces for communication and control*, Clinical Neurophysiology, 2002
- Jayaram & Barachant, *MOABB*, Journal of Neural Engineering, 2018
