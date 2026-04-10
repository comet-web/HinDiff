# HinDiff: A Masked Diffusion Language Model for Hindi Text Generation

## Overview

HinDiff transforms a pretrained Hindi BERT encoder into an iterative generative model using the Masked Diffusion Language Model (MDLM) framework. Unlike standard autoregressive decoders that generate text left to right, HinDiff uses discrete diffusion to progressively denoise a fully masked sequence into coherent Hindi text. This represents a paradigm shift: a bidirectional encoder, traditionally limited to understanding tasks, is repurposed for generation by training it to denoise across the full noise spectrum.

## The Paradigm Shift: From Static Masking to Continuous Diffusion

Standard BERT Masked Language Modeling (MLM) trains with a fixed 15% masking rate. The model only ever sees one noise level, which makes it a strong contextual encoder but entirely unsuitable for generation. It cannot reconstruct text from high-entropy, heavily corrupted states because it was never exposed to them during training.

HinDiff discards the static 15% mask rate entirely. Instead, during each forward pass, a continuous time variable `t` is sampled uniformly from `[0, 1]`, and the masking probability for each token is computed using a cosine noise schedule:

```
P_mask(t) = 1 - cos(t * pi / 2)
```

This non-linear schedule ensures the model trains across the full noise spectrum, from near-clean inputs (low `t`) to nearly fully corrupted inputs (high `t`). At low noise levels, the model learns fine-grained grammatical and syntactic refinement. At high noise levels, it learns abstract semantic construction from minimal context. By spanning the entire range, the model acquires the ability to generate text from pure noise through iterative denoising steps.

## Mathematical Foundation

### Cosine Noise Schedule

For each training sample, a random timestep `t ~ U(0,1)` is drawn. The masking probability is:

```
mask_prob = 1 - cos(t * pi / 2)
```

Each token in the sequence is independently masked with this probability. This creates a continuous training distribution over all corruption levels rather than a single fixed point.

### Importance-Weighted Loss

Training across highly variable noise levels introduces gradient instability. A sequence with 90% masking naturally produces a much larger raw cross-entropy loss than one with 10% masking. Without correction, the high-noise samples would dominate the optimizer.

To address this, each sample's loss is scaled by an importance weight:

```
w = N_maskable / N_mask
```

where `N_maskable` is the number of tokens eligible for masking (excluding special tokens like `[CLS]` and `[SEP]`) and `N_mask` is the number of tokens actually masked. This normalizes the gradient contribution per sample regardless of noise level.

Additionally, label smoothing with epsilon = 0.1 is applied to the target distributions. This prevents overconfident predictions that would otherwise destabilize the iterative sampling process during generation.

### Iterative Denoising for Generation

During generation, the model performs reverse diffusion over T = 50 steps:

1. Start with a sequence of all `[MASK]` tokens.
2. At each step, run the model to obtain probability distributions for all positions.
3. Among still-masked positions, identify those with the highest prediction confidence.
4. Reveal those tokens by sampling from their predicted distributions.
5. The number of tokens to reveal decreases as the denoising progresses.
6. Remaining masked tokens are filled in the final step.

Temperature annealing (decaying from 1.2 to 0.1) and Nucleus Top-p filtering (p = 0.9) are used during generation to improve stability and restrict the latent space exploration to highly probable vocabulary combinations.

## Model and Dataset

### Base Model

- **Model**: `l3cube-pune/hindi-bert-v2` (237.8M parameters)
- **Architecture**: BertForMaskedLM from HuggingFace Transformers
- **Vocabulary size**: 197,285 tokens
- **Gradient checkpointing**: Enabled to reduce GPU VRAM usage by approximately 4 GB

### Dataset

The AI4Bharat Samanantar parallel corpus (approximately 10.1 million Hindi-English sentence pairs) is used. From this corpus:

- **Training set**: 100,000 Hindi sentences
- **Validation set**: 3,000 Hindi sentences
- **Test set**: 2,000 Hindi sentences

Sequences are tokenized using the pretrained `l3cube-pune/hindi-bert-v2` tokenizer with a maximum sequence length of 64 tokens. Sentences with fewer than 4 maskable tokens are filtered out.

### Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Standard for fine-tuning BERT-class models |
| Learning rate | 3e-5 | Standard fine-tuning rate |
| Batch size | 32 | Per-device batch size |
| Gradient accumulation | 8 steps | Effective batch size of 256 |
| Epochs | 3 | Pretrained weights converge quickly |
| Label smoothing | 0.1 | Prevents overconfident predictions for diffusion sampling |
| Warmup steps | 200 | Linear warmup before cosine annealing |
| LR scheduler | Cosine annealing | With 5% minimum floor |
| Mixed precision | FP16 via torch.amp | For speed and memory efficiency |
| Gradient clipping | Max norm 1.0 | For training stability |
| Generation steps | 50 | Number of iterative denoising steps |

### Environment

- **Platform**: Google Colab
- **GPU**: NVIDIA Tesla T4 (15.6 GB VRAM)
- **Runtime**: Approximately 63 minutes for 3 epochs
- **Python**: 3.x
- **Key libraries**: `transformers`, `datasets`, `accelerate`, `sentencepiece`, `torch`, `numpy`, `matplotlib`

## Setup Guide

### Prerequisites

- Python 3.8 or later
- A CUDA-capable GPU (NVIDIA T4 or better recommended, minimum 16 GB VRAM)
- pip package manager

### Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/HinDiff.git
cd HinDiff
```

2. Install the required dependencies:

```bash
pip install transformers datasets accelerate sentencepiece torch numpy matplotlib tqdm
```

### Running the Notebook

The project is implemented as a single Jupyter notebook (`HinDiff_prefinal.ipynb`). You can run it:

**On Google Colab (recommended):**

1. Upload `HinDiff_prefinal.ipynb` to Google Colab.
2. Set the runtime to GPU (Runtime > Change runtime type > T4 GPU).
3. Run all cells sequentially. The notebook handles dependency installation, data loading, training, evaluation, and generation.

**Locally:**

1. Ensure CUDA is properly installed and configured.
2. Launch Jupyter Notebook or JupyterLab.
3. Open `HinDiff_prefinal.ipynb` and run all cells.

The notebook downloads the Samanantar dataset, the pretrained Hindi BERT model, and handles the full training pipeline. The complete run takes approximately 63 minutes on a T4 GPU.

## Results

### Validation Performance

The model achieved a best validation Negative Log-Likelihood (NLL) of **2.1145** at epoch 2, with a token recovery accuracy of **58.9%** at a fixed 20% mask rate.

Note: The MDLM validation accuracy (approximately 58%) is expected to be lower than standard BERT MLM accuracy (70-85%) because the model distributes its capacity across all noise levels rather than specializing at a single mask rate.

### Token Recovery Across Noise Levels

The model demonstrates graceful degradation across the full noise spectrum:

| Mask Rate | Recovery Accuracy |
|---|---|
| 5% | 68.1% |
| 10% | 64.6% |
| 15% | 64.4% |
| 20% | 61.5% |
| 30% | 56.4% |
| 40% | 51.3% |
| 50% | 40.4% |
| 60% | 33.3% |
| 70% | 23.8% |
| 80% | 15.4% |

A standard static Hindi BERT baseline (trained only at 15% masking) fails completely beyond 50% corruption. The continuous cosine schedule and importance-weighted loss successfully distribute the model's capacity, allowing it to interpret highly degraded states that crash standard encoders.

### Infilling (Conditional Generation)

In the conditional infilling task with 30% masking, the model successfully restores sentences with contextual awareness, achieving up to 100% accuracy on syntactic targets. For example:

- "शिक्षा के बिना जीवन में सफलता प्राप्त करना बहुत कठिन होता है।" was perfectly restored (100% accuracy).
- "दिल्ली भारत की राजधानी है और यहाँ लाखों लोग रहते हैं।" was also perfectly restored (100% accuracy).

### Unconditional Generation

Pure unconditional generation from 100% noise produces grammatically plausible Hindi at low temperatures. For example:

> "इस तरह की परिस्थिति में निपटने के लिए राज्य सरकार ने कारगर उपायों का काम किया है।"

However, at higher temperatures, the model occasionally produces mixed-language tokens or semantically incoherent outputs. This is an inherent artifact of bidirectional attention lacking strict left-to-right anchoring: at high temperatures during early diffusion steps, the model simultaneously samples random anchor tokens across the sequence and must then reconcile unrelated anchors, sometimes generating valid syntax that bridges a logical gap with no semantic sense.

Top-p sampling and dynamic temperature annealing significantly reduce mixed-language hallucinations and improve structural coherence.

## Future Scope

- **Scaling pretraining compute**: Training on larger subsets of the Samanantar corpus or other Hindi corpora with more epochs and larger batch sizes.
- **Dynamic re-masking**: Allowing the model to backtrack from syntactic dead-ends during generation by re-masking low-confidence tokens.
- **Lexical guidance**: Anchoring the initial latent space exploration with seed tokens to address long-range semantic coherence issues in unconditional generation.
- **Evaluation metrics**: Incorporating perplexity, BLEU, and human evaluation for more rigorous quality assessment of generated text.
- **Downstream applications**: Adapting the MDLM framework for Hindi text summarization, paraphrasing, and data augmentation.

## References

1. Austin, J., Johnson, D. D., Ho, J., Tarlow, D., & van den Berg, R. (2021). Structured Denoising Diffusion Models in Discrete State-Spaces. *NeurIPS 2021*.
2. Chang, H., Zhang, H., Jiang, L., Liu, C., & Freeman, W. T. (2022). MaskGIT: Masked Generative Image Transformer. *CVPR 2022*.
3. Sahoo, S., Arriola, A., & Schuurmans, D. (2024). Simple and Effective Masked Diffusion Language Models. *NeurIPS 2024*.
4. Nichol, A. Q., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. *ICML 2021*.
5. Joshi, R., Goel, P., & Joshi, R. (2022). L3Cube-HindBERT and DevBERT: Pre-Trained BERT Transformer models for Devanagari based Hindi and Marathi Languages. *arXiv:2211.11418*.
6. Ramesh, P., Doddapaneni, S., et al. (2022). Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages. *TACL 2022*.

## License

This project is for academic and research purposes.
