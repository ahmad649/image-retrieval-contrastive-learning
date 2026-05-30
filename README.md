# Image Retrieval with Contrastive Learning — A Study of Triplet Mining Strategies

> Building a content-based image retrieval system using triplet networks, with progressive experiments comparing random sampling against semi-hard triplet mining for learning discriminative embeddings.

**Course:** COMP 6831 – Applied Machine Learning (Concordia University, Microprogram in Applied AI)

---

## Overview

Content-based image retrieval requires a model that maps images into an embedding space where semantically similar images are pulled together and dissimilar images are pushed apart. Triplet loss is one of the most widely used objectives for this, but its effectiveness is highly dependent on *how* triplets are sampled during training.

This project walks through three controlled experiments, each building on the last, to study how triplet mining strategies and validation discipline affect retrieval quality.

## Experiments

### Experiment 1 — Sanity Check and System Validation
Confirmed that the end-to-end pipeline (data loading, triplet construction, triplet loss, embedding extraction, retrieval evaluation) produces meaningful behavior. The goal was correctness, not accuracy.

### Experiment 2 — Baseline Triplet Network with Validation Split
Introduced a proper train/validation split and tracked generalization. Established a clean baseline with random triplet selection that later experiments could be compared against.

### Experiment 3 — Semi-Hard Triplet Mining
Replaced random triplet selection with semi-hard mining, where for each anchor-positive pair the model trains against negatives that are harder than the positive but still within the margin. This forces the network to focus on informative triplets rather than easy or impossibly hard ones.

## Results

| Experiment | Sampling Strategy | Retrieval Accuracy | Notes |
|---|---|---|---|
| 1 — Sanity Check | Random | _add_ | Validated end-to-end pipeline |
| 2 — Baseline | Random | _add_ | First clean train/val measurement |
| 3 — Semi-Hard Mining | Semi-hard | _add_ | Best generalization, sharper embeddings |

*Replace placeholders with your actual metrics (e.g., Recall@1, Recall@5, mAP).*

## Approach

1. **Embedding network** — CNN backbone producing L2-normalized embeddings.
2. **Triplet loss** — anchor-positive-negative loss with configurable margin.
3. **Triplet construction** — implemented both random sampling and semi-hard mining over each training batch.
4. **Evaluation** — retrieval-style metrics (Recall@K, mAP) computed on a held-out validation split, with qualitative inspection of nearest neighbors.

## Tech Stack

`Python` · `PyTorch` · `torchvision` · `NumPy` · `matplotlib` · `Jupyter`

## Repository Structure

```
.
├── notebook/
│   └── image_retrieval_contrastive.ipynb   # Final analysis notebook (all 3 experiments)
├── report/
│   └── image_retrieval_report.pdf          # Full written report
├── figures/                                # Embedding visualizations and retrieval examples
├── requirements.txt
└── README.md
```

## How to Run

```bash
git clone https://github.com/ahmad649/image-retrieval-contrastive-learning.git
cd image-retrieval-contrastive-learning
pip install -r requirements.txt
jupyter notebook notebook/image_retrieval_contrastive.ipynb
```

## Key Takeaways

- Triplet selection matters as much as the architecture: switching from random to semi-hard mining gave a clear retrieval-quality improvement with no change to the backbone.
- Easy triplets (where the negative is already far away) produce near-zero gradient and waste training signal. Semi-hard mining avoids this without the instability of hardest-negative mining.
- A disciplined validation split is essential for triplet networks — leakage between train and val image identities trivially inflates retrieval scores.

## Author

Ahmad Hasan — Master of Applied Computer Science + Microprogram in Applied AI, Concordia University
[LinkedIn](https://www.linkedin.com/in/ahmad-hasan5250/) · [GitHub](https://github.com/ahmad649)

## License

MIT
