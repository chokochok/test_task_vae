# Task Definition

## TEST TASK: VAE FOR IMAGE GENERATION

### Goal
Train a Variational Autoencoder that can:
- reconstruct input images well, and
- generate plausible new images by sampling from the latent space.

You must demonstrate you can debug typical VAE failure modes and provide quantitative + qualitative evidence.

### Dataset
Pick one dataset below:
- CIFAR-10 (32×32, RGB);
- CelebA (crop/resize to 64×64, RGB);
- Fashion-MNIST (28×28, grayscale).

You may use torchvision dataset loaders if available.

### Requirements

#### Implement a convolutional VAE
- Build a VAE with Encoder, Decoder, Reparameterization.
- You must write down and implement the ELBO loss.

#### Make it train stably
Add at least two of the following (and explain why you chose them):
- KL warm-up / annealing schedule
- β-VAE (β ≠ 1)
- Gradient clipping
- Free bits
- Better decoder likelihood (choosing a likelihood model that matches image data better)

If you diagnose any training issues, you must explicitly show the evidence.

#### Evaluation
Provide qualitative and quantitative evaluations.
Also include a "failure gallery", showing 10–20 bad samples/reconstructions and hypothesize why.

### Engineering deliverables
A GitHub repository that contains (minimal requirements):
- `train.py` that runs end-to-end;
- `evaluate.py` producing metrics + image grids;
- reproducible config (JSON/YAML/etc.);
- clear folder structure;
- saved artifacts: checkpoints, generated images, training curves (loss, recon term, KL term);
- `README.md` with a report.

### Candidate report (1–2 pages max)
MUST HAVE:
- dataset & preprocessing;
- architecture summary + parameter counts;
- loss design + why;
- metrics + grids;
- top issues encountered and fixes;
- next steps (what you'd do to improve sharpness/diversity).

### What we will grade
- **Correctness (30%)**: proper ELBO, reparameterization, stable training
- **Generative quality (25%)**: sample realism/diversity + your metric
- **Debugging maturity (20%)**: you identify issues and fix them with justified changes
- **Engineering (15%)**: reproducible runs, clean code, clear outputs
- **Communication (10%)**: concise report with evidence and trade-offs

### Execution Time
You will have up to 2 days to complete the task.
