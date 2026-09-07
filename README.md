# VisionGuard

VisionGuard is a defensive academic project for detecting controlled,
adversarial-like image perturbation simulations before images are supplied to
vision-language models. The planned system combines multi-domain image features,
classical machine-learning classifiers, explainability, and a Flask web interface.

VisionGuard detects statistical, spectral, texture, and residual anomalies
associated with controlled perturbation distributions represented during
training and evaluates generalization to held-out perturbation families and
prompts. It does not claim universal detection of VLM soft prompts.

## Current model system

The web application provides one approved production RandomForest detector and
five selectable development candidates: RandomForest, XGBoost, LightGBM,
CatBoost, and SVM. The production artifact uses its saved 45-feature schema; the
development artifacts use the enhanced 65-feature schema and their own
validation-selected thresholds. Development labels are retained because these
experimental artifacts have not replaced the production model.

The Prompt Injection Lab also uses a separately trained, reference-aware pair
detector. It consumes 130 signed and absolute feature-delta values from an
original/modified image pair. It is not a standalone image scanner.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
python3 -m compileall .
pytest -q
```

Python 3.10 or newer is required.

## Scientific scope

The initial generated samples in later chapters are controlled adversarial-like
perturbation simulations. They must not be described as genuine optimized VLM
soft-prompts or as attacks against external commercial services.

## Prompt Injection Lab

The local `/prompt-lab` page turns a user prompt into a deterministic,
prompt-conditioned controlled perturbation. SHA-256 supplies a stable seed that
controls spatial frequency, phase, orientation, and RGB channel weights. Four
safe strengths bound the per-channel perturbation to epsilon values of 0.5,
1.0, 2.0, or 4.0 pixels; output is clipped and normalized to PNG.

The lab runs the user-selected standalone detector and XAI pipeline on both the
normalized original and modified copy, then runs the reference-aware pair
detector. It reports actual probabilities, model identity, risk change, pair
confidence, top model attributions, and the largest Statistical, DCT,
FFT/Spectral, Texture, and Residual feature changes. Raw prompts are not stored
in experiment metadata and no external model or commercial API is contacted.

The prompt-conditioned perturbation generator is a controlled research
simulator and is not evidence of a successful optimized soft-prompt attack
against a specific VLM. Its results measure only how the current detector reacts
to this synthetic bounded perturbation family.
