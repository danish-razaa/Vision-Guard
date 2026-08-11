# VisionGuard

VisionGuard is a defensive academic project for detecting controlled,
adversarial-like image perturbation simulations before images are supplied to
vision-language models. The planned system combines multi-domain image features,
classical machine-learning classifiers, explainability, and a Flask web interface.

## Development status

Chapter 1 establishes the repository structure and configuration only. No dataset
has been downloaded, no model has been trained, and no experimental metrics are
reported yet.

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
