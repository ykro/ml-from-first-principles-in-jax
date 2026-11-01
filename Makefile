PYTHON ?= $(shell if [ -x ../../.venv-shared/bin/python ]; then echo ../../.venv-shared/bin/python; else command -v python3; fi)
PYTEST ?= $(shell if [ -x ../../.venv-shared/bin/pytest ]; then echo ../../.venv-shared/bin/pytest; else command -v pytest; fi)
RUFF ?= $(shell if [ -x ../../.venv-shared/bin/ruff ]; then echo ../../.venv-shared/bin/ruff; else command -v ruff; fi)

.PHONY: all experiments verify test lint clean gcp-gpu gcp-tpu

all: lint test verify

experiments:
	$(PYTHON) experiments/01_rules_vs_learned_weights.py
	$(PYTHON) experiments/02_gradient_descent_geometry.py
	$(PYTHON) experiments/03_logistic_classification_cross_entropy.py
	$(PYTHON) experiments/04_two_layer_mlp_from_scratch.py
	$(PYTHON) experiments/05_overfitting_and_weight_decay.py

verify:
	$(PYTHON) verify_ml_principles.py

test:
	$(PYTEST) -v

lint:
	$(RUFF) check .

gcp-gpu:
	bash gcp/run_on_demand_ml.sh gpu

gcp-tpu:
	bash gcp/run_on_demand_ml.sh tpu

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache experiments/__pycache__ tests/__pycache__
