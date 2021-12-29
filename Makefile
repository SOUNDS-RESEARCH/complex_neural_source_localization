 .PHONY: install clean lint preprocessing

install:
	@pip install -r requirements.txt

lint:
	@flake8

test:
	@pytest tests/

test-train:
	@python -m neural_tdoa.test_train

train:
	@python train.py