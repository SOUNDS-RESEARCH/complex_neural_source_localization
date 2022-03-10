 .PHONY: install clean lint preprocessing

test:
	@pytest tests/

train:
	@python train.py

visualizations:
	@python visualizations.py

install:
	@pip install -r requirements.txt

lint:
	@flake8