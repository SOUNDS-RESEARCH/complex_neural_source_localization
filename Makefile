 .PHONY: install clean lint preprocessing

install:
	@pip install -r requirements.txt

lint:
	@flake8

test:
	@pytest tests/

download-dcase-2021:
	@python -m datasets.dcase_2021.download

preprocessing:
	@python -m preprocessing.split_dcase_2021_samples

test-train:
	@python -m yoho.test_train