.PHONY: install test run notebook

install:
	pip install -r requirements.txt

test:
	python -m pytest -q

run:
	python scripts/run_pipeline.py

notebook:
	cd notebooks && jupyter nbconvert --to notebook --execute --inplace credit_card_fraud_detection.ipynb
