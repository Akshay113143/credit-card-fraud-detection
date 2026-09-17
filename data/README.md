# Data

Put `creditcard.csv` in this folder. It is not committed (150 MB, Kaggle terms).

- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- With the Kaggle CLI: `kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip`
- Expected: 284,807 rows, 31 columns, 492 frauds
- SHA-256 of the file used for the results: `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`

The code stops with an error if the file is missing; it never substitutes synthetic data.
