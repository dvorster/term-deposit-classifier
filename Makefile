# Makefile for term deposit classifier pipeline

# Default target
all: report/index.html report/index.pdf

# Download raw data
data/raw/raw_data_sample.csv:
	python scripts/download_data.py --id=222 --write_to=data/raw

# Validate raw data
validate: data/raw/raw_data_sample.csv
	python scripts/data_validation.py --raw_data=data/raw/raw_data_sample.csv

# Exploratory Data Analysis
eda: data/raw/raw_data_sample.csv
	python scripts/eda.py \
		--loaded-data data/raw/raw_data_sample.csv \
		--processed-data data/processed_data \
		--plot-to results/figures

# Preprocess data for training
preprocess: eda
	python scripts/preprocess.py \
		--train-csv-file data/processed_data/train.csv \
		--test-csv-file data/processed_data/test.csv \
		--data-to data/processed_data \
		--preprocessor-to results/models \
		--plot-to results/figures

# Train the classifier
train: preprocess
	python scripts/term_deposit_classifier.py \
		--processed-train-data data/processed_data/preprocess_train.csv \
		--preprocessor results/models/data_preprocessor.pickle \
		--pipeline-to results/models \
		--plot-to results/figures \
		--table-to results/tables \
		--target-col target \
		--seed 522

# Evaluate the model
evaluate: train
	python scripts/evaluate_term_deposit_classifier.py \
		--processed-test-data=data/processed_data/preprocess_test.csv \
		--pipeline-from=results/models/svc_pipeline.pickle \
		--plot-to=results/figures \
		--table-to=results/tables \
		--target-col=target

# Generate final report
report/index.html: evaluate report/term-deposit-analysis.qmd
	quarto render report/term-deposit-analysis.qmd --to html

# Generate final report
report/index.pdf: evaluate report/term-deposit-analysis.qmd
	quarto render report/term-deposit-analysis.qmd --to pdf

# Clean up generated files
clean:
	rm -rf data/processed_data/* results/figures/* results/models/* results/tables/* report/index.html report/index.pdf

.PHONY: all validate eda preprocess train evaluate clean