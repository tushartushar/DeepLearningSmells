import os
import autoencoder

TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/tokenizer_out/"
OUT_FOLDER = "/users/pa18/tushar/smellDetectionML/learning_smells/results/rq1/raw"
# TOKENIZER_OUT_PATH = r"..\..\data\tokenizer_out"
# OUT_FOLDER = r"..\results\rq1\raw"

smell_list = ["MultifacetedAbstraction"]
DIM = "1d"

for smell in smell_list:
    data_path = os.path.join(TOKENIZER_OUT_PATH, smell, DIM)
    autoencoder.main_lstm(smell, data_path, skip_iter=5)