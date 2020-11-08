import os
import rq2_rnn_emb_lstm as rnn
import inputs

# --- Parameters --
DIM = "1d"

MODE = "train_cs" # "train_cs" or "train_java" representing whether the training data is coming from C# samples or java.

if MODE == "train_cs":
    # TRAINING_TOKENIZER_OUT_PATH = "../../data/tokenizer_out_cs/"
    # EVAL_TOKENIZER_OUT_PATH = "../../data/tokenizer_out_java/"
    # OUT_FOLDER = "../results/rq2/raw"
    TRAINING_TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/tokenizer_out/"
    EVAL_TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/tokenizer_out_java/"
    OUT_FOLDER = "/users/pa18/tushar/smellDetectionML/learning_smells/results/rq2/raw"
else:
    # TRAINING_TOKENIZER_OUT_PATH = "../../data/java_500/tokenizer_out/"
    # EVAL_TOKENIZER_OUT_PATH = "../../data/cs_100/tokenizer_out/"
    # OUT_FOLDER = "../results/rq2/raw"
    TRAINING_TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/java_500/tokenizer_out/"
    EVAL_TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/cs_100/tokenizer_out/"
    OUT_FOLDER = "/users/pa18/tushar/smellDetectionML/learning_smells/results/rq2/raw"
# -----

smell = "ComplexConditional"

training_data_path = os.path.join(os.path.join(TRAINING_TOKENIZER_OUT_PATH, smell), DIM)
eval_data_path = os.path.join(os.path.join(EVAL_TOKENIZER_OUT_PATH, smell), DIM)
inputs.preprocess_data(training_data_path)
inputs.preprocess_data(eval_data_path)
rnn.main(training_data_path, eval_data_path, smell)