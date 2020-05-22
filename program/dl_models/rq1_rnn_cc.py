import os
import rq1_rnn_emb_lstm as rnn
import inputs

# --- Parameters --
DIM = "1d"
C2V = True # It means whether we are analyzing plain source code that is tokenized (False) or vectors from Code2Vec (True)

if C2V:
    # TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/c2v_vectors/"
    TOKENIZER_OUT_PATH = r"..\..\data\c2v_vectors"
else:
    # TOKENIZER_OUT_PATH = "../../data/tokenizer_out_cs/"
    TOKENIZER_OUT_PATH = "/users/pa18/tushar/smellDetectionML/data/c2v_vectors/"
# ---

smell = "ComplexConditional"
if C2V:
    data_path = os.path.join(TOKENIZER_OUT_PATH, smell)
    inputs.preprocess_data_c2v(data_path)
else:
    data_path = os.path.join(TOKENIZER_OUT_PATH, smell, DIM)
    inputs.preprocess_data(data_path)
rnn.main(data_path, smell)