class CNN_config:
    def __init__(self, layers, filters, kernel, pooling_window, epochs):
        self.layers = layers
        self.filters = filters
        self.kernel = kernel
        self.pooling_window = pooling_window
        self.epochs = epochs

class RNN_config:
    def __init__(self, lstm_layers, lstm_units, batch_size, epochs):
        self.lstm_layers = lstm_layers
        self.lstm_units = lstm_units
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = 0.2,
        self.shuffle = True
        
class RNN_emb_lstm_config:
    def __init__(self, emb_output, lstm_units, layers, epochs, dropout):
        self.emb_output=emb_output
        self.lstm_units = lstm_units
        self.layers = layers
        self.epochs = epochs
        self.shuffle = True
        self.dropout=dropout
        
