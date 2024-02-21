# ShakespeareBot
A simple chatbot created from a recurrent neural network trained on eight plays by William Shakespeare

Exploratory analysis was performed before making models, looking at TD-IDF, knowledge graphs generated using spaCy, Latent Semantic Analysis, and Latent Dirichlet Allocation.

4 RNN models were created then tested for accuracy before choosing one to create the chatbot.
The eight plays were first split into sentences using the NLTK Python package. Then the Tokenizer function from TensorFlow Keras was then used to transform the sentences into token sequences, then split into 86,000+ n-grams. Due to computational resources, 30,000 n-grams were selected randomly for model training.

The models were variations of the same model:
  -> Embedding layer (128 or 64 dimensions)
  -> 2 or 3 Bidirectional LSTM layers (128-64, 128-64-32, or 64-32 neurons)
  -> 2 Dense Layers
The most accurate model (0.832 ACC, 0.232 VAL ACC) used 64 dimension for embedding then 2 Bi-LSTM layers (64, 32 neurons).
