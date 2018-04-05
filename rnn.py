from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import pandas as pd

vocab_size = 1000
embedding_size = 100
seq_len = 15
categories = 7

df_train = pd.read_csv('processed/deepdive_train.csv', index_col=0)
df_test = pd.read_csv('processed/deepdive_test.csv', index_col=0)
df_val = pd.read_csv('processed/deepdive_val.csv', index_col=0)
tokenizer = Tokenizer(num_words=vocab_size - 1, oov_token=None)
tokenizer.fit_on_texts(df_train.text)
X_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=seq_len)
X_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=seq_len)
X_val = pad_sequences(tokenizer.texts_to_sequences(df_val.text), maxlen=seq_len)
y_train = to_categorical(df_train.category, categories)
y_test = to_categorical(df_test.category, categories)
y_val = to_categorical(df_val.category, categories)
# FUNCTION TO IMPORT THE SPELLCHECKED DATA. IMPORT 4 arrays

# Tokenize it
token = Tokenizer(num_words=None, oov_token='<UNK>')

# Pad Sequences
x_train = sequence.pad_sequences(token.texts_to_sequences(['TEXT IN HERE']), maxlen=15)
x_test = sequence.pad_sequences(token.texts_to_sequences(['TEXT IN HERE']), maxlen=15)

embedding_vector_length = 32

model = Sequential()
model.add(Embedding(input_dim='VOCAB SIZE', output_dim="OUTPUT", input_length=10))
model.add(LSTM(128, ))
