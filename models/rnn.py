from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import pandas as pd
#import matplotlib.pyplot as plt
from keras.optimizers import Adam

vocab_size = 1500
embedding_size = 100
seq_len = 15
categories = 7

df_train = pd.read_csv('../processed/deepdive_train.csv', index_col=0)
df_test = pd.read_csv('../processed/deepdive_test.csv', index_col=0)
df_val = pd.read_csv('../processed/deepdive_val.csv', index_col=0)

tokenizer = Tokenizer(num_words=vocab_size - 1, oov_token=None)
tokenizer.fit_on_texts(df_train.text)

X_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=seq_len)
X_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=seq_len)
X_val = pad_sequences(tokenizer.texts_to_sequences(df_val.text), maxlen=seq_len)
y_train = to_categorical(df_train.category, categories)
y_test = to_categorical(df_test.category, categories)
y_val = to_categorical(df_val.category, categories)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=seq_len))
model.add(Bidirectional(LSTM(128, recurrent_dropout=0.7, return_sequences=True)))
model.add(Bidirectional(LSTM(128, recurrent_dropout=0.7)))
model.add(Dense(categories, activation='softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=2, validation_data=(X_test, y_test))

model.save('model1.h5')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

