import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.core import TimeDistributedDense
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)

#load the dataset but only keep the top n words, zero the rest
#top_words = 5000
#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
X_train = numpy.genfromtxt("input_train.csv", delimiter=",",dtype=str)
Y_train = numpy.loadtxt("input_target.csv", delimiter=",")

X_test = numpy.genfromtxt("input_test.csv", delimiter=",", dtype=str)
Y_test = numpy.loadtxt("input_testtarget.csv", delimiter=",")

X_predict = numpy.genfromtxt("input_predict.csv", delimiter=",",dtype=str)

#truncate and pad input sequences
#max_review_length = 500
#X_train = dtrain
#X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
#X_test = dtest
#X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

#create the model
embedding_vecor_length = 32
model = Sequential()
#model.add(Dense(20, input_dim=1, init='glorot_uniform', activation='relu'))
model.add(TimeDistributedDense(10,input_dim=10)) # output shape: (nb_samples, timesteps, 10)
#model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
#model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
#model.add(MaxPooling1D(pool_length=2))
#model.add(LSTM(100))
#model.add(TimeDistributedDense(10,input_dim=1)) # output shape: (nb_samples, timesteps, 10)
model.add(LSTM(10, return_sequences=True)) # output shape: (nb_samples, timesteps, 10)
#model.add(Dropout(0.2))
#model.add(Dense(1, activation='sigmoid'))
model.add(Dense(output_dim=3, init='glorot_uniform', activation='softplus'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=3, batch_size=64)




# Final evaluation of the model
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))
