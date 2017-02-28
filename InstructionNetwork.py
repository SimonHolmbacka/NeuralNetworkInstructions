import numpy
import os
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Dropout, Activation, Flatten
from keras.layers.core import TimeDistributedDense
from keras.layers import TimeDistributed
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from scikits.statsmodels.api import categorical
# fix random seed for reproducibility
numpy.random.seed(7)

#Original training set begin
p_ack_a15 = 1.35
p_ack_a7 = 0.73
perf_ack = 2.026

p_i32d_a15 = 1.65
p_i32d_a7 = 0.595
perf_i32d = 1.64

p_ln2_a15 = 0.92
p_ln2_a7 = 0.319
perf_ln2 = 1.343

p_cdb_a15 = 1.82
p_cdb_a7 = 0.53
perf_cdb = 0.513

p_call_a15 = 1.68
p_call_a7 = 0.897
perf_call = 1.297

p_dit_a15 = 1.89
p_dit_a7 = 0.568
perf_dit = 0.878

p_eul_a15 = 1.12
p_eul_a7 = 0.345
perf_eul = 0.752

p_fib_a15 = 1.39
p_fib_a7 = 0.612
perf_fib = 1.210

p_gam_a15 = 1.71
p_gam_a7 = 0.556
perf_gam = 0.686

p_i32f_a15 = 2.08
p_i32f_a7 = 0.664
perf_i32f = 0.770
#Original training set end

total_error = 0


def column(matrix, i):
    return [row[i] for row in matrix]


#Load files
#Dictionary for instructions
X_Dictionary_str = numpy.genfromtxt("input/input_Dict.csv",dtype=str)
#The training set
X_train_str = numpy.genfromtxt("input/input_instructionsTRAIN15.csv",dtype=str,delimiter=',')
X_train_str = X_train_str.transpose()
#The target set
Y_train = numpy.loadtxt("input/input_power_train15.csv", delimiter=",")
#The test set
X_test_str = numpy.genfromtxt("input/input_instructionsTEST.csv", dtype=str,delimiter=',')
X_test_str = X_test_str.transpose()
#The test target set
Y_test = numpy.loadtxt("input/input_power_test.csv", delimiter=",")

#Number of applications used for training
napps = 15
ntestapps = 10
seqlen = 2000

#We make all strings to an identification integer
vocab = set()
for i in range(0,napps):
    for instruction in list(X_train_str[i])+list(X_Dictionary_str):
        vocab.add(instruction)


for i in range(0,ntestapps):
    for instruction in list(X_test_str[i]):
        vocab.add(instruction)



X_Dictionary = dict((t, i) for i, t in enumerate(vocab))
id2token = dict((i, t) for i, t in enumerate(vocab))
numpy.set_printoptions(threshold=numpy.nan)

tshape_train = (napps,seqlen)
tshape_test = (ntestapps,seqlen)
X_train = numpy.zeros(tshape_train)
X_test = numpy.zeros(tshape_test)
for i in range(0,napps):
    X_train[i] = [X_Dictionary[x] for x in X_train_str[i]] #string to int
for i in range(0,ntestapps):
    X_test[i] = [X_Dictionary[x] for x in X_test_str[i]] #string to int

#X_train = numpy.reshape(X_train,(10,2000))
#X_test = pad_sequences([X_test], maxlen=20000, dtype='float32')
#X_test = numpy.reshape(X_test,(10,2000))


max_len = seqlen
batch_size = 1
member_berries = 100
dropout = 0.1
rows = 2000 #100 gives 368% error 30 gives 516% error
epoch = 25 #30 gives almost perfect
load_old_model = 1


#Choose to make a new model (will take some time)
if load_old_model == 0:
    input_layer = Input(shape=(max_len,))
    emb_layer = Embedding(len(X_Dictionary), 2000, input_length=max_len)(input_layer)
    lstm_layer = LSTM(member_berries,go_backwards=False, dropout_W=dropout)(emb_layer)
    output_layer = Dense(3, init='glorot_normal', activation='softplus')(lstm_layer)
    model = Model(input=input_layer, output=output_layer)
    opt = optimizers.Adam(lr=0.0001)
    model.compile(loss='mse',optimizer=opt) #mse adam
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epoch, validation_data=(X_test, Y_test))
    #model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epoch, validation_split=0.1)
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
#Choose to use a saved model (Will be fast)
elif load_old_model == 1:
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data
    model.compile(loss='mse',optimizer='adam')
else:
    print ("Error in model load/store")


#Predictions

###Ackermann###
if rows == 100:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_ack_100.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_ack_30.txt",dtype=str)
elif rows == 2000:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_ack_2000.txt",dtype=str)
else:
    print("Invalid row number")
X_predict = [X_Dictionary[x] for x in X_predict_str] #string to int
X_predict = pad_sequences([X_predict], maxlen=max_len, dtype='float32')
X_predict = numpy.reshape(X_predict,(1,max_len))
predictions = model.predict(X_predict)
print("Ackermann")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
error_a15 = abs((numpy.average(column(predictions,0))/p_ack_a15)-1.0)
error_a7 = abs((numpy.average(column(predictions,1))/p_ack_a7)-1.0)
error_perf = abs((numpy.average(column(predictions,2))/perf_ack)-1.0)
total_error += error_a15 + error_a7 + error_perf
print(round(error_a15,2),round(error_a7,2),round(error_perf,2))
print()

###Int32Double###
if rows == 100:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_i32d_100.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_i32d_30.txt",dtype=str)
elif rows == 2000:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_i32d_2000.txt",dtype=str)
else:
    print("Invalid row number")
X_predict = [X_Dictionary[x] for x in X_predict_str] #string to int
X_predict = pad_sequences([X_predict], maxlen=max_len, dtype='float32')
X_predict = numpy.reshape(X_predict,(1,max_len))
predictions = model.predict(X_predict)
print("Int32Double")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
error_a15 = abs((numpy.average(column(predictions,0))/p_i32d_a15)-1.0)
error_a7 = abs((numpy.average(column(predictions,1))/p_i32d_a7)-1.0)
error_perf = abs((numpy.average(column(predictions,2))/perf_i32d)-1.0)
total_error += error_a15 + error_a7 + error_perf
print(round(error_a15,2),round(error_a7,2),round(error_perf,2))
print()

###ln2###
if rows == 100:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_ln2_100.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_ln2_30.txt",dtype=str)
elif rows == 2000:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_ln2_2000.txt",dtype=str)
else:
    print("Invalid row number")

X_predict = [X_Dictionary[x] for x in X_predict_str] #string to int
X_predict = pad_sequences([X_predict], maxlen=max_len, dtype='float32')
X_predict = numpy.reshape(X_predict,(1,max_len))
predictions = model.predict(X_predict)
print("ln2")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
error_a15 = abs((numpy.average(column(predictions,0))/p_ln2_a15)-1.0)
error_a7 = abs((numpy.average(column(predictions,1))/p_ln2_a7)-1.0)
error_perf = abs((numpy.average(column(predictions,2))/perf_ln2)-1.0)
total_error += error_a15 + error_a7 + error_perf
print(round(error_a15,2),round(error_a7,2),round(error_perf,2))
print()

###Cdouble###
if rows == 100:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_cdouble_100.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_cdouble_30.txt",dtype=str)
elif rows == 2000:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_cdouble_2000.txt",dtype=str)

else:
    print("Invalid row number")
X_predict = [X_Dictionary[x] for x in X_predict_str] #string to int
X_predict = pad_sequences([X_predict], maxlen=max_len, dtype='float32')
X_predict = numpy.reshape(X_predict,(1,max_len))
predictions = model.predict(X_predict)
print("Cdouble")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
error_a15 = abs((numpy.average(column(predictions,0))/p_cdb_a15)-1.0)
error_a7 = abs((numpy.average(column(predictions,1))/p_cdb_a7)-1.0)
error_perf = abs((numpy.average(column(predictions,2))/perf_cdb)-1.0)
total_error += error_a15 + error_a7 + error_perf
print(round(error_a15,2),round(error_a7,2),round(error_perf,2))
print()

###Callfunc###
if rows == 100:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_callfunc_100.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_callfunc_30.txt",dtype=str)
elif rows == 2000:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_callfunc_2000.txt",dtype=str)

else:
    print("Invalid row number")
X_predict = [X_Dictionary[x] for x in X_predict_str] #string to int
X_predict = pad_sequences([X_predict], maxlen=max_len, dtype='float32')
X_predict = numpy.reshape(X_predict,(1,max_len))
predictions = model.predict(X_predict)
print("Callfunc")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
error_a15 = abs((numpy.average(column(predictions,0))/p_call_a15)-1.0)
error_a7 = abs((numpy.average(column(predictions,1))/p_call_a7)-1.0)
error_perf = abs((numpy.average(column(predictions,2))/perf_call)-1.0)
total_error += error_a15 + error_a7 + error_perf
print(round(error_a15,2),round(error_a7,2),round(error_perf,2))
print()

###Dither###
if rows == 100:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_dither_100.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_dither_30.txt",dtype=str)
elif rows == 2000:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_dither_2000.txt",dtype=str)

else:
    print("Invalid row number")
X_predict = [X_Dictionary[x] for x in X_predict_str] #string to int
X_predict = pad_sequences([X_predict], maxlen=max_len, dtype='float32')
X_predict = numpy.reshape(X_predict,(1,max_len))
predictions = model.predict(X_predict)

print("Dither")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
error_a15 = abs((numpy.average(column(predictions,0))/p_dit_a15)-1.0)
error_a7 = abs((numpy.average(column(predictions,1))/p_dit_a7)-1.0)
error_perf = abs((numpy.average(column(predictions,2))/perf_dit)-1.0)
total_error += error_a15 + error_a7 + error_perf
print(round(error_a15,2),round(error_a7,2),round(error_perf,2))
print()

###Euler###
if rows == 100:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_euler_100.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_euler_30.txt",dtype=str)
elif rows == 2000:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_euler_2000.txt",dtype=str)

else:
    print("Invalid row number")
X_predict = [X_Dictionary[x] for x in X_predict_str] #string to int
X_predict = pad_sequences([X_predict], maxlen=max_len, dtype='float32')
X_predict = numpy.reshape(X_predict,(1,max_len))
predictions = model.predict(X_predict)
print("Euler")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
error_a15 = abs((numpy.average(column(predictions,0))/p_eul_a15)-1.0)
error_a7 = abs((numpy.average(column(predictions,1))/p_eul_a7)-1.0)
error_perf = abs((numpy.average(column(predictions,2))/perf_eul)-1.0)
total_error += error_a15 + error_a7 + error_perf
print(round(error_a15,2),round(error_a7,2),round(error_perf,2))
print()

###Fibonacci###
if rows == 100:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_fibonacci_100.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_fibonacci_30.txt",dtype=str)
elif rows == 2000:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_fibonacci_2000.txt",dtype=str)

else:
    print("Invalid row number")
X_predict = [X_Dictionary[x] for x in X_predict_str] #string to int
X_predict = pad_sequences([X_predict], maxlen=max_len, dtype='float32')
X_predict = numpy.reshape(X_predict,(1,max_len))
predictions = model.predict(X_predict)
print("Fibonacci")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
error_a15 = abs((numpy.average(column(predictions,0))/p_fib_a15)-1.0)
error_a7 = abs((numpy.average(column(predictions,1))/p_fib_a7)-1.0)
error_perf = abs((numpy.average(column(predictions,2))/perf_fib)-1.0)
total_error += error_a15 + error_a7 + error_perf
print(round(error_a15,2),round(error_a7,2),round(error_perf,2))
print()

###Gamma###
if rows == 100:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_gamma_100.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_gamma_30.txt",dtype=str)
elif rows == 2000:
        X_predict_str = numpy.genfromtxt("input/predict/input_predict_gamma_2000.txt",dtype=str)
else:
    print("Invalid row number")
X_predict = [X_Dictionary[x] for x in X_predict_str] #string to int
X_predict = pad_sequences([X_predict], maxlen=max_len, dtype='float32')
X_predict = numpy.reshape(X_predict,(1,max_len))
predictions = model.predict(X_predict)
print("Gamma")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
error_a15 = abs((numpy.average(column(predictions,0))/p_gam_a15)-1.0)
error_a7 = abs((numpy.average(column(predictions,1))/p_gam_a7)-1.0)
error_perf = abs((numpy.average(column(predictions,2))/perf_gam)-1.0)
total_error += error_a15 + error_a7 + error_perf
print(round(error_a15,2),round(error_a7,2),round(error_perf,2))
print()

###Int32Float###
if rows == 100:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_i32f_100.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_i32f_30.txt",dtype=str)
elif rows == 2000:
    X_predict_str = numpy.genfromtxt("input/predict/input_predict_i32f_2000.txt",dtype=str)

else:
    print("Invalid row number")
X_predict = [X_Dictionary[x] for x in X_predict_str] #string to int
X_predict = pad_sequences([X_predict], maxlen=max_len, dtype='float32')
X_predict = numpy.reshape(X_predict,(1,max_len))
predictions = model.predict(X_predict)
print("I32f")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
error_a15 = abs((numpy.average(column(predictions,0))/p_i32f_a15)-1.0)
error_a7 = abs((numpy.average(column(predictions,1))/p_i32f_a7)-1.0)
error_perf = abs((numpy.average(column(predictions,2))/perf_i32f)-1.0)
total_error += error_a15 + error_a7 + error_perf
print(round(error_a15,2),round(error_a7,2),round(error_perf,2))
print()

###Test###
X_predict_str = numpy.genfromtxt("input/input_test.txt",dtype=str)

X_predict = [X_Dictionary[x] for x in X_predict_str] #string to int
X_predict = pad_sequences([X_predict], maxlen=max_len, dtype='float32')
X_predict = numpy.reshape(X_predict,(1,max_len))
predictions = model.predict(X_predict)
print("Test")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
print()

print("Total error ",round(total_error*100,2), "%")
