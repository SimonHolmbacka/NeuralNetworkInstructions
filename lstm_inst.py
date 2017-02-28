import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout, Activation, Flatten
from keras.layers.core import TimeDistributedDense
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from scikits.statsmodels.api import categorical
# fix random seed for reproducibility
numpy.random.seed(7)
p_ack_a15 = 1.35
p_ack_a7 = 0.73
perf_ack = 2.026

p_cdb_a15 = 1.82
p_cdb_a7 = 0.53
perf_cdb = 0.513

p_i32d_a15 = 1.65
p_i32d_a7 = 0.595
perf_i32d = 1.64

p_ln2_a15 = 0.92
p_ln2_a7 = 0.319
perf_ln2 = 1.343

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

total_error = 0

def getInstructionIdentifier(array_str, array_nr, str):
    for x in range (0, array_str.size-1):
        if array_str[x] == str:
            return array_nr[x]
    print("Error could not find instruction")
    return -100


def column(matrix, i):
    return [row[i] for row in matrix]


#We make all strings to an identification integer
#X_Dictionary_str = numpy.genfromtxt("input/input_train.csv",dtype=str)
X_Dictionary_str = numpy.genfromtxt("input/input_Dict.csv",dtype=str)
X_Dictionary = categorical(X_Dictionary_str, drop=True)
X_Dictionary = X_Dictionary.argmax(1)


#X_train_str = numpy.genfromtxt("input/input_train.csv",dtype=str)
#X_train = categorical(X_train_str, drop=True)
#X_train = X_train.argmax(1)
X_train_str = X_Dictionary_str
X_train = X_Dictionary


#Target power and performance values
Y_train = numpy.loadtxt("input/input_target.csv", delimiter=",")

#Test
X_test_str = numpy.genfromtxt("input/input_test.csv", dtype=str)
X_test = numpy.zeros(X_test_str.size)
for x in range (0, X_test_str.size-1):
    X_test[x] = getInstructionIdentifier(X_train_str,X_train, X_test_str[x])
    if X_test[x] == -100:
        print "Could not find the instruction ", X_test_str[x], " please add it to the dictionary!"

#Float
Y_test = numpy.loadtxt("input/input_testtarget.csv", delimiter=",")


max_features = 38000
batch_size = 20 #20
member_berries = 200 #should be more member berries if input sequence is longer (512 seems to be the max)
dropout = 0.1
rows = 30
epoch = 2

#create the model
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(Convolution1D(nb_filter=2, filter_length=2, border_mode='same', activation='linear'))
model.add(LSTM(member_berries))
model.add(Dropout(dropout)) #Applies Dropout to the input. Dropout consists in randomly setting a fraction p of input units to 0 at each update during training time, which helps prevent overfitting.
model.add(Dense(output_dim=3, init='glorot_normal', activation='softplus'))

model.compile(loss='mean_absolute_error',optimizer='nadam',metrics=['accuracy']) #nadam mean_absolute_error
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epoch,validation_data=(X_test, Y_test))


#Predictions

###Ackermann###
if rows == 60:
    X_predict_str = numpy.genfromtxt("input/input_predict_ack_60.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input/input_predict_ack_30.txt",dtype=str)
elif rows == 10:
    X_predict_str = numpy.genfromtxt("input/input_predict_ack_10.txt",dtype=str)
elif rows == 5:
    X_predict_str = numpy.genfromtxt("input/input_predict_ack_5.txt",dtype=str)
else:
    print("Invalid row number")
X_predict = numpy.zeros(X_predict_str.size)
for x in range (0, X_predict_str.size-1):
    X_predict[x] = getInstructionIdentifier(X_train_str,X_train, X_predict_str[x])
predictions = model.predict(X_predict)
print("Ackermann")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
error_a15 = (numpy.average(column(predictions,0))-p_ack_a15)/numpy.average(column(predictions,0))
error_a7 = (numpy.average(column(predictions,1))-p_ack_a7)/numpy.average(column(predictions,1))
error_perf = (numpy.average(column(predictions,2))-perf_ack)/numpy.average(column(predictions,2))
total_error += abs(error_a15) + abs(error_a7) + abs(error_perf)
print(round(error_a15,2),round(error_a7,2),round(error_perf,2))
print()

###Int32Double###
if rows == 60:
    X_predict_str = numpy.genfromtxt("input/input_predict_i32d_60.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input/input_predict_i32d_30.txt",dtype=str)
elif rows == 10:
    X_predict_str = numpy.genfromtxt("input/input_predict_i32d_10.txt",dtype=str)
elif rows == 5:
    X_predict_str = numpy.genfromtxt("input/input_predict_i32d_5.txt",dtype=str)
else:
    print("Invalid row number")
X_predict = numpy.zeros(X_predict_str.size)
for x in range (0, X_predict_str.size-1):
    X_predict[x] = getInstructionIdentifier(X_train_str,X_train, X_predict_str[x])
predictions = model.predict(X_predict)
print("Int32Double")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
error_a15 = (numpy.average(column(predictions,0))-p_i32d_a15)/numpy.average(column(predictions,0))
error_a7 = (numpy.average(column(predictions,1))-p_i32d_a7)/numpy.average(column(predictions,1))
error_perf = (numpy.average(column(predictions,2))-perf_i32d)/numpy.average(column(predictions,2))
total_error += abs(error_a15) + abs(error_a7) + abs(error_perf)
print(round(error_a15,2),round(error_a7,2),round(error_perf,2))
print()

###ln2###
if rows == 60:
    X_predict_str = numpy.genfromtxt("input/input_predict_ln2_60.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input/input_predict_ln2_30.txt",dtype=str)
elif rows == 10:
    X_predict_str = numpy.genfromtxt("input/input_predict_ln2_10.txt",dtype=str)
elif rows == 5:
    X_predict_str = numpy.genfromtxt("input/input_predict_ln2_5.txt",dtype=str)
else:
    print("Invalid row number")
X_predict = numpy.zeros(X_predict_str.size)
for x in range (0, X_predict_str.size-1):
    X_predict[x] = getInstructionIdentifier(X_train_str,X_train, X_predict_str[x])
predictions = model.predict(X_predict)
print("ln2")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
error_a15 = (numpy.average(column(predictions,0))-p_ln2_a15)/numpy.average(column(predictions,0))
error_a7 = (numpy.average(column(predictions,1))-p_ln2_a7)/numpy.average(column(predictions,1))
error_perf = (numpy.average(column(predictions,2))-perf_ln2)/numpy.average(column(predictions,2))
total_error += abs(error_a15) + abs(error_a7) + abs(error_perf)
print(round(error_a15,2),round(error_a7,2),round(error_perf,2))
print()

###Cdouble###
if rows == 60:
    X_predict_str = numpy.genfromtxt("input/input_predict_cdouble_60.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input/input_predict_cdouble_30.txt",dtype=str)
elif rows == 10:
    X_predict_str = numpy.genfromtxt("input/input_predict_cdouble_10.txt",dtype=str)
elif rows == 5:
    X_predict_str = numpy.genfromtxt("input/input_predict_cdouble_5.txt",dtype=str)
else:
    print("Invalid row number")
X_predict = numpy.zeros(X_predict_str.size)
for x in range (0, X_predict_str.size-1):
    X_predict[x] = getInstructionIdentifier(X_train_str,X_train, X_predict_str[x])
predictions = model.predict(X_predict)
print("Cdouble")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
error_a15 = (numpy.average(column(predictions,0))-p_cdb_a15)/numpy.average(column(predictions,0))
error_a7 = (numpy.average(column(predictions,1))-p_cdb_a7)/numpy.average(column(predictions,1))
error_perf = (numpy.average(column(predictions,2))-perf_cdb)/numpy.average(column(predictions,2))
total_error += abs(error_a15) + abs(error_a7) + abs(error_perf)
print(round(error_a15,2),round(error_a7,2),round(error_perf,2))
print()

###Callfunc###
if rows == 60:
    X_predict_str = numpy.genfromtxt("input/input_predict_callfunc_60.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input/input_predict_callfunc_30.txt",dtype=str)
elif rows == 10:
    X_predict_str = numpy.genfromtxt("input/input_predict_callfunc_10.txt",dtype=str)
elif rows == 5:
    X_predict_str = numpy.genfromtxt("input/input_predict_callfunc_5.txt",dtype=str)
else:
    print("Invalid row number")
X_predict = numpy.zeros(X_predict_str.size)
for x in range (0, X_predict_str.size-1):
    X_predict[x] = getInstructionIdentifier(X_train_str,X_train, X_predict_str[x])
predictions = model.predict(X_predict)
print("Callfunc")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
error_a15 = (numpy.average(column(predictions,0))-p_call_a15)/numpy.average(column(predictions,0))
error_a7 = (numpy.average(column(predictions,1))-p_call_a7)/numpy.average(column(predictions,1))
error_perf = (numpy.average(column(predictions,2))-perf_call)/numpy.average(column(predictions,2))
total_error += abs(error_a15) + abs(error_a7) + abs(error_perf)
print(round(error_a15,2),round(error_a7,2),round(error_perf,2))
print()

###Dither###
if rows == 60:
    X_predict_str = numpy.genfromtxt("input/input_predict_dither_60.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input/input_predict_dither_30.txt",dtype=str)
elif rows == 10:
    X_predict_str = numpy.genfromtxt("input/input_predict_dither_10.txt",dtype=str)
elif rows == 5:
    X_predict_str = numpy.genfromtxt("input/input_predict_dither_5.txt",dtype=str)
else:
    print("Invalid row number")
X_predict = numpy.zeros(X_predict_str.size)
for x in range (0, X_predict_str.size-1):
    X_predict[x] = getInstructionIdentifier(X_train_str,X_train, X_predict_str[x])
predictions = model.predict(X_predict)
print("Dither")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
error_a15 = (numpy.average(column(predictions,0))-p_dit_a15)/numpy.average(column(predictions,0))
error_a7 = (numpy.average(column(predictions,1))-p_dit_a7)/numpy.average(column(predictions,1))
error_perf = (numpy.average(column(predictions,2))-perf_dit)/numpy.average(column(predictions,2))
total_error += abs(error_a15) + abs(error_a7) + abs(error_perf)
print(round(error_a15,2),round(error_a7,2),round(error_perf,2))
print()

###Euler###
if rows == 60:
    X_predict_str = numpy.genfromtxt("input/input_predict_euler_60.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input/input_predict_euler_30.txt",dtype=str)
elif rows == 10:
    X_predict_str = numpy.genfromtxt("input/input_predict_euler_10.txt",dtype=str)
elif rows == 5:
    X_predict_str = numpy.genfromtxt("input/input_predict_euler_5.txt",dtype=str)
else:
    print("Invalid row number")
X_predict = numpy.zeros(X_predict_str.size)
for x in range (0, X_predict_str.size-1):
    X_predict[x] = getInstructionIdentifier(X_train_str,X_train, X_predict_str[x])
predictions = model.predict(X_predict)
print("Euler")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
error_a15 = (numpy.average(column(predictions,0))-p_eul_a15)/numpy.average(column(predictions,0))
error_a7 = (numpy.average(column(predictions,1))-p_eul_a7)/numpy.average(column(predictions,1))
error_perf = (numpy.average(column(predictions,2))-perf_eul)/numpy.average(column(predictions,2))
total_error += abs(error_a15) + abs(error_a7) + abs(error_perf)
print(round(error_a15,2),round(error_a7,2),round(error_perf,2))
print()

###Fibonacci###
if rows == 60:
    X_predict_str = numpy.genfromtxt("input/input_predict_fibonacci_60.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input/input_predict_fibonacci_30.txt",dtype=str)
elif rows == 10:
    X_predict_str = numpy.genfromtxt("input/input_predict_fibonacci_10.txt",dtype=str)
elif rows == 5:
    X_predict_str = numpy.genfromtxt("input/input_predict_fibonacci_5.txt",dtype=str)
else:
    print("Invalid row number")
X_predict = numpy.zeros(X_predict_str.size)
for x in range (0, X_predict_str.size-1):
    X_predict[x] = getInstructionIdentifier(X_train_str,X_train, X_predict_str[x])
predictions = model.predict(X_predict)
print("Fibonacci")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
error_a15 = (numpy.average(column(predictions,0))-p_fib_a15)/numpy.average(column(predictions,0))
error_a7 = (numpy.average(column(predictions,1))-p_fib_a7)/numpy.average(column(predictions,1))
error_perf = (numpy.average(column(predictions,2))-perf_fib)/numpy.average(column(predictions,2))
total_error += abs(error_a15) + abs(error_a7) + abs(error_perf)
print(round(error_a15,2),round(error_a7,2),round(error_perf,2))
print()

###Gamma###
if rows == 60:
    X_predict_str = numpy.genfromtxt("input/input_predict_gamma_60.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input/input_predict_gamma_30.txt",dtype=str)
elif rows == 10:
    X_predict_str = numpy.genfromtxt("input/input_predict_gamma_10.txt",dtype=str)
elif rows == 5:
    X_predict_str = numpy.genfromtxt("input/input_predict_gamma_5.txt",dtype=str)
else:
    print("Invalid row number")
X_predict = numpy.zeros(X_predict_str.size)
for x in range (0, X_predict_str.size-1):
    X_predict[x] = getInstructionIdentifier(X_train_str,X_train, X_predict_str[x])
predictions = model.predict(X_predict)
print("Gamma")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
error_a15 = (numpy.average(column(predictions,0))-p_gam_a15)/numpy.average(column(predictions,0))
error_a7 = (numpy.average(column(predictions,1))-p_gam_a7)/numpy.average(column(predictions,1))
error_perf = (numpy.average(column(predictions,2))-perf_gam)/numpy.average(column(predictions,2))
total_error += abs(error_a15) + abs(error_a7) + abs(error_perf)
print(round(error_a15,2),round(error_a7,2),round(error_perf,2))
print()

###Int32Float###
if rows == 60:
    X_predict_str = numpy.genfromtxt("input/input_predict_i32f_60.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input/input_predict_i32f_30.txt",dtype=str)
elif rows == 10:
    X_predict_str = numpy.genfromtxt("input/input_predict_i32f_10.txt",dtype=str)
elif rows == 5:
    X_predict_str = numpy.genfromtxt("input/input_predict_i32f_5.txt",dtype=str)
else:
    print("Invalid row number")
X_predict = numpy.zeros(X_predict_str.size)
for x in range (0, X_predict_str.size-1):
    X_predict[x] = getInstructionIdentifier(X_train_str,X_train, X_predict_str[x])
predictions = model.predict(X_predict)
print("I32f")
print(numpy.average(column(predictions,0)),numpy.average(column(predictions,1)),numpy.average(column(predictions,2)))
error_a15 = (numpy.average(column(predictions,0))-p_i32f_a15)/numpy.average(column(predictions,0))
error_a7 = (numpy.average(column(predictions,1))-p_i32f_a7)/numpy.average(column(predictions,1))
error_perf = (numpy.average(column(predictions,2))-perf_i32f)/numpy.average(column(predictions,2))
total_error += abs(error_a15) + abs(error_a7) + abs(error_perf)
print(round(error_a15,2),round(error_a7,2),round(error_perf,2))
print()

print("Total error ",round(total_error*100,2), "%")
