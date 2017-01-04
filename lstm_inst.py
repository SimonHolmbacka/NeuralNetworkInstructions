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

total_error = 0

def getInstructionIdentifier(array_str, array_nr, str):
    for x in range (0, array_str.size-1):
        if array_str[x] == str:
            return array_nr[x]
    print("Error could not find instruction")


def column(matrix, i):
    return [row[i] for row in matrix]


#We make all strings to an identification integer
X_train_str = numpy.genfromtxt("input_train.csv",dtype=str)
X_train = categorical(X_train_str, drop=True)
X_train = X_train.argmax(1)

#Target power and performance values
Y_train = numpy.loadtxt("input_target.csv", delimiter=",")

#Test
X_test_str = numpy.genfromtxt("input_test.csv", dtype=str)
X_test = numpy.zeros(X_test_str.size)
for x in range (0, X_test_str.size-1):
    X_test[x] = getInstructionIdentifier(X_train_str,X_train, X_test_str[x])
#Float
Y_test = numpy.loadtxt("input_testtarget.csv", delimiter=",")


max_features = 18000
batch_size = 4 #4 is good and 8 makes it a little worse
member_berries = 128 #should be more member berries if input sequence is longer (512 seems to be the max)
rows = 60

#create the model
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(Convolution1D(nb_filter=2, filter_length=2, border_mode='same', activation='linear'))
model.add(LSTM(member_berries))
model.add(Dropout(0.1)) #Applies Dropout to the input. Dropout consists in randomly setting a fraction p of input units to 0 at each update during training time, which helps prevent overfitting.
model.add(Dense(output_dim=3, init='glorot_normal', activation='softplus'))

model.compile(loss='mean_absolute_error',optimizer='nadam',metrics=['accuracy']) #nadam mean_absolute_error
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=2,validation_data=(X_test, Y_test))


#Predictions
if rows == 60:
    X_predict_str = numpy.genfromtxt("input_predict_ack_60.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input_predict_ack_30.txt",dtype=str)
elif rows == 10:
    X_predict_str = numpy.genfromtxt("input_predict_ack_10.txt",dtype=str)
elif rows == 5:
    X_predict_str = numpy.genfromtxt("input_predict_ack_5.txt",dtype=str)
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

if rows == 60:
    X_predict_str = numpy.genfromtxt("input_predict_i32d_60.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input_predict_i32d_30.txt",dtype=str)
elif rows == 10:
    X_predict_str = numpy.genfromtxt("input_predict_i32d_10.txt",dtype=str)
elif rows == 5:
    X_predict_str = numpy.genfromtxt("input_predict_i32d_5.txt",dtype=str)
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

if rows == 60:
    X_predict_str = numpy.genfromtxt("input_predict_ln2_60.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input_predict_ln2_30.txt",dtype=str)
elif rows == 10:
    X_predict_str = numpy.genfromtxt("input_predict_ln2_10.txt",dtype=str)
elif rows == 5:
    X_predict_str = numpy.genfromtxt("input_predict_ln2_5.txt",dtype=str)
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

if rows == 60:
    X_predict_str = numpy.genfromtxt("input_predict_cdouble_60.txt",dtype=str)
elif rows == 30:
    X_predict_str = numpy.genfromtxt("input_predict_cdouble_30.txt",dtype=str)
elif rows == 10:
    X_predict_str = numpy.genfromtxt("input_predict_cdouble_10.txt",dtype=str)
elif rows == 5:
    X_predict_str = numpy.genfromtxt("input_predict_cdouble_5.txt",dtype=str)
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

print("Total error ",round(total_error*100,2), "%")

