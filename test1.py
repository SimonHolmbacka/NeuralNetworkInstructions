from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("CDOUB.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:6]
Y = dataset[:,6:9]

print(X)

# create model
model = Sequential()

model.add(Dense(20, input_dim=6, init='glorot_uniform', activation='relu'))
#model.add(Dense(6, init='uniform', activation='softplus'))
model.add(Dense(output_dim=3, init='glorot_uniform', activation='softplus'))


# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, nb_epoch=200, batch_size=10,  verbose=1)
# calculate predictions
predictions = model.predict(X)
print(predictions)
# round predictions
#rounded = [round(x) for x in predictions]
#print(rounded)

dataset2 = numpy.loadtxt("CDOUBin.csv", delimiter=",")
# split into input (X) and output (Y) variables
newX = dataset2[:,0:6]
#newPred = model.predict(dataset2)
newPred = model.predict(newX)
print()
print(newPred)
#newRounded = [round(x) for x in newPred]
#print(newRounded)
