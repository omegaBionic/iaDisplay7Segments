import csv

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD

# Open and parse csv
input_array = []
teacher_array = []
with open("datas.csv", newline='') as csv_file:
    spam_reader = csv.reader(csv_file, delimiter=',')
    next(spam_reader)
    for row in spam_reader:
        arr_input = np.zeros((1, 7))
        arr_teacher = np.zeros((1, 9))

        # print("arr_input: '{}'".format(row[0:6]))
        # print("arr_teacher: '{}'".format(row[7:15]))
        # print("----")
        arr_input = list(map(int, row[0:7]))
        arr_teacher = list(map(int, row[7:16]))
        input_array.append(arr_input)
        teacher_array.append(arr_teacher)

input_np_array = np.array(input_array)
teacher_np_array = np.array(teacher_array)

input_np_array = np.reshape(input_np_array, (len(input_np_array), 1, 7))
teacher_np_array = np.reshape(teacher_np_array, (len(teacher_np_array), 1, 9))

print("input_np_array: '{}'\n".format(input_np_array))
print("teacher_np_array: '{}'\n".format(teacher_np_array))

# Train perceptron
# learning rate
sgd = SGD(lr=0.01)

model = Sequential()
model.add(Dense(8, input_dim=7, activation='sigmoid'))

#hiden layers
model.add(Dense(9, activation='sigmoid'))

#output layer
model.add(Dense(9, activation='sigmoid'))

model.summary()
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(input_np_array, teacher_np_array, epochs=10000)
