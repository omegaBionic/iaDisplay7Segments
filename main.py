import csv

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical

# Open and parse csv for learning
input_array = []
teacher_array = []
with open("datas_learn.csv", newline='') as csv_file:
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

input_learn_np_array = np.array(input_array)
teacher_learn_np_array = np.array(teacher_array)

#teacher_array_output = to_categorical(teacher_array)

input_learn_np_array = np.reshape(input_learn_np_array, (len(input_learn_np_array), 1, 7))
teacher_learn_np_array = np.reshape(teacher_learn_np_array, (len(teacher_learn_np_array), 1, 9))

print("input_learn_np_array: '{}'\n".format(input_learn_np_array))
print("teacher_learn_np_array: '{}'\n".format(teacher_learn_np_array))


# Open and parse csv for testing
input_array = []
teacher_array = []
with open("datas_test.csv", newline='') as csv_file:
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

input_test_np_array = np.array(input_array)
teacher_test_np_array = np.array(teacher_array)

input_test_np_array = np.reshape(input_test_np_array, (len(input_test_np_array), 1, 7))
teacher_test_np_array = np.reshape(teacher_test_np_array, (len(teacher_test_np_array), 1, 9))

print("input_test_np_array: '{}'\n".format(input_test_np_array))
print("teacher_test_np_array: '{}'\n".format(teacher_test_np_array))


# Train perceptron
# learning rate
sgd = SGD(lr=0.05)

model = Sequential()
model.add(Dense(8, input_dim=7, activation='relu'))

# hiden layers
model.add(Dense(9, activation='sigmoid'))

# output layer
model.add(Dense(9, activation='softmax'))

model.summary()
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(input_learn_np_array, teacher_learn_np_array, epochs=10000, validation_data=(input_test_np_array, teacher_test_np_array))
# TODO: callback checkpoint
# TODO: evaluate
