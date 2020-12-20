import csv

import numpy as np
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Open and parse csv for learning
input_array = []
teacher_array = []
with open("datas_learn.csv", newline='') as csv_file:
    spam_reader = csv.reader(csv_file, delimiter=',')
    next(spam_reader)
    for row in spam_reader:
        #arr_input = np.zeros((1, 7))
        arr_teacher = 0

        arr_input = list(map(int, row[0:7]))
        arr_teacher = int(row[7])
        input_array.append(arr_input)
        teacher_array.append(arr_teacher)

input_learn_np_array = np.array(input_array)
teacher_learn_np_array = np.array(teacher_array)

input_learn_np_array = np.reshape(input_learn_np_array, (len(input_learn_np_array), 1, 7))
teacher_learn_np_array = to_categorical(teacher_learn_np_array, dtype='int32')
teacher_learn_np_array = np.reshape(teacher_learn_np_array, (len(teacher_learn_np_array), 1, 10))
#teacher_learn_np_array = np.delete(teacher_learn_np_array, [0, ])

print("input_learn_np_array: '{}'\n".format(input_learn_np_array))
print("teacher_learn_np_array: '{}'\n".format(teacher_learn_np_array))

#Open and parse csv for testing
input_array = []
teacher_array = []
with open("datas_test.csv", newline='') as csv_file:
    spam_reader = csv.reader(csv_file, delimiter=',')
    next(spam_reader)
    for row in spam_reader:
        arr_input = np.zeros((1, 7))
        arr_teacher = 0

        # print("arr_input: '{}'".format(row[0:6]))
        # print("arr_teacher: '{}'".format(row[7:15]))
        # print("----")
        arr_input = list(map(int, row[0:7]))
        arr_teacher = row[7]
        input_array.append(arr_input)
        teacher_array.append(arr_teacher)

input_test_np_array = np.array(input_array)
teacher_test_np_array = np.array(teacher_array)

input_test_np_array = np.reshape(input_test_np_array, (len(input_test_np_array), 1, 7))
teacher_test_np_array = to_categorical(teacher_test_np_array, dtype='int32')
teacher_test_np_array = np.reshape(teacher_test_np_array, (len(teacher_test_np_array), 1, 10))

print("input_test_np_array: '{}'\n".format(input_test_np_array))
print("teacher_test_np_array: '{}'\n".format(teacher_test_np_array))

# Train perceptron
# learning rate
sgd = SGD(lr=0.05)

model = Sequential()
model.add(Dense(8, input_dim=7, activation='relu', input_shape=(1, 7)))

# second hiden layer
#model.add(Dense(9, activation='sigmoid'))

# output layer
model.add(Dense(10, activation='softmax'))

model.summary()
#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_history = model.fit(input_learn_np_array, teacher_learn_np_array, epochs=5000,
                validation_data=(input_test_np_array, teacher_test_np_array))

#   Voyons voir si notre réseau de neurones a bien appris !
value_to_guess = np.array([[[0, 1, 1, 0, 0, 1, 0]], [[0, 1, 0, 0, 0, 0, 0]]])
predictions = model.predict(value_to_guess)
print(f'Predictions : {predictions}')
for v in predictions:
    #print(f'Rounded value : {round(v[0])}')
    result = np.where(v[0] == np.amax(v[0]))
    print('List of Indices of maximum element :', result[0])
    print('Returned tuple of arrays :', result)
    print(f'Not rounded value : {np.amax(v[0])}')

model_history.history['accuracy']
#tracer les courbes d'apprentissage
# summarize history for accuracy
f = plt.figure(1)
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.grid()

# summarize history for loss
g = plt.figure(2)
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.grid()
plt.show()
