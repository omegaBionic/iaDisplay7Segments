import csv

import numpy as np

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
        arr_input = row[0:7]
        arr_teacher = row[8:16]
        input_array.append(arr_input)
        teacher_array.append(arr_teacher)

input_np_array = np.array(input_array)
teacher_np_array = np.array(teacher_array)

input_np_array = np.reshape(input_np_array, (len(input_np_array), 1, 7))
print("input_np_array: '{}'\n".format(input_np_array))
print("teacher_np_array: '{}'\n".format(teacher_np_array))
