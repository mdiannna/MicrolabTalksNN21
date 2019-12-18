#!/usr/bin/env python
# coding: utf-8

# # Sum numbers with Recurrent Neural Networks

# TODO: intro about task

# TODO: intro about Recurrent Neural Networks

# TODO: 
# - import tensorflow - should work!!! + google colab cu acces doar de rulare? sau doar de citire/copiere
# - Bibliografie
# - link catre prezentare??????
# - articol? nu cred, pt ca e o idee "imprumutata"

# In[1]:


# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow import keras
import tensorflow.python.keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from sklearn.metrics import mean_squared_error

# # In[9]:
from numpy import argmax

from random import seed
from random import randint


def generate_random_number(max_digits=3):
    return randint(0, 10**max_digits)

x1 = generate_random_number(3)
x2 = generate_random_number()
print(x1, '+',x2, '=', x1+x2)

def generate_random_number_pair(max_digits=3):
  a = []
  a.append(generate_random_number(max_digits))
  a.append(generate_random_number(max_digits))
  return a

def generate_random_number_dataset(size, max_digits=3):
  result = []
  sum_result = []

  for i in range(0, size):
    numbers = generate_random_number_pair(max_digits)
    result.append(numbers)
    sum_result.append(numbers[0]+numbers[1])

  return result, sum_result


dataset_size = 5000

numbers_dataset, sum_dataset = generate_random_number_dataset(dataset_size)
# print(numbers_dataset)
# print(sum_dataset)

# # Note: Daca ambele numere au maxim 3 cifre, suma va avea maxim 4 cifre


# # ## Encoding

# - First of all, individual digits will be transformed to characters for each number.
# 
# *eg. a=123, b=4 will be transformed in '['1', '2', '3'] and ['4']*
# 
# - After that, will use padding to obtain arrays of equal size (3 characters); The value of 10 is used for padding, because it will be the 10th character in the alphabet.
# 
# *a = ['1','2','3'], b=['10','10','4']*
# 
# - After that, digit characters for both numbers will be concatenated.
# 
# *['1','2','3',.'10','10','4']*
# 
# - And last, will use one hot encoding for encoding.
# 
# *[0 1 0 0 0 0 0 0 0 0 0]*
# 
# *[0 0 1 0 0 0 0 0 0 0 0]*
# 
# *[0 0 0 1 0 0 0 0 0 0 0]*
# 
# *[0 0 0 0 0 0 0 0 0 0 1]*
# 
# *[0 0 0 0 0 0 0 0 0 0 1]*
# 
# *[0 0 0 0 1 0 0 0 0 0 0]*
# 
# 

# In[ ]:


# from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


def separate_digits(number):
    return [int(i) for i in str(number)]


def encode_numbers(numbers_dataset, dataset_size, max_lenght = 3, is_train = True):
    X = np.zeros((dataset_size, (max_lenght)*2, 11))
    Y = np.zeros((dataset_size, max_lenght, 11))

    for i in range(0, dataset_size):
      number1 = numbers_dataset[i][0] 
      number2 = numbers_dataset[i][1] 

      a = separate_digits(number1)
      b = separate_digits(number2)
          
      # XX = np.zeros(((max_lenght)*2, 11))
      # YY = np.zeros((max_lenght, 11))
    
      s = number1 + number2
      s = separate_digits(s)
      
      if is_train:
        a,b,s = pad_sequences([a,b,s],maxlen = max_lenght, value=10)
        x , y = np.concatenate([a,b]), s
      else:
        a,b = pad_sequences([a,b],maxlen = max_lenght, value=10)
        x = np.concatenate([a,b])

      print('x:', x)
      print('x:', enumerate(x))
      for j, char in enumerate(x):
        X[i, j, char] = 1
       
      if(is_train):
        for j, char in enumerate(s):
          Y[i, j, char] = 1

    return X, Y
            
  
# print(x)
# print(y)

X,Y = encode_numbers(numbers_dataset, dataset_size, is_train=True)

# X = []
# Y = []
# max_lenght = 3


# for i in range(0, dataset_size):
#   print(numbers_dataset[i][0], numbers_dataset[i][1])
#   x, y = encode_numbers(numbers_dataset[i][0], numbers_dataset[i][1], is_train=True)
#   X[i] = x
#   Y[i] = y

print('dataset size:', dataset_size)
print("#########")
print(X[0:10])

print(X.size)
print(X.shape)
# print(Y[0:4])


# # In[ ]:


# def encode_numbers(numbers_dataset, is_train=False, max_lenght = 3):
#   n = len(numbers_dataset)
# #   print(numbers_dataset[0][0])
#   X = np.zeros((n, (max_lenght)*2, 11))
#   Y = np.zeros((n, max_lenght, 11))
  
#   for i in range(n):
#     a = numbers_dataset[i][0]
#     b = numbers_dataset[i][1]
    
#     if(is_train):
#       s = a + b
#       s = [int(i) for i in str(s)]

#     a = [int(i) for i in str(a)]
#     b = [int(i) for i in str(b)]
  
#     print(a,b)
    
#     if is_train:
#       a,b,s = pad_sequences([a,b,s],maxlen = max_lenght, value=10)
#       x , y = np.concatenate([a,b]), s
#     else:
#       a,b = pad_sequences([a,b],maxlen = max_lenght, value=10)
#       x = np.concatenate([a,b])

#     for j, char in enumerate(x):
#          X[i, j, char] = 1
     
#     if(is_train):
#       for j, char in enumerate(s):
#          Y[i, j, char] = 1
          
#   if(is_train):
#       return X, Y
#   return X

# X , Y = encode_numbers(numbers, True)

# # print(numbers[0:10])
# # print(X[0:10])
# # print(Y[0:10])


# # ### Step2 - Configure network and train model

# # In[ ]:


alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ']
chars_len = len(alphabet)

input_len = 6
output_len = 3

model = Sequential()
model.add(LSTM(100, input_shape=(input_len, chars_len)))
model.add(RepeatVector(output_len))
model.add(LSTM(50, return_sequences=True))
model.add(TimeDistributed(Dense(chars_len, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# # In[ ]:
max_lenght = 3
# X = X.resize(dataset_size, (max_lenght)*2, 11)
model.fit(X, Y, epochs=20)
# model.fit(X, Y, epochs=100)



# # ## Decode

# # The neural network outputs the results being one hot encoded, this is why a function for decoding is needed. This function takes the maximum argument for each row and determines the individual digit, then we use the integer_combine function to combine digits and obtain the number.

# # In[3]:


def decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]

def integer_combine(n_array):
  result = 0
  for num in n_array:
    if(num!=10):
      result = result * 10 + round(num)
  return result

y = Y.astype(int)

print(y.shape)
result1 = [integer_combine(decode(x)) for x in y]
print(result1)


# # ## Test on new data and evaluate performance

# # In[ ]:


# # NR_SAMPLES_TEST = 3000
# NR_SAMPLES_TEST = 100

# # TODO: generate_numbers
# generator = test_generator(batch_size=NR_SAMPLES_TEST)

# x_test, numbers_test, numbers_sum_test = next(generator)


# # In[ ]:


# X_test, Y_test = encode_numbers(numbers_test, True)

dataset_size_test = 100
numbers_dataset_test, sum_dataset_test = generate_random_number_dataset(dataset_size)

X_test,Y_test = encode_numbers(numbers_dataset_test, dataset_size_test)


# # In[ ]:


result = model.predict(X_test)

# predicted = [invert(x, alphabet) for x in result]
predicted = [integer_combine(decode(x)) for x in result]
expected = [integer_combine(decode(x)) for x in Y_test]

loss, acc = model.evaluate(X_test, Y_test)

print("Loss on test set: %f" % loss)
print("Accuracy on test set: %f" % acc)
# print("Mean squared error:")
mse = mean_squared_error(expected, predicted)
print('Mean squared error: %f' % mse)

# rmse = sqrt(mean_squared_error(expected, predicted))
# print('RMSE: %f' % rmse)

print("--First 20 examples result:--")
for i in range(20):
	print('Expected=%s, Predicted=%s' % (expected[i], predicted[i]))
	print('Difference=%s' % (abs(expected[i] - predicted[i])))
  
plt.plot(expected, predicted, 'oc')
plt.xlabel("Expected")
plt.ylabel("Predicted")

plt.show()


# # In[4]:



# MODEL_NAME = "SUM_NUMBERS1.h5"
# model.save(MODEL_NAME)
# # for loading model:
# # model = load_model('sum_digits_RNN.h5')


# # ## Conclusions:TODO

# # We created a model for adding 2 numbers using a neural network. The strategy used for this task was to build an encoder-decoder model architecture (also named sequence to sequence) and predict the sum of the integers based on encoded sequences of initial numbers. The neural network consisted of 2 LSTM layers (the first with 100 cells and the second with 50) and a Dense layer. The accuracy obtained on the training set was 99.28% and on the test set was 99.04%
