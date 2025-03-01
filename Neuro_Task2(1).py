#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

inputs = np.array([0.05, 0.10])
targets = np.array([0.01, 0.99])

weights_input_hidden = np.array([[0.15, 0.20], [0.25, 0.30]])  
weights_hidden_output = np.array([[0.40, 0.45], [0.50, 0.55]])  

bias_hidden = np.array([0.35, 0.35])  
bias_output = np.array([0.60, 0.60])  

lr = 0.5

hidden_inputs = np.dot(weights_input_hidden, inputs) + bias_hidden
hidden_outputs = sigmoid(hidden_inputs)

final_inputs = np.dot(weights_hidden_output, hidden_outputs) + bias_output
final_outputs = sigmoid(final_inputs)

errors = 0.5 * (targets - final_outputs) ** 2

output_errors = (final_outputs - targets) * sigmoid_derivative(final_outputs)
hidden_errors = np.dot(weights_hidden_output.T, output_errors) * sigmoid_derivative(hidden_outputs)

weights_hidden_output -= lr * np.outer(output_errors, hidden_outputs)
weights_input_hidden -= lr * np.outer(hidden_errors, inputs)

bias_output -= lr * output_errors
bias_hidden -= lr * hidden_errors

print("Updated Weights (Input to Hidden):\n", weights_input_hidden)
print("Updated Weights (Hidden to Output):\n", weights_hidden_output)
print("Updated Biases (Hidden Layer):\n", bias_hidden)
print("Updated Biases (Output Layer):\n", bias_output)


# In[ ]:




