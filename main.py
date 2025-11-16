# This is a sample Python script.

import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

class WeightsVector(object):

    def __init__(self, val, etha = 0.1):
        self.val = val
        self.etha = etha

    def update_weights(self, update: dict):

        #update is expected in form [{'tag': Bool, 'example': list}]

        self.val = self.val + (2*update['tag'] - 1) * update['example']

x = WeightsVector(np.array([1,2,3]))
print(x.val)

class ExamplesMatrix(object):

    #create data

    def __init__(self, p, d, tags = pd.DataFrame(None), etha = 0.1):
        self.val = pd.DataFrame(data = [[random.uniform(-10,10) for dim in range(d)] for ex in range(p)])
        self.tags = tags

    def generate_random_tags(self):
        self.tags = pd.DataFrame([random.choice([True, False]) for ex in self.val.iterrows()])

    #analyse

    def get_classification(self,weights_vec):
        output = np.heaviside((self.val @ weights_vec), 0)
        return output

    def check_weights(self, weights_vec: WeightsVector):
        output = np.ones(len(self.val)) - np.abs(self.tags - self.get_classification(weights_vec.val))
        return output

    def get_first_mistake_index(self, weights_vec: WeightsVector):
        first_mistake = -1
        for i, val in self.check_weights(weights_vec.val).enumerate():
            if val != 1:
                first_mistake = i
                break
        if first_mistake == -1:
            return 'No mistakes'
        return first_mistake

    def get_first_mistake(self, weights_vec):
        first_mistake_index = self.get_first_mistake_index(weights_vec)
        first_mistake = self.val.iloc[first_mistake_index]
        first_mistake_tag = self.tags.iloc[first_mistake_index]

        output = {'tag': first_mistake_tag,
                  'example': first_mistake}
        return output

    #plot

    def plot_classification(self, classification_vec: np.array):
        true_dots = self.val[classification_vec == 1]
        false_dots = self.val[classification_vec == 0]
        #     [self.val.iloc[ex] for ex in range(len(self.val)) if classification_vec[ex] == 1]
        # false_dots = [self.val.iloc[ex] for ex in range(len(self.val)) if classification_vec[ex] == 0]

        plt.scatter(true_dots.iloc[:,0], true_dots.iloc[:,1], color = 'blue', label = 'True')
        plt.scatter(false_dots.iloc[:,0], false_dots.iloc[:,1], color = 'red', label = 'False')
        plt.show()

def iteration(examples_matrix, weights_vec: WeightsVector):
    first_mistake_dict = examples_matrix.get_first_mistake(weights_vec)
    weights_vec.update_weights(first_mistake_dict)


x = ExamplesMatrix(15,2)
print(x.val)
x.plot_classification(np.array([1 for i in range(15)]))
x.generate_random_tags()
c = x.check_weights([1,1])
print(c)



def get_error(examples_matrix, tags_vector, weights_vector):
    None

def learning_iteration(examples_matrix, tags_vector, weights_vector):
    return WeightsVector