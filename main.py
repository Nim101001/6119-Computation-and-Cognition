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

        self.val = self.val + self.etha*(2*update['tag'] - 1) * update['example']

    def get_dividing_plane_func(self):
        val = self.val
        slope = -(val[0]/val[1]) #accoridng to (a,b)*(c,d)=0->c=-(b/a)*d for d =1 c=-(b/a) so slope = d/c=-(a/b)
        def func(x):
            return x*slope
        return func

class ExamplesMatrix(object):

    #create data

    def __init__(self, p, d, tags = pd.DataFrame(None)):
        self.val = pd.DataFrame(data = [[random.uniform(-10,10) for dim in range(d)] for ex in range(p)])
        self.tags = tags

    def generate_random_tags(self):
        self.tags = pd.Series([random.choice([True, False]) for ex in range(len(self.val))])

    #analyse

    def get_classification(self, weights_vec: WeightsVector):
        output = np.heaviside((self.val @ weights_vec.val), 0)
        return output

    def check_weights(self, weights_vec: WeightsVector) -> pd.Series:
        #output[i]=1 if example is rightly classified by weights_vec,output[i]=0 if not.
        output = pd.Series(np.ones(len(self.val)) - np.abs(self.tags.to_numpy() - np.array(self.get_classification(weights_vec))))
        return output

    def get_first_mistake_index(self, weights_vec: WeightsVector):
        first_mistake = -1
        for i, val in enumerate(self.check_weights(weights_vec)):
            if val != 1:
                first_mistake = i
                break
        if first_mistake == -1:
            return 'No mistakes'
        return first_mistake

    def get_first_mistake(self, weights_vec: WeightsVector):
        first_mistake_index = self.get_first_mistake_index(weights_vec)
        first_mistake = self.val.iloc[first_mistake_index]
        first_mistake_tag = self.tags.iloc[first_mistake_index]

        output = {'tag': first_mistake_tag,
                  'example': first_mistake}
        return output

    #plot

    def plot_classification(self, weights_vec: WeightsVector):

        #assumes d = 2

        # classification_vec = self.check_weights(weights_vec)
        classification_vec = self.get_classification(weights_vec)
        weights_success_vec = self.check_weights(weights_vec)

        true_dots = self.val[classification_vec == 1]
        false_dots = self.val[classification_vec == 0]
        success_dots = self.val[weights_success_vec == 1]
        failure_dots = self.val[weights_success_vec == 0]

        plt.scatter(success_dots.iloc[:, 0], success_dots.iloc[:, 1], s = 40, color='green', label='Successful Classification')
        plt.scatter(failure_dots.iloc[:, 0], failure_dots.iloc[:, 1], s = 40, color='brown', label='Failed Classification')
        plt.scatter(true_dots.iloc[:,0], true_dots.iloc[:,1], s = 20, color = 'blue', label = 'Classified as True')
        plt.scatter(false_dots.iloc[:,0], false_dots.iloc[:,1], s = 20, color = 'red', label = 'Classified as False')

        dividing_plane_func = weights_vec.get_dividing_plane_func()
        plt.plot([-10,10], [dividing_plane_func(-10), dividing_plane_func(10)], label = 'Weights Vector Dividing Plane', color = 'black')

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

        plt.legend()
        plt.show()

def main(examples_matrix, weights_vec: WeightsVector, iterations: int):
    examples_matrix.plot_classification(weights_vec)
    for i in range(iterations):
        first_mistake_dict = examples_matrix.get_first_mistake(weights_vec)
        weights_vec.update_weights(first_mistake_dict)
    examples_matrix.plot_classification(weights_vec)

    return weights_vec


x = ExamplesMatrix(4,2)
x.generate_random_tags()
weights_vector = WeightsVector([1,1], 0.05)
print('ExamplesMatrix: ', x.val,
    'weights_vector: ', weights_vector.val)
x.plot_classification(weights_vector)
# print('weights_vector: ', weights_vector.val)

# classification_vector = x.get_classification(weights_vector)
# x.plot_classification(classification_vector)
# weights_vector = iteration(x.val, weights_vector)
# classification_vector = x.get_classification(weights_vector)
# x.plot_classification(classification_vector)


