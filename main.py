# This is a sample Python script.

import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt


class WeightsVector(object):

    def __init__(self, val, etha=0.1):
        self.val = val
        self.etha = etha

    def update_weights(self, update: dict):
        # update is expected in form [{'tag': Bool, 'example': list}]

        self.val = self.val + self.etha * (2 * update['tag'] - 1) * update['example']

    def get_dividing_plane_func(self):
        val = self.val
        slope = -(val[0] / val[1])  # accoridng to (a,b)*(c,d)=0->c=-(b/a)*d for d =1 c=-(b/a) so slope = d/c=-(a/b)

        def func(x):
            return x * slope

        return func


class ExamplesMatrix(object):

    # create data

    def __init__(self, p, d, tags=pd.DataFrame(None)):
        self.val = pd.DataFrame(data=[[random.uniform(-10, 10) for dim in range(d)] for ex in range(p)])
        self.tags = tags

    def generate_random_tags(self):
        self.tags = pd.Series([random.choice([True, False]) for ex in range(len(self.val))])

    def generate_ex_tags(self):
        self.tags = pd.Series([ex[0] > ex[1] for idx, ex in self.val.iterrows()]).astype(int)

    # analyse

    def get_classification(self, weights_vec: WeightsVector):
        output = np.heaviside((self.val @ weights_vec.val), 0)
        return output

    def check_weights(self, weights_vec: WeightsVector) -> pd.Series:
        # output[i]=1 if example is rightly classified by weights_vec,output[i]=0 if not.
        output = pd.Series(
            np.ones(len(self.val)) - np.abs(self.tags.to_numpy() - np.array(self.get_classification(weights_vec))))
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

    def get_mistake(self, index, weights_vec: WeightsVector):
        mistake = self.val.iloc[index]
        mistake_tag = self.tags.iloc[index]

        output = {'index': first_mistake_index,
                  'tag': first_mistake_tag,
                  'example': first_mistake}
        return output

    # replace get_first_mistake with _get_mistake
    def get_first_mistake(self, weights_vec: WeightsVector):
        first_mistake_index = self.get_first_mistake_index(weights_vec)
        first_mistake = self.val.iloc[first_mistake_index]
        first_mistake_tag = self.tags.iloc[first_mistake_index]

        output = {'index': first_mistake_index,
                  'tag': first_mistake_tag,
                  'example': first_mistake}
        return output

    # plot

    def plot_classification(self, weights_vec: WeightsVector, title: str):

        # assumes d = 2

        plt.title(title)
        classification_vec = self.get_classification(weights_vec)
        weights_success_vec = self.check_weights(weights_vec)

        true_dots = self.val[classification_vec == 1]
        false_dots = self.val[classification_vec == 0]
        success_dots = self.val[weights_success_vec == 1]
        failure_dots = self.val[weights_success_vec == 0]

        # plt.scatter(true_dots.iloc[:,0], true_dots.iloc[:,1], s = 40, color = 'blue', label = 'Classified as True')
        # plt.scatter(false_dots.iloc[:,0], false_dots.iloc[:,1], s = 40, color = 'red', label = 'Classified as False')
        plt.scatter(success_dots.iloc[:, 0], success_dots.iloc[:, 1], s=20, color='green',
                    label='Successful Classification')
        plt.scatter(failure_dots.iloc[:, 0], failure_dots.iloc[:, 1], s=20, color='brown',
                    label='Failed Classification')

        plt.plot([0, weights_vec.val[0]], [0, weights_vec.val[1]], color='black', label='Weights Vector')
        dividing_plane_func = weights_vec.get_dividing_plane_func()
        plt.plot([-10, 10], [dividing_plane_func(-10), dividing_plane_func(10)], color='grey',
                 label='Weights Vector Dividing Plane')

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

        plt.legend()
        plt.show()


def find_weights_vector(examples_matrix, weights_vec: WeightsVector):
    i = 0
    examples_matrix.plot_classification(weights_vec, title=f'Start: i = {i}')

    mistake_dict_before_update = {'index': -1}
    mistake_dict_after_update = examples_matrix.get_first_mistake(weights_vec)

    while not mistake_dict_after_update['index'] == -1:
        check_weights = examples_matrix.check_weights(weights_vec)
        for idx, val in check_weights.enumerate():
            if val == 0:
                mistake = examples_matrix.get_mistake(idx)
                weights_vec.update_weights(mistake)

        mistake_dict_before_update = mistake_dict_after_update  # before_update gets value of mistake dict after *previous* update. Then after_update is updated, and if no change was found, algprithm halts.
        weights_vec.update_weights(mistake_dict_after_update)
        mistake_dict_after_update = examples_matrix.get_first_mistake(weights_vec)
        i += 1
    examples_matrix.plot_classification(weights_vec, title=f'Finish: i = {i}')

    return weights_vec


examples_matrix = ExamplesMatrix(100, 2)
examples_matrix.generate_ex_tags()
print('tags: ', examples_matrix.tags)
weights_vector = WeightsVector([1, 1], 0.05)
find_weights_vector(examples_matrix, weights_vector)


