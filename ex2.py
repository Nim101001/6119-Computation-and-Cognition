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


class ExamplesVector(object):

    # create data

    def __init__(self, p, tags=pd.DataFrame(None)):
        self.p = p
        self.val = np.array([[random.uniform(-1, 1), 1] for ex in range(p)])
        self.tags = np.array([val[0]**3 - val[0]**2 for val in self.val])

    # def generate_random_tags(self):
    #     self.tags = pd.Series([random.choice([True, False]) for ex in range(len(self.val))])
    #
    # def generate_ex_tags(self):
    #     self.tags = pd.Series([ex[0] > ex[1] for idx, ex in self.val.iterrows()]).astype(int)

    # analyse

    def get_correlation_matrix(self):
        cor_matrix = 1/self.p * [[ex2 * ex1 for ex2 in self.val]for ex1 in self.val]
        return cor_matrix

    def get_correlation_vector(self):
        cor_vector = self.val * self.tags
        return cor_vector

    def get_optimal_weights_vector(self):
        cor_matrix = self.get_correlation_matrix()
        cor_vector = self.get_correlation_vector()
        #transpose


    def get_classification(self, weights_vec: WeightsVector):
        output = self.val * weights_vec.val
        return output

    # def check_weights(self, weights_vec: WeightsVector) -> pd.Series:
    #     # output[i]=1 if example is rightly classified by weights_vec,output[i]=0 if not.
    #     output = pd.Series(np.ones(len(self.val)) - np.abs(self.tags.to_numpy() - np.array(self.get_classification(weights_vec))))
    #     return output

    def get_mistake(self, classification: np.array) -> np.array:
        mistake_vec = self.tags - classification
        return mistake_vec

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

        plt.scatter(true_dots.iloc[:,0], true_dots.iloc[:,1], s = 40, color = 'blue', label = 'Classified as True')
        plt.scatter(false_dots.iloc[:,0], false_dots.iloc[:,1], s = 40, color = 'red', label = 'Classified as False')
        # plt.scatter(success_dots.iloc[:, 0], success_dots.iloc[:, 1], s=20, color='green',
        #             label='Successful Classification')
        # plt.scatter(failure_dots.iloc[:, 0], failure_dots.iloc[:, 1], s=20, color='brown',
        #             label='Failed Classification')

        plt.plot([0, weights_vec.val[0]], [0, weights_vec.val[1]], color='black', label='Weights Vector')
        dividing_plane_func = weights_vec.get_dividing_plane_func()
        plt.plot([-10, 10], [dividing_plane_func(-10), dividing_plane_func(10)], color='grey',
                 label='Weights Vector Dividing Plane')

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

        plt.legend()
        plt.show()


def find_weights_vector(examples_matrix, weights_vec: WeightsVector):
    # examples_matrix.plot_classification(weights_vec, title='Start')

    i = 0
    check_weights = examples_matrix.check_weights(weights_vec)
    x = set(check_weights.astype(int))
    while not (check_weights.astype(int) == 1).all():
        for idx, val in enumerate(check_weights):
            if val == 0:
                mistake = examples_matrix.get_mistake(idx)
                weights_vec.update_weights(mistake)
        check_weights = examples_matrix.check_weights(weights_vec)
        i+=1
        if i == 5000:
            print('An example surpassed 5000 iterations')
    # examples_matrix.plot_classification(weights_vec, title=f'Finish: after {i} iterations')
    # print('weights vector: ', weights_vec,
    #       'check weights: ', check_weights)
    return weights_vec

def get_mistake_angle(examples_matrix, weights_vec: WeightsVector):
    weights_vec = find_weights_vector(examples_matrix, weights_vec)
    correct_vector = np.array([1,-1])
    cos_angle = correct_vector.dot(weights_vec.val)/(np.linalg.norm(np.array(weights_vec.val)) * np.linalg.norm(np.array(correct_vector)))
    angle = np.arccos(cos_angle)
    return angle

#new
examples_vector = ExamplesVector(500)
weights_vec = None
mistake = examples_vector.get_mistake(examples_vector.get_classification(weights_vec))

#old
examples_matrix = ExamplesMatrix(1000, 2)
examples_matrix.generate_ex_tags()
weights_vector = WeightsVector([1, 1], 1)


#2d

weights_vector = WeightsVector([1, 1], 1)
examples_number = [5, 20, 30, 50, 100, 150, 200, 500]
p_mistakes = []
for p in examples_number:
    mistakes = []
    for i in range(100):
        examples_matrix = ExamplesMatrix(p,2)
        examples_matrix.generate_ex_tags()
        mistake_angle = get_mistake_angle(examples_matrix, weights_vector)
        mistakes.append(mistake_angle)
    average_mistake = np.average(np.array(mistakes))
    p_mistakes.append(average_mistake)
print(p_mistakes)

plt.title('Mistake angle as a function of examples number, \n averaged on 100 simultaions')
plt.xlabel('Examples Number')
plt.ylabel('Average mistake (rads)')
plt.plot(examples_number, p_mistakes)
plt.scatter(examples_number, p_mistakes)
plt.show()