import numpy as np
from math import comb

def _num_combi(a,b):
    if b < 1 or a+b < 1:
        return 0
    return comb(a+b-1,b-1)

class Partial():
    def __init__(self,  order_list):
        self.order_list = order_list
        self.dimension = len(order_list)
        self.order = sum(order_list)

    def __str__(self):
        return f"D_{self.order_list}"



    def get_index(self):
        index = 0
        curr_order = self.order

        for i, o in enumerate(self.order_list):
            for j in range(curr_order - o):
                index += _num_combi(j, self.dimension-i-1)
            curr_order -= o
        return index


class LinearOperator():
    def __init__(self, coeffs, partials):
        self.coeffs = coeffs
        self.partials = partials

        # Check if you have a coefficient for each partial
        if len(coeffs) != len(partials):
            raise ValueError('Number of coefficients is different from the number of partials')

        # Check if all partials have the same dimensions
        dimensions = [partial.dimension for partial in partials]
        if dimensions.count(dimensions[0]) != len(dimensions):
            raise ValueError('Dimensions of partials are not the same')

        self.dimension = dimensions[0]
        self.order = max([partial.order for partial in partials])

    def __str__(self):
        partials_with_coeffs = []
        for i in range(len(self.coeffs)):
            partials_with_coeffs.append(f"{self.coeffs[i]}*{self.partials[i]}")
        return "+".join(partials_with_coeffs)
        
        


    def vectorize(self):
        size = sum([_num_combi(n,self.dimension) for n in range(self.order+1)])
        encoded = np.zeros(size)
        for i, partial in enumerate(self.partials):
            index = sum([_num_combi(n,self.dimension) for n in range(partial.order)]) + partial.get_index()
            encoded[index] = self.coeffs[i]
        
        return encoded
    

    def from_vector(vector, dimension, order):
        # TODO: implement
        pass







