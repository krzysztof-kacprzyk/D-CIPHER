from types import new_class
import numpy as np
from scipy.special import comb
from abc import ABC, abstractmethod

TOL = 0.0

def _num_combi(a,b):
    if b < 1 or a+b < 1:
        return 0
    return comb(a+b-1,b-1, exact=True)



class Partial():
    def __init__(self,  order_list):
        self.order_list = list(order_list)
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

    def get_global_index(self):
        return sum([_num_combi(n,self.dimension) for n in range(self.order)]) + self.get_index()


    def next_partial(self):
        new_order_list = list(self.order_list)
        if np.sum(np.array(self.order_list) == 0) == self.dimension:
            return None # there is only one partial of order 0
        else:
            last_non_zero_index = np.max(np.nonzero(self.order_list))
            last_non_zero = self.order_list[last_non_zero_index]
            if last_non_zero_index == len(self.order_list) - 1:
                # If the last non-zero element is at the end
                if last_non_zero == self.order:
                    # If it is the last possible partial
                    return None
                else:
                    second_last_non_zero_index = np.max(np.nonzero(self.order_list[:-1]))
                    new_order_list[second_last_non_zero_index] -= 1
                    new_order_list[last_non_zero_index] = 0
                    new_order_list[second_last_non_zero_index+1] = last_non_zero + 1
            else:
                # If the last non-zero element is NOT at the end
                if last_non_zero == 1:
                    new_order_list[last_non_zero_index] = 0
                    new_order_list[last_non_zero_index+1] = 1
                else:
                    new_order_list[last_non_zero_index] -= 1
                    new_order_list[last_non_zero_index+1] = 1
            return Partial(new_order_list)

    def is_zero(self):
        for o in self.order_list:
            if o != 0:
                return False
        return True


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
            index = partial.get_global_index()
            encoded[index] = self.coeffs[i]
        
        return encoded

    def get_length(self, norm='l2'):
        result = 0
        if norm == 'l2':
            for coeff in self.coeffs:
                result += (coeff ** 2)
            return np.sqrt(result)
        elif norm == 'l1':
            for coeff in self.coeffs:
                result += np.abs(coeff)
            return result

        

    def from_vector(vector, dimension, order, zero_partial=True):
    
        if zero_partial:
            offset = 0
        else:
            offset = 1
        all_partials = LinearOperator.get_all_partials(dimension, order)
        partials = []
        coeffs = []
        for i in range(0,len(all_partials)-offset):
            if np.abs(vector[i]) > TOL:
                partials.append(all_partials[i+offset])
                coeffs.append(vector[i])

        return LinearOperator(coeffs, partials)

    def get_adjoint(self):
        new_coeffs = []
        for i, coeff in enumerate(self.coeffs):
            if self.partials[i].order % 2 == 0:
                new_coeffs.append(coeff)
            else:
                new_coeffs.append(-coeff)
        return LinearOperator(new_coeffs, self.partials)

    def get_vector_length(dimension, order):
        return sum([_num_combi(n,dimension) for n in range(order+1)])

    #TODO: this can be probably improved by changing next_partial for some recursive generating algorithm
    def get_all_partials(dimension, order):
        partials = []
        for n in range(order+1):
            partial = Partial([n]+([0]*(dimension-1)))
            for i in range(_num_combi(n,dimension)):
                partials.append(partial)
                partial = partial.next_partial()
        return partials

    def normalize(self, norm='l2'):
        new_coeffs = []
        length = self.get_length(norm=norm)
        for coeff in self.coeffs:
            new_coeffs.append(coeff / length)
        return LinearOperator(new_coeffs, self.partials)

    def get_sign(self):
        vector = self.vectorize()
        for v in vector:
            if v != 0.0:
                return np.sign(v)
        return 0.0

    def reverse_sign(self):
        new_coeffs = []
        for coeff in self.coeffs:
            new_coeffs.append(-coeff)
        return LinearOperator(new_coeffs, self.partials)

def id(x):
    return x

def square(x):
    return x ** 2


class EFunction:

    def __init__(self, act, symbol):
        self.act_f = act
        self.symbol = symbol
    
    def act(self,x,u):
        return self.act_f(x,u)
    
    def __str__(self):
        return self.symbol
    

proj_0 = EFunction(lambda vx,vu: vu[0], "u_0")
proj_1 = EFunction(lambda vx,vu: vu[1], "u_1")
square_0 = EFunction(lambda vx,vu: vu[0]**2, "u_0^2")
cubic_0 = EFunction(lambda vx,vu: vu[0]**3, "u_0^3")

class ED:

    def __init__(self, partial, h, a=None):
        if a is not None:
            print("a is not implemented yet")
        if not isinstance(partial,Partial):
            print("partial has to be an instance of Partial")

        self.partial = partial
        self.h = h
        self.a = a
    
    def __str__(self):
        return f"{self.partial}({self.h})"

    def sign(self):
        return (-1) ** self.partial.order
    

def extract_differential_operator(dictQ, weights):
    return " + ".join([f"{weight} * {e_derivative}" for (weight,e_derivative) in zip(weights,dictQ)])


if __name__ == "__main__":

    dictionaryQ = [ED(Partial([1,0]),proj_0),
                                ED(Partial([0,1]),square_0)]
    print(extract_differential_operator(dictionaryQ,[1.2,0.8]))

    print(ED(Partial([0,1]),square_0).sign())

    print(proj_0.act([0],[1,2]))







# p = Partial([3,0,0,0])
# for i in range(20):
#     if p == None:
#         print("None")
#         break
#     print(p)
#     p = p.next_partial()

# L = LinearOperator([1,-2,3,5,-6],[Partial([0,1,2,0]), Partial([3,0,0,0]),Partial([1,1,1,0]),Partial([0,0,0,3]),Partial([0,0,1,2])])
# vector = L.vectorize()
# print(L)
# print(vector)
# Q = LinearOperator.from_vector(vector,4,3)
# print(Q) 
