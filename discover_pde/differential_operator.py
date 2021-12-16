import numpy as np
from math import comb

TOL = 0.001

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
        all_partials = LinearOperator.get_all_partials(dimension, order)
        partials = []
        coeffs = []
        for i in range(len(all_partials)):
            if np.abs(vector[i]) > TOL:
                partials.append(all_partials[i])
                coeffs.append(vector[i])

        return LinearOperator(coeffs, partials)

    def get_adjoint(self):
        #TODO: implement
        pass

    def get_vector_length(dimension, order):
        return sum([_num_combi(n,dimension) for n in range(order+1)])

    def get_all_partials(dimension, order):
        partials = []
        for n in range(order+1):
            partial = Partial([n]+([0]*(dimension-1)))
            for i in range(_num_combi(n,dimension)):
                partials.append(partial)
                partial = partial.next_partial()
        return partials



    

        



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
