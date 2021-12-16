import numpy as np
import torch
from itertools import product
from .differential_operator import LinearOperator

EPS = torch.tensor(0.0001)

def _variational_loss(x, weights, alpha, beta, num_lower_order):
    pred = torch.mul(weights, x).sum(-1)
    # with torch.no_grad():
    #     weights.div_(torch.norm(weights,2))
    return torch.mul(pred, pred).sum() + alpha * torch.norm(weights,1) - beta * torch.min(torch.log(torch.mul(weights[num_lower_order:],weights[num_lower_order:])+EPS))    

def find_weights(estimated_dataset, field_indexes, full_grid, dimension, order, basis, index_limits, seed=0):
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(index_limits) != basis.num_indexes:
        raise ValueError("Lenght of index_limits does not match num_indexes of basis")

    basis_indexes = list(product(*[range(1, index_limits[i]+1) for i in range(len(index_limits))]))
    S = len(basis_indexes)

    all_partials = LinearOperator.get_all_partials(dimension, order)
    J = len(all_partials)

    test_function_part = np.zeros((S,J,*full_grid.shape))
    for j, partial in enumerate(all_partials):
        for s, indexes in enumerate(basis_indexes):
            test_function_part[s][j] = basis.get_tensor(indexes, full_grid, partial)

    D = estimated_dataset.shape[0]
    N = estimated_dataset.shape[1]

    for j in field_indexes:
        if j >= N:
            raise ValueError(f"There is no field with index {j}")
        
        static_part = np.zeros([D,S,J,*full_grid.shape])

        for d in range(D):
            static_part[d] = np.multiply(test_function_part, estimated_dataset[d][j])
        integrals = np.multiply(static_part, full_grid.for_integration()).sum(axis=tuple(range(3, len(static_part.shape))))

        assert order > 0
        num_lower_order = LinearOperator.get_vector_length(dimension, order-1)
        integrals_tensor = torch.from_numpy(integrals)

        weights = torch.randn(LinearOperator.get_vector_length(dimension, order), requires_grad=True)

       

        optimizer = torch.optim.SGD([weights], lr=0.01, momentum=0.9)

        for i in range(200):
            loss = _variational_loss(integrals_tensor, weights, torch.tensor(0.5), torch.tensor(1.0), num_lower_order)
            print(f"Loss: {loss}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Weights: {weights.div(torch.norm(weights,2))}")
        target = torch.tensor([0.0,1.0,-1.0])
        print(target.div(torch.norm(target,2)))

        print(_variational_loss(integrals_tensor,torch.tensor([0.0,1.0,-1.0]),torch.tensor(0.5),torch.tensor(1.0), num_lower_order))




        
        





         
            



    

