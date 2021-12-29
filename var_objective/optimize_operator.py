import numpy as np
import torch
from itertools import product
from .differential_operator import LinearOperator, all_derivatives

EPS = torch.tensor(0.0001)
INF = torch.tensor(1000000)

def normalize(x):
    with torch.no_grad():
        return x.div(torch.norm(x,2))


   
    
class Model:

    def __init__(self, derivative_tensor, g_tensor, alpha, beta, num_lower_order, homogeneous, norm_sum=False):
        self.derivative_tensor = derivative_tensor
        self.g_tensor = g_tensor
        self.alpha = alpha
        self.beta = beta
        self.num_lower_order = num_lower_order
        self.homogeneous = homogeneous
        self.norm_sum = norm_sum

    def loss(self, weights):
        if self.homogeneous:
            with torch.no_grad():
                weights.div_(torch.norm(weights,2))
        pred = torch.mul(weights, self.derivative_tensor).sum(-1) - self.g_tensor
        if self.norm_sum:
            square_loss = torch.pow(pred, 2.0).mean()
        else:
            square_loss = torch.pow(pred, 2.0).sum() 
        l1_loss = torch.norm(weights,1)
        min_loss = torch.max(torch.log(torch.pow(weights[self.num_lower_order:],2)+EPS))
        # print(f"Sq: {square_loss} | l1: {l1_loss} | min: {min_loss}")
        return square_loss + self.alpha * l1_loss  - self.beta * min_loss




class VariationalWeightsFinder:

    def __init__(self,estimated_dataset, field_index, full_grid, dimension, order, basis, index_limits, alpha=0.1, beta=0.1, optim_name='sgd', optim_params={'lr':0.01}, num_epochs=100, patience=10,  seed=0):
        self.estimated_dataset = estimated_dataset
        self.field_index = field_index
        self.full_grid = full_grid
        self.dimension = dimension
        self.order = order
        self.basis = basis
        self.index_limits = index_limits
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.optim_name = optim_name
        self.optim_params = optim_params
        self.num_epochs = num_epochs
        self.patience = patience

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if len(self.index_limits) != self.basis.num_indexes:
            raise ValueError("Length of index_limits does not match num_indexes of basis")

        #TODO: check dataset shape and full_grid shape

        basis_indexes = list(product(*[range(1, self.index_limits[i]+1) for i in range(len(self.index_limits))]))
        self.S = len(basis_indexes)

        all_partials = LinearOperator.get_all_partials(self.dimension, self.order)
        self.J = len(all_partials)

        self.test_function_part = np.zeros((self.S,self.J,*self.full_grid.shape))
        for j, partial in enumerate(all_partials):
            for s, indexes in enumerate(basis_indexes):
                self.test_function_part[s][j] = self.basis.get_tensor(indexes, self.full_grid, partial)

        self.D = self.estimated_dataset.shape[0]
        self.N = self.estimated_dataset.shape[1]
        self.M = self.full_grid.num_dims
        self.J = LinearOperator.get_vector_length(self.dimension, self.order)

        if self.field_index >= self.N:
            raise ValueError(f"There is no field with index {self.field_index}")
            
        static_part = np.zeros([self.D,self.S,self.J,*self.full_grid.shape])

        for d in range(self.D):
            static_part[d] = np.multiply(self.test_function_part, self.estimated_dataset[d][field_index])
        self.integrals = np.multiply(static_part, self.full_grid.for_integration()).sum(axis=tuple(range(3, len(static_part.shape))))

        assert self.integrals.shape == (self.D, self.S, self.J)

        assert self.order > 0
        self.num_lower_order = LinearOperator.get_vector_length(self.dimension, self.order-1)
        

        self.grid_and_fields = np.zeros([self.D,(self.M + self.N), *full_grid.shape])
        for d in range(self.D):
            self.grid_and_fields[d] = np.stack([*full_grid.by_axis(),*(estimated_dataset[d])],axis=0)


    def find_weights(self, g_part=None, from_covariates=True, normalize_g=True):

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # g_part_func should have shape (D,*self.full_grid.shape) or be None

        homogeneous = (g_part is None)
          
        if homogeneous:
            g_integrals = np.zeros((self.D, self.S))
        # else:
        #     if from_covariates:
        #         g_part = np.reshape(g_part, (self.D, *(self.full_grid.shape)))
        #     assert g_part.shape == (self.D,*self.full_grid.shape)

        #     g_part = np.multiply(g_part[:,np.newaxis], self.test_function_part[np.newaxis,:,0])
        #     assert g_part.shape == (self.D,self.S,*(self.full_grid.shape))

        #     g_integrals = np.multiply(g_part, self.full_grid.for_integration()).sum(axis=tuple(range(2, len(g_part.shape))))
        #     assert g_integrals.shape == (self.D, self.S)

            integrals_tensor = torch.from_numpy(self.integrals[:,:,1:]) # shape: (D,S,J)

            model = Model(integrals_tensor,torch.from_numpy(g_integrals),self.alpha,self.beta,self.num_lower_order,homogeneous)

            weights = torch.randn(self.J-1, requires_grad=True)
            if self.optim_name == 'sgd':
                optimizer = torch.optim.SGD([weights],**self.optim_params)
            elif self.optim_name == 'adam':
                optimizer = torch.optim.Adam([weights],**self.optim_params)
            else:
                raise ValueError(f'Unknown optimizer {self.optim_name}')

            prev_best_loss = INF
            prev_best_weights = torch.zeros_like(weights)
            epochs_no_improvement = 0
            counter = 0

            for i in range(self.num_epochs):
                loss = model.loss(weights)
                counter += 1

                if loss >= prev_best_loss:
                    epochs_no_improvement += 1
                else:
                    prev_best_loss = loss
                    with torch.no_grad():
                        prev_best_weights = torch.clone(weights)
                    epochs_no_improvement = 0

                if epochs_no_improvement >= self.patience:
                    break

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


            print(f"Loss: {prev_best_loss} | Epochs: {counter}")         
            print(f"Weights: {prev_best_weights}")
            target = torch.tensor([-1.0,-2.0])
            print(model.loss(target))
            return (prev_best_loss, prev_best_weights, None)

        else:

            if from_covariates:
                g_part = np.reshape(g_part, (self.D, *(self.full_grid.shape)))

            assert g_part.shape == (self.D,*self.full_grid.shape)

            g_part = np.multiply(g_part[:,np.newaxis], self.test_function_part[np.newaxis,:,0])
            assert g_part.shape == (self.D,self.S,*(self.full_grid.shape))

            g_integrals = np.multiply(g_part, self.full_grid.for_integration()).sum(axis=tuple(range(2, len(g_part.shape))))
            assert g_integrals.shape == (self.D, self.S)

            if normalize_g:
                length = np.linalg.norm(g_integrals,2)
                n = len(g_integrals)
                if length == 0.0:
                    g_integrals[:] = 1.0 / np.sqrt(n)
                else:
                    g_integrals = g_integrals / length

            X = np.reshape(self.integrals,(-1,self.J))
            y = np.reshape(g_integrals,(-1,))

            weights, res, rank, s = np.linalg.lstsq(X[:,1:],y,rcond=None) # we do not take the first column of X because we do not want 0th order partial

            return (res[0],weights)
            

        
            

class MSEWeightsFinder:

    def __init__(self, dataset, field_index, grid, dimension, order, engine, alpha=0.1, beta=0.1, optim_name='sgd', optim_params={'lr':0.01}, num_epochs=100, patience=10, seed=0):
        self.dataset = dataset
        self.field_index = field_index
        self.grid = grid
        self.dimension = dimension
        self.order = order
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.optim_name = optim_name
        self.optim_params = optim_params
        self.num_epochs = num_epochs
        self.patience = patience
        self.engine = engine
        self.found_weights = None

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        #TODO: check dataset shape and grid shape

        self.D = self.dataset.shape[0]
        self.N = self.dataset.shape[1]
        self.M = self.grid.num_dims
        self.J = LinearOperator.get_vector_length(self.dimension, self.order)

        if self.field_index >= self.N:
            raise ValueError(f"There is no field with index {self.field_index}")
            

        self.derivative_dataset = np.stack([all_derivatives(dataset[d,field_index],grid,dimension,order, self.engine) for d in range(self.D)], axis=0)
        assert self.derivative_dataset.shape == (self.D,self.J,*grid.shape)

        assert self.order > 0
        self.num_lower_order = LinearOperator.get_vector_length(self.dimension, self.order-1)

        self.grid_and_fields = np.zeros([self.D,(self.M + self.N), *self.grid.shape])
        for d in range(self.D):
            self.grid_and_fields[d] = np.stack([*self.grid.by_axis(),*(self.dataset[d])],axis=0)


    def find_weights(self, g_part=None, from_covariates=True, normalize_g=True):

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # g_part_func should have shape (D,*self.full_grid.shape) or be None

        if g_part is None:
            homogeneous = True
        else:
            homogeneous = False

        if homogeneous:
            g_part = np.zeros((self.D, *self.grid.shape))

            assert g_part.shape == (self.D, *self.grid.shape)

            derivative_tensor = torch.from_numpy(np.moveaxis(self.derivative_dataset,1,-1))

            g_tensor = torch.from_numpy(g_part)


            model = Model(derivative_tensor,g_tensor,self.alpha,self.beta,self.num_lower_order,homogeneous,norm_sum=True)

            weights = torch.randn(self.J, requires_grad=True)
            if self.optim_name == 'sgd':
                optimizer = torch.optim.SGD([weights],**self.optim_params)
            elif self.optim_name == 'adam':
                optimizer = torch.optim.Adam([weights],**self.optim_params)
            else:
                raise ValueError(f'Unknown optimizer {self.optim_name}')

            prev_best_loss = INF
            prev_best_weights = torch.zeros_like(weights)
            epochs_no_improvement = 0
            counter = 0
            for i in range(self.num_epochs):
                loss = model.loss(weights)
                counter += 1
                # print(f"Epoch {i+1} | Loss: {loss}")

                if loss >= prev_best_loss:
                    epochs_no_improvement += 1
                else:
                    prev_best_loss = loss
                    with torch.no_grad():
                        prev_best_weights = torch.clone(weights)
                    epochs_no_improvement = 0

                if epochs_no_improvement >= self.patience:
                    break

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print(f"Loss: {prev_best_loss} | Epochs: {counter}")        
            print(f"Weights: {prev_best_weights}")
            # target = torch.tensor([0.0,1.0,-1.0])
            # print(normalize(target))
            # print(model.loss(target))
            return (prev_best_loss.item(), prev_best_weights, None)

        else:

            if from_covariates:
                if normalize_g:
                    length = np.linalg.norm(g_part,2)
                    n = len(g_part)
                    if length == 0.0:
                        g_part[:] = 1.0
                    else:
                        g_part = (np.sqrt(n) * g_part) / length
                g_part = np.reshape(g_part, (self.D, *(self.grid.shape)))

            assert g_part.shape == (self.D, *self.grid.shape)
            derivative_part = np.moveaxis(self.derivative_dataset,1,-1)

            X = np.reshape(derivative_part,(-1,self.J))
            y = np.reshape(g_part,(-1,))

            weights, res, rank, s = np.linalg.lstsq(X[:,1:],y,rcond=None)

            return (res[0],weights)
            






        
        





         
            



    

