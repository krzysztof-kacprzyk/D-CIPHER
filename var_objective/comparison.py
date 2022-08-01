import numpy as np
import pandas as pd
from .utils.lstsq_solver import UnitL1NormLeastSquare_CVX, UnitLstsqLARSImproved, UnitLstsqMD
import time
from datetime import datetime
from tqdm import tqdm
import pickle

if __name__ == "__main__":


    np.random.seed(0)

    # filename = 'results/test_objects.p'
    # with open(filename,'rb') as f:
    #     test_objects = pickle.load(f)
    
    num_tests = 1000
    m = 1000
    n = 7

    df = pd.DataFrame()

    # A = test_objects['A']

    
    for b in tqdm(range(num_tests)):
        
        record = {}

        A = np.random.randn(m,n)

        x = np.random.rand(n) - 0.5
        x /= np.linalg.norm(x,1)

        x *= (np.random.rand() * 4 - 1)  
        
        b = np.dot(A,x)

        # b = np.random.randn(m)

        md = UnitLstsqMD(A)
        lars_imp = UnitLstsqLARSImproved(A)
        cvx = UnitL1NormLeastSquare_CVX(A)
        start = time.time()
        try: 
            loss_md, x_md = md.solve(b, take_mean=False)
            # print(f"MD | Loss {loss_md} | Time: {end - start} seconds")
        except:
            loss_md = np.nan
            # print("MD failed")
        end = time.time()
        record['md_loss'] = loss_md
        record['md_time'] = end - start

        start = time.time()
        try:
            loss_cvx, x_cvx = cvx.solve(b, take_mean=False)
            # print(f"CVX | Loss: {loss_cvx} | Time: {end - start} seconds")
        except:
            loss_cvx = np.nan
            # print("CVX failed")
        end = time.time()
        record['cvx_loss'] = loss_cvx
        record['cvx_time'] = end - start

        start = time.time()
        try: 
            loss_lars_imp, x_lars_imp  = lars_imp.solve(b, take_mean=False)
            # print(f"LARS_Improved | Loss {loss_lars_imp} | Time: {end - start} seconds")
        except:
            loss_lars_imp = np.nan
            # print("LARS improved failed")
        end = time.time()
        record['lars_imp_loss'] = loss_lars_imp
        record['lars_imp_time'] = end - start

        # standard_least_square = np.linalg.inv(np.transpose(A) @ A) @ np.transpose(A) @ b
        # L1_norm = np.sum(np.absolute(standard_least_square))

        # if np.abs(loss_lars_imp - loss_cvx) / loss_cvx > 0.01:
        #     print(np.abs(loss_lars_imp - loss_cvx) / loss_cvx)
        #     print(x)
        #     print(np.linalg.norm(x,1))
        #     print(loss_cvx)
        #     print(loss_lars_imp)
        #     print(x_cvx)
        #     print(x_lars_imp)
        # print(f"LSTSQ loss: {((np.dot(A,standard_least_square/L1_norm)-b) ** 2).sum()}")
        # print(L1_norm)

        record['length_ground_truth'] = np.linalg.norm(x,1)


        df = df.append(record,ignore_index=True)
    
    dt = datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
    filename = f"results/comparison_n{n}_{dt}.csv"
    df.to_csv(filename)
    