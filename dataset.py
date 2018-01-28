import numpy as np
import pandas as pd

dataset = []

def create_dataset(from_num, to_num, file_name):
    
    raw_data = np.array(pd.read_excel(file_name))
    price = raw_data[from_num:to_num, 0]
    price = price.reshape(to_num - from_num, 1)
    log_price = np.log(price)
    sol = raw_data[from_num:to_num, -2:]
    sol = sol.reshape(to_num - from_num, 2)
    
    for i in range(from_num+10,to_num):
        features = np.vstack((log_price[i],
                         log_price[i-1],
                         log_price[i-2],
                         log_price[i-3],
                         log_price[i-4],
                         log_price[i-5],
                         log_price[i-6],
                         log_price[i-7],
                         log_price[i-8],
                         log_price[i-9],
                         log_price[i-10],
                          ))
        features = (features - log_price[i, 0])*100
        dataset.append(features)

    return dataset, sol[10:]



