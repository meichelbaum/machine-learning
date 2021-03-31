import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PATH_TO_DATA = 'training_data/25082014/test'




def get_single_dataset(path, number=1):
    dataset = pd.read_csv(path + str(number) + ".xls", delimiter='\t')
    dataset.drop(dataset[["Time", 'Remark', 'Normal/Error']], axis=1, inplace=True)
    dataset.set_index('Zeit:', inplace=True)
    dataset.iloc[:, 0:2] = dataset.iloc[:, 0:2].astype(str) \
        .apply(lambda x: x.str.replace(',', '.')) \
        .astype(float)
    
    return dataset


def get_combined_dataset(path, start, end):
    if not start < end:
        print("End has to be larger than start.")
        return
    res = get_single_dataset(path, start)
    for i in range(start + 1, end):
        res = res.append(get_single_dataset(path, i))
    return res


if __name__ == "__main__":
    get_combined_dataset(PATH_TO_DATA, 1, 14).to_csv(PATH_TO_DATA + '_combined.csv', index=False)
