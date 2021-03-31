import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PATH_TO_DATA = 'training_data/Typ1_intrapoliert/test'


# left data = xls'

def get_single_dataset(path, number=1):
    right_dataset = pd.read_csv(path + str(number) + ".txt", delimiter="\t")
    right_dataset.drop(right_dataset[['Totalisator', 'Alarm Nr. 1', 'Unnamed: 4']], axis=1, inplace=True)
    right_dataset.set_index('Zeit:', inplace=True)
    right_dataset.sort_index(inplace=True)
    right_dataset.iloc[:, 0] = right_dataset.iloc[:, 0].astype(str) \
        .apply(lambda x: x.replace(',', '.')) \
        .astype(float)

    left_dataset = pd.read_csv(path + str(number) + ".xls", delimiter='\t')
    left_dataset.drop(left_dataset[["Time", 'Remark', 'Normal/Error']], axis=1, inplace=True)
    left_dataset.set_index('Zeit:', inplace=True)
    left_dataset.iloc[:, 0:2] = left_dataset.iloc[:, 0:2].astype(str) \
        .apply(lambda x: x.str.replace(',', '.')) \
        .astype(float)

    result = pd.merge_asof(left_dataset, right_dataset, on="Zeit:")
    return result


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
