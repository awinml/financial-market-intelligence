import pandas as pd
import numpy as np
import os

df = pd.read_csv("./datasets/raw/abbvie.csv")
columns = list(df.columns)


def change_name_year(df, name, year_list, columns):
    arr = df.to_numpy()

    for row in range(len(arr)):
        for col in range(len(arr[0])):
            if col == 0:
                arr[row][col] = name
            if col == 1:
                arr[row][col] = year_list[row]

    return pd.DataFrame(arr, columns=columns)


# Returns combined dataset
def combine_dataset(PATH, raw_data_files, company_names_list, year_list):
    for i, (file, company) in enumerate(zip(raw_data_files, company_names_list)):
        df = pd.read_csv(PATH + file)
        df = df.iloc[:, 1:]
        # Setting up appropriate names and year
        df = change_name_year(df, company, year_list, columns)

        # Skipping concatenation in the first iteration
        if i == 0:
            arr = df.to_numpy()
        else:
            arr1 = df.to_numpy()
            arr = np.concatenate((arr, arr1), axis=0)

    return pd.DataFrame(arr, columns=columns)


raw_data_files = os.listdir("./datasets/raw/")
company_names_list = [
    "Meta",
    "Merck & Co.",
    "Alphabet Inc.",
    "Microsoft Corporation",
    "Costco",
    "Pfizer",
    "PepsiCo",
    "AbbVie",
    "Coca-Cola",
    "Mastercard",
]


PATH = './datasets/raw/'

year_list = ['2021','2020','2019','2018','2017']

final_combined_dataset = combine_dataset(PATH, raw_data_files, company_names_list, year_list)

final_combined_dataset.to_csv('./datasets/final_combined_dataset.csv')