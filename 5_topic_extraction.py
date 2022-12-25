import pandas as pd
import numpy as np
import nltk
from rake_nltk import Rake

nltk.download("punkt")
nltk.download("stopwords")

summarized_data = pd.read_csv("./datasets/summarized_data.csv")
li = list(summarized_data.iloc[:, 1])

topic_extractor = Rake()
topic_extractor.extract_keywords_from_text(summarized_data.iloc[0, 2])
topic_extractor.get_ranked_phrases()
topic_extractor.get_ranked_phrases_with_scores()


def topic_extraction(df, exclude_columns):

    topic_extractor = Rake()

    temp1 = df.iloc[:, :exclude_columns]
    arr = df.iloc[:, exclude_columns:].to_numpy()
    columns = list(df.iloc[:, exclude_columns:].columns)

    for row in range(len(arr)):
        for col in range(len(arr[0])):
            topic_extractor.extract_keywords_from_text(arr[row][col])
            arr[row][col] = topic_extractor.get_ranked_phrases_with_scores()

    temp2 = pd.DataFrame(arr, columns=columns)

    return pd.concat([temp1, temp2], axis=1)


summarized_data = topic_extraction(summarized_data, exclude_columns=2)
summarized_data.to_csv('./datasets/topic_extraction_rakeNLTK.csv', index=False)