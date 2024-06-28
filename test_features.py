#!/usr/bin/env python3

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from statistics import mean
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import mutual_info_score

sia = SentimentIntensityAnalyzer()


def extract_features(text):
    features = {}
    compound_scores = []
    positive_scores = []
    neutral_scores = []
    negative_scores = []

    # Divide el texto en oraciones. Por lo que lei, asi es mejor para peliculas
    for sentence in nltk.sent_tokenize(text):
        # print(sentence)
        compound_scores.append(sia.polarity_scores(sentence)["compound"])
        positive_scores.append(sia.polarity_scores(sentence)["pos"])
        neutral_scores.append(sia.polarity_scores(sentence)["neu"])
        negative_scores.append(sia.polarity_scores(sentence)["neg"])
        # print(
        #     compound_scores[-1],
        #     positive_scores[-1],
        #     neutral_scores[-1],
        #     negative_scores[-1],
        # )

    features["compound"] = mean(compound_scores)
    features["pos"] = mean(positive_scores)
    features["neu"] = mean(neutral_scores)
    features["neg"] = mean(negative_scores)

    return features


def extract_features_df(df):
    df_dict = {
        "compound": [],
        "positive": [],
        "neutral": [],
        "negative": [],
        "label": [],
    }
    for i, row in df.iterrows():
        features = extract_features(row["message"])
        # El de abajo hace sobre todo el texto. En la practica no es tan bueno
        # features = sia.polarity_scores(row["message"])
        # The compound is the aggregated score
        df_dict["compound"].append(features["compound"])
        df_dict["positive"].append(features["pos"])
        df_dict["neutral"].append(features["neu"])
        df_dict["negative"].append(features["neg"])
        df_dict["label"].append(row["label"])

    df_final = pd.DataFrame(df_dict)
    return df_final


train = pd.read_csv("sentiment/train.csv")
train_sample = train.loc[0:500, :]

df = extract_features_df(train_sample)
np_features = df.iloc[:, 0:-1].to_numpy()

kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(np_features)

results_rs = rand_score(df.iloc[:, -1].to_numpy(), kmeans.labels_)
results_ss = silhouette_score(np_features, kmeans.labels_)
results_mi = mutual_info_score(df.iloc[:, -1].to_numpy(), kmeans.labels_)
print("Rand score: ", results_rs)
print("Silhouette Score score: ", results_ss)
print("Mutual Info score: ", results_mi)
