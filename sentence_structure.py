import nltk
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from unidecode import unidecode


def tokenize(fname):
    f = open(fname, "r")
    taggedSents = []
    s = f.read()
    s = unicode(s, errors="ignore")
    sentences = nltk.sent_tokenize(s)
    for i in sentences:
        taggedSents.append(nltk.pos_tag(nltk.word_tokenize(i)))
    return taggedSents


taggedSents1 = tokenize(
    "/home/shirish/BTECHSEM2/project/books/stories/new_complete_stories.txt")
taggedSents0 = tokenize(
    "/home/shirish/BTECHSEM2/project/books/annotated_books/stories_without_anecdotes/new_all_stories.txt1")

imp_cols = ["N", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]


def featureset_df(taggedSents, value):
    sents1 = []
    for i in taggedSents:
        l = {}
        for j in i:
            if j[1].isalpha():
                # If Noun
                if j[1][0] == "N" or j[1] == "PRP":
                    l["N"] = 1
                else:
                    l[j[1]] = 1
        sents1.append(l)
    imp_cols = ["N", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    df = pd.DataFrame(sents1)
    df.fillna(0, inplace=True)
    features = df[imp_cols]
    features["target"] = value
    return features


features1 = featureset_df(taggedSents1, 1)
features2 = featureset_df(taggedSents1, 0)
features = pd.concat([features1, features2])


model = RandomForestClassifier(n_estimators=100)
model = model.fit(features[imp_cols].values, features["target"].values)
model.score(features1[imp_cols].values, features1["target"].values)
