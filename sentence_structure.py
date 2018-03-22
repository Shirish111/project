
# coding: utf-8

import nltk
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix


def tokenize(fname):
    """
    @param fname = filename
    Returns = list of sentences, where each sentence is a list of POS tagged words
    """
    f = open(fname, "r")
    taggedSents = []
    s = f.read()
    s = unicode(s, errors="ignore")
    sentences = nltk.sent_tokenize(s)
    for i in sentences:
        taggedSents.append(nltk.pos_tag(nltk.word_tokenize(i)))
    return taggedSents


def featureset_df(taggedSents, value):
    """
    This function returns a dataframe consisting of the imp_cols and target column after
    removing the NaN values
    @param taggedSents = list of sentences, where each sentence is a list of POS tagged words
    @param value = Value given to the target column in the dataframe
    Returns = dataframe with columns as imp_cols and target with value @param value
    """
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
    global imp_cols
    #imp_cols = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    df = pd.DataFrame(sents1)
    df.fillna(0, inplace=True)
    features = df[imp_cols]
    features["target"] = value
    return features


# ## Important Columns
#  * VB : Verb
#  * VBD : Past Tense Verb
#  * VBG : Gerund
#  * VBN : Past Participle
#  * VBP : Present Tense not 3rd person singular
#  * VBZ : Present Tense 3rd person singular

imp_cols = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]


taggedSents1 = tokenize(
    "/home/shirish/BTECHSEM2/project/books/stories/new_complete_stories.txt")
taggedSents0 = tokenize(
    "/home/shirish/BTECHSEM2/project/books/annotated_books/stories_without_anecdotes/new_all_stories.txt2")



taggedSents0[0]


features1 = featureset_df(taggedSents1, 1)
features0 = featureset_df(taggedSents0, 0)
features = pd.concat([features1, features0])


features = shuffle(features)


features.head(10)


# # Visualization

plt.hist(features[imp_cols], bins=2, label=imp_cols)
plt.legend()
plt.title("Frequency of occurrence of 1 or 0 in Columns")
plt.xlabel("Columns")
plt.ylabel("Count")
plt.show()


zeros=[]
ones=[]
for i in features.columns:
    series1 = features[i].value_counts()
    zeros.append(series1.loc[0])
    ones.append(series1.loc[1])
for i in zip(features.columns, zeros):
    print "  ",i
print "-------------------"
for i in zip(features.columns, ones):
    print "  ",i


x00=[]
x01=[]
x10=[]
x11=[]
ones=[]
for i in imp_cols:
    series1 = features.ix[features["target"] == 0, i].value_counts()
    series2 = features.ix[features["target"] == 1, i].value_counts()
    x00.append(series1.loc[0])
    x01.append(series1.loc[1])
    x10.append(series2.loc[0])
    x11.append(series2.loc[1])


plt.bar([i for i in range(0, len(imp_cols) * 2, 2)], x00, label='Zeros')
plt.bar([i for i in range(1, len(imp_cols) * 2 + 1, 2)], x01, label='Ones')
plt.legend()
plt.xlabel(imp_cols)
plt.ylabel("Count")
plt.title("Non Anecdotal Sentences")


plt.bar([i for i in range(0, len(imp_cols) * 2, 2)], x10, label='Zeros')
plt.bar([i for i in range(1, len(imp_cols) * 2 + 1, 2)], x11, label='Ones')
plt.legend()
plt.xlabel(imp_cols)
plt.ylabel("Count")
plt.title("Anecdotal Sentences")


X_train, X_test, y_train, y_test = train_test_split(features[imp_cols].values, features.target.values, test_size=0.10)


len(X_train)


len(X_test)


1285.0 / (1285 + 143)


model = RandomForestClassifier(n_estimators=150)
model = model.fit(X_train, y_train)
model.score(X_test, y_test)


from sklearn.svm import SVC
svc = SVC(kernel='rbf')


svc = svc.fit(X_train, y_train)


svc.score(X_test, y_test)


predicted = svc.predict(X_test)


confusion_matrix(predicted, y_test)


import numpy as np


svc_max_score = 0
svc_min_score = 1
random_forest_min = 1
random_forest_max = 0
svc_list=[]
random_forest_list=[]
def print_scores(name1, min1, max1, avg1):
    print "Model =", name1
    print "min_score =", min1, "max_score =", max1
    print "Average =", avg1
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(features[imp_cols].values, features.target.values, test_size=0.25)    
    svc = svc.fit(X_train, y_train)
    model = model.fit(X_train, y_train)
    random_forest_score = model.score(X_test, y_test)
    svc_score = svc.score(X_test, y_test)
    svc_max_score = max(max_score, svc_score)
    svc_min_score = min(min_score, svc_score)
    random_forest_min = min(random_forest_min, random_forest_score)
    random_forest_max = max(random_forest_max, random_forest_score)
    svc_list.append(svc_score)
    random_forest_list.append(random_forest_score)
print_scores("SVM", svc_min_score,svc_max_score, np.mean(svc_list))
print_scores("Random Forests", random_forest_min, random_forest_max, np.mean(random_forest_list))


print np.std(svc_list)
print np.std(random_forest_list)

