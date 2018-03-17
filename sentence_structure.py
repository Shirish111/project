import nltk
import pandas as pd
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.utils import shuffle


def tokenize(fname):
    """
    @param fname = filename
    Returns = list of sentences, where each sentence is a list of POS
              tagged words
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
    This function returns a dataframe consisting of the imp_cols and
    target column after removing the NaN values
    @param taggedSents = list of sentences, where each sentence is a
                         list of POS tagged words
    @param value = Value given to the target column in the dataframe
    Returns = dataframe with columns as imp_cols and target with value
              @param value
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
    imp_cols = ["N", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    df = pd.DataFrame(sents1)
    df.fillna(0, inplace=True)
    features = df[imp_cols]
    features["target"] = value
    return features


taggedSents1 = tokenize(
    "/home/shirish/BTECHSEM2/project/books/stories/new_complete_stories.txt")
taggedSents0 = tokenize(
    """/home/shirish/BTECHSEM2/project/books/annotated_books/
    stories_without_anecdotes/new_all_stories.txt1""")

imp_cols = ["N", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

features1 = featureset_df(taggedSents1, 1)
features0 = featureset_df(taggedSents0, 0)
features = pd.concat([features1, features0])

features = shuffle(features)

X_train, X_test, y_train, y_test = train_test_split(
    features[imp_cols].values, features.target.values)

svc = SVC(kernel='rbf')

svc = svc.fit(X_train, y_train)

print "Score = ", cross_val_score(svc, X_test, y_test).mean()

predicted = svc.predict(X_test)

print "Confusion Matrix:"
print confusion_matrix(predicted, y_test)
