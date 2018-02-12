
# coding: utf-8

# In[106]:


import nltk
import pandas as pd


# In[107]:


fname = "/home/shirish/BTECHSEM2/project/books/stories/new_complete_stories.txt"
f = open(fname, "r")
sentences = nltk.sent_tokenize(f.read().encode('utf8'))
taggedSents = [(nltk.pos_tag(nltk.word_tokenize(i))) for i in sentences]


# In[108]:


taggedSents[1]


# In[109]:


sents1 = []


# In[110]:


s = "Shirish"


# In[111]:


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


# In[112]:


df = pd.DataFrame(sents1)


# In[113]:


df.columns


# In[114]:


imp_cols=["N", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]


# In[115]:


df[imp_cols]


# In[116]:


# Fill all NaN values with 0
df.fillna(0, inplace=True)


# In[117]:


features = df[imp_cols]


# In[118]:


features["target"] = 1


# In[119]:


features.head()


# In[120]:


features.shape


# # Example of using this feature set for ML
# # Decision Tree Classifier

# In[121]:


from sklearn import tree


# In[122]:


model = tree.DecisionTreeClassifier()


# In[123]:


model = model.fit(features[imp_cols].values, features["target"].values)

