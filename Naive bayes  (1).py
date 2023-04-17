#!/usr/bin/env python
# coding: utf-8

# In[111]:


# importing necessary libraries
import pandas as pd
import numpy as np
import re
df = pd.read_csv('IMDB Dataset.csv')
df.head()


# In[112]:


df.shape


# In[113]:


# Subset
df = df.sample(1000)
# resetting index
df.reset_index(drop=True, inplace=True)
# sample dataset size
df.shape


# In[122]:


# positive:1 , negative:0
df['sentiment'].replace({'positive':1,'negative':0},inplace=True)
df.head()


# In[138]:


# functions to remove noise
# remove html tags
def clean_html(text):   
 clean=re.compile('<.*?>')
 return re.sub(clean,'',text)
# remove brackets
def remove_brackets(text):
 return re.sub('\[[^]]*\]','',text)
# lower the cases
def lower_cases(text):
 return text.lower()
# remove special characters
def remove_char(text):
 pattern=r'[^a-zA-z0-9\s]'
 text=re.sub(pattern,'',text)    
 return text
# remove noise(combine above functions)
def remove_noise(text):
 text = clean_html(text)
 text = remove_brackets(text)
 text = lower_cases(text) 
 text = remove_char(text) 
 return text
# call the function on predictors
df['review']=df['review'].apply(remove_noise)


# In[140]:


from nltk.stem.porter import PorterStemmer
def stem_words(text):
 ps = PorterStemmer()
 stem_list = [ps.stem(word) for word in text.split()] 
 text=''.join(ps.stem(word)for word in text)
 
 return text
df['review']=df['review'].apply(stem_words)


# In[176]:


# importing from nlptoolkit library
import nltk
from nltk.corpus import stopwords
# creating list of english stopwords
#stopword_list = stopwords.words(‘english’)
stopword_list=stopwords.words('english')
# removing the stopwords from review
def remove_stopwords(text):
    # list to add filtered words from review
    filtered_text = []
    # verify & append words from the text to filtered_text list
        for word in text.split():
            if word not in stopword_list:
                filtered_text.append(word)
    # add content from filtered_text list to new variable
        clean_review = filtered_text[:]
    # emptying the filtered_text list for new review
        filtered_text.clear()
        return clean_review
df['review']=df['review'].apply(remove_stopwords)
df['review']
# join back all words as single paragraph
def join_back(text):
    return ' '.join(text)
df['review'] = df['review'].apply(join_back)
# check if changes are applied
df.head()


# In[157]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=800)
# vectorizing words and storing in variable X(predictor)
X=cv.fit_transform(df['review']).toarray()
# predictor
X
# X size
X.shape
output: (1000, 800)
# target
y = df.iloc[:,-1].values
# y size
y.shape


# In[149]:


# train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[151]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score
# Naive Bayes Classifiers
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
# fitting and predicting
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
mnb.fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)
bnb.fit(X_train, y_train)
y_pred_bnb = bnb.predict(X_test)
# accuracy scores
print("Gaussian", accuracy_score(y_test, y_pred_gnb))
print("Multinomial", accuracy_score(y_test, y_pred_mnb))
print("Bernoulli", accuracy_score(y_test, y_pred_bnb))


# In[ ]:




