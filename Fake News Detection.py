#!/usr/bin/env python
# coding: utf-8

# FAKE NEWS DETECTION
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


real_df=pd.read_csv("/content/drive/MyDrive/Fake News.zip (Unzipped Files)/Fake News/True.csv")
fake_df=pd.read_csv("/content/drive/MyDrive/Fake News.zip (Unzipped Files)/Fake News/Fake.csv")


# In[ ]:


real_df.head()


# In[ ]:


fake_df.head()


# In[ ]:


real_df.columns,fake_df.columns


# In[ ]:


real_df.shape,fake_df.shape


# In[ ]:


real_df.info()
print('\n')
fake_df.info()


# In[ ]:


real_df.describe()


# In[ ]:


fake_df.describe()


# In[ ]:


real_df['subject'].unique()


# In[ ]:


fake_df['subject'].unique()


# In[ ]:


real_words = " ".join([x for x in real_df['title']])
wordcloud1 = WordCloud(width=500, height=500, random_state=40).generate(real_words)


plt.figure(figsize=(20,8))
plt.imshow(wordcloud1,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:


#Visualization of most frequent words
fake_words = " ".join([x for x in fake_df['title']])
wordcloud1 = WordCloud(width=500, height=500, random_state=40).generate(fake_words)


plt.figure(figsize=(20,8))
plt.imshow(wordcloud1,interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:


real_df['Target']=1
fake_df['Target']=0


# In[ ]:


df=pd.concat([real_df,fake_df])
df.head()


# In[ ]:


df.tail()


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.plot()


# In[ ]:


df['Text_len']=df['text'].apply(len)
df['Title_len']=df['title'].apply(len)


# In[ ]:


df.info()


# In[ ]:


df.groupby('Target').describe()


# In[ ]:


df.groupby('Target').median()


# In[ ]:


df.groupby('Target').boxplot()


# ### Insights
# - Target=0 is fake data Target=1 is real data.
# - Average length of Titles of real data is 64.66 and fake data is 94.19 .
# - Length of title fake data is more than that of real data.

# In[ ]:


df.drop(columns=['date', 'subject'])


# ### Vectorization
# - The process of converting words or text into numbers or vectors are called Text-Vectorization.
#    - can be done using :-
#        - 1.CountVectorizer()
#        - 2.TfidfVectorizer()

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[ ]:


X = df['text']
Y = df['Target']


# In[ ]:


X = cv.fit_transform(X)


# ### Train-Test split of data
# - Splitting dataset into Training and Testing Datasets.
#     - Training as 70%
#     - Testing as 30%

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 101)


# ### a. Naive Bayes

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
nb = MultinomialNB()
nb.fit(X_train, Y_train)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


nb_prediction = nb.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test,nb_prediction)


# In[ ]:


conf_mat = confusion_matrix(Y_test, nb_prediction)
conf_mat


# #### ~ Accuracy is 95% for Naive Bayes.

# ### b.Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model1=LogisticRegression(solver='lbfgs', max_iter=1000)
model1.fit(X_train,Y_train)


# In[ ]:


prediction=model1.predict(X_test)


# In[ ]:


accuracy_score(Y_test,prediction)


# In[ ]:


conf_mat = confusion_matrix(Y_test, prediction)
conf_mat


# In[ ]:


prediction=model1.predict(X_test)


# #### ~ Accuracy is 99.53% for Logistic Regression.

# #### c. Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
ref=RandomForestClassifier(n_estimators=135,
    criterion='gini',
    max_depth=None,
    max_features='auto',
    max_leaf_nodes=None,
    bootstrap=True,
    n_jobs=None,
    random_state=25,
)


# In[ ]:


ref.fit(X_train,Y_train)


# In[ ]:


ref_pred = ref.predict(X_test)


# In[ ]:


accuracy_score(Y_test, ref_pred)


# In[ ]:


conf_mat = confusion_matrix(Y_test, ref_pred)
conf_mat


# #### ~ Accuracy is 98.74% for Random Forest.

# In[ ]:


plot_y=[nb.score(X_test,Y_test),ref.score(X_test,Y_test),accuracy_score(Y_test,prediction)]


# In[ ]:


plt.figure(figsize=(8,5))
plt.bar(['Naive','RandomForest','LogisticRegression'],plot_y,width=0.3)
plt.xlabel('Models',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.show()


# ### Result
#     - Logistic Regression gives best results for this.
#          ~  Accuracy=99.53%
