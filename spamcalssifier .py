#!/usr/bin/env python
# coding: utf-8

# In[6]:


# importing the Dataset

import pandas as pd

messages = pd.read_csv('smsspamcollection.txt', sep='\t',
                           names=["label", "message"]) ## sep = '\t' column are sep by tap lable my two clomns with label and massage 


# In[7]:


messages.head()


# ### Data cleaning and preprocessing

# In[42]:


import re ## regular expression 
import nltk 
nltk.download('stopwords') 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords  
from nltk.stem.porter import PorterStemmer  #### for stemming 
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# In[43]:


corpus = []  ##### main informaion of data will be store in this list 

for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])  ### remove ncessecery character  
    review = review.lower()   ## convert all words in lower case 
    review = review.split()   ## split into list 
    
    review = [ lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)  ## agr yeh nhi karenge to sab list me form hongr hum saari list ko join kar rhe h by using ' '
    corpus.append(review)


# #### Creating the Bag of Words model

# In[44]:


from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer(max_features=2500) ##max_features=2500
X = cv.fit_transform(corpus).toarray()  ## as of now my X is independent data 

y=pd.get_dummies(messages['label'])  ## dummies ham ==1 , spam ==0
y=y.iloc[:,1].values  ### reduce the one column 


# In[45]:


X


# In[ ]:





# ### Train Test Split

# In[46]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)  ## 80% training data 20% teest data 0.20


# #### Training model using Naive bayes classifier
# 
# based on probability 
# 

# In[47]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)


# In[48]:


from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test,y_pred)


# In[49]:


confusion_m


# In[50]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)


# In[51]:


accuracy


# In[ ]:




