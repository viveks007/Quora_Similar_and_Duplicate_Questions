#!/usr/bin/env python
# coding: utf-8

# The goal of this competition is to predict which of the provided pairs of questions contain two questions with the same meaning. The ground truth is the set of labels that have been supplied by human experts. The ground truth labels are inherently subjective, as the true meaning of sentences can never be known with certainty. Human labeling is also a 'noisy' process, and reasonable people will disagree. As a result, the ground truth labels on this dataset should be taken to be 'informed' but not 100% accurate, and may include incorrect labeling. We believe the labels, on the whole, to represent a reasonable consensus, but this may often not be true on a case by case basis for individual items in the dataset.
# Data fields
# id - the id of a training set question pair
# qid1, qid2 - unique ids of each question (only available in train.csv)
# question1, question2 - the full text of each question
# is_duplicate - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


from bs4 import BeautifulSoup
import re
import nltk


# In[7]:


data=pd.read_csv(r"D:\PC\Data Science\Machine Learning Project\Quora\train.csv")


# In[8]:


df= data.sample(50000, random_state=2)


# In[9]:


df.shape


# In[10]:


df.head()


# In[11]:


df.info()


# In[12]:


#missing value
df.isnull().sum()


# In[13]:


#completelty duplicate rows
df.duplicated().sum()


# In[14]:


#distribution of duplicate and non-duplicate questions
 
print(df['is_duplicate'].value_counts())
print((df['is_duplicate'].value_counts()/df['is_duplicate'].count())*100)
df['is_duplicate'].value_counts().plot(kind='bar')


# In[15]:


#Repeated questions

qid= pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
print('Number of unique questions', np.unique(qid).shape[0])
x=qid.value_counts()>1
print('Number of questions getting repeated', x[x].shape[0])


# In[16]:


#repeated questions histogram
plt.hist(qid.value_counts().values, bins=160)
plt.yscale('log')
plt.show()


# In[17]:


def preprocess(q):
    q= str(q).lower().strip()
    
    #replace certain special characters with their string equivalent
    q=q.replace('%','percent')
    q=q.replace('$','dollar')
    q=q.replace('₹','rupee')
    q=q.replace('€','euro')
    q=q.replace('@','at')
    
    #The pattern '[math]' appaers 
    q=q.replace('[math]','')
    
    #Replace whole numbers with string equivalent
    q=q.replace(',000,000,000','b')
    q=q.replace(',000,000','m')
    q=q.replace(',000','k')
    q=re.sub(r'([0-9]+)000000000',r'\1b',q)
    q=re.sub(r'([0-9]+)000000',r'\1m',q)
    q=re.sub(r'([0-9]+)000',r'\1k',q)
    
    #Decontracting words
    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }
    
    q_decontracted=[]
    
    for word in q.split():
        if word in contractions:
            word=contractions[word]
        q_decontracted.append(word)
        
    q=' '.join(q_decontracted)
    q=q.replace("'ve", "have")
    q=q.replace("n't","not")
    q=q.replace("'re'","are")
    q=q.replace("'ll'","will")
    
    #removing html tags
    q=BeautifulSoup(q)
    q=q.get_text()
    
    #remove puntuations
    pattern=re.compile('\W')
    q=re.sub(pattern, ' ', q).strip()
    return q


# In[18]:


preprocess("I've already")


# In[19]:


#Apply preprocess in question 1 & 2
df['question1']=df['question1'].apply(preprocess)
df['question2']=df['question2'].apply(preprocess)


# In[31]:


df.head()


# #feature engineering

# In[20]:


#length of sentences
df['q1_len']=df['question1'].str.len()
df['q2_len']=df['question2'].str.len()
df.head()


# In[21]:


#split the sentences
df['q1_num_words']=df['question1'].apply(lambda row: len(row.split(" ")))
df['q2_num_words']=df['question2'].apply(lambda row: len(row.split(" ")))
df.head()


# In[22]:


#common words between sentences

def common_words(row):
    w1=set(map(lambda word: word.lower().strip(),row['question1'].split(" ")))
    w2=set(map(lambda word: word.lower().strip(),row['question2'].split(" ")))
    return len(w1 & w2)


# In[23]:


df['word_common']=df.apply(common_words, axis=1)
df.head()


# In[24]:


# total words in both sentences
def total_words(row):
    w1=set(map(lambda word:word.lower().strip(), row['question1'].split(" ")))
    w2=set(map(lambda word:word.lower().strip(), row['question2'].split(" ")))
    return (len(w1)+len(w2))


# In[26]:


df['word_total1']=df.apply(total_words, axis=1)
df.head()


# In[27]:


#add share of words common words/ total words
df['word_share']= round(df['word_common']/df['word_total1'],2)
df.head()


# #advance features

# In[28]:


from nltk.corpus import stopwords


# In[31]:


nltk.download('stopwords')


# In[32]:


def fetch_token_features(row):
    q1=row['question1']
    q2=row['question2']
    
    #incase len is 0 then in division ,it become infinity
    SAFE_DIV= 0.0001
    
    STOP_WORDS=stopwords.words("english")
    
    #since i have to extract 8 features , so made list contains 8 zeroes
    token_features=[0.0]*8
    
    #coverting sentences into tokens
    q1_tokens=q1.split()
    q2_tokens=q2.split()
    
    if len(q1_tokens)==0 or len(q2_tokens)==0:
        return token_features
    
    #get the non-stopwords in questions
    q1_words=set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words=set([word for word in q2_tokens if word not in STOP_WORDS])
    
    #get the stopwords
    q1_stops=set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops=set([word for word in q2_tokens if word in STOP_WORDS])
    
    #get common non-stopwords from question pairs
    common_word_count=len(q1_words.intersection(q2_words))
    
    #get common stopwords from question pairs
    common_stop_count=len(q1_stops.intersection(q2_stops))
    
    #get common tokens from question pairs
    common_token_count=len(set(q1_tokens).intersection(set(q2_tokens)))
    
    token_features[0]=common_word_count/(min(len(q1_words),len(q2_words))+SAFE_DIV)
    token_features[1]=common_word_count/(max(len(q1_words),len(q2_words))+SAFE_DIV)
    token_features[2]=common_stop_count/(min(len(q1_words),len(q2_words))+SAFE_DIV)
    token_features[3]=common_stop_count/(max(len(q1_words),len(q2_words))+SAFE_DIV)
    token_features[4]=common_token_count/(min(len(q1_words),len(q2_words))+SAFE_DIV)
    token_features[5]=common_token_count/(max(len(q1_words),len(q2_words))+SAFE_DIV)
    
    #last word of both question is same or not
    token_features[6]=int(q1_tokens[-1]==q2_tokens[-1])
    
    #first word of both question is same or not
    token_features[7]=int(q1_tokens[0]==q2_tokens[0])
    
    return token_features


# In[33]:


#apply in data
token_features=df.apply(fetch_token_features, axis=1)


# In[34]:


df["cwc_min"]= list(map(lambda x: x[0], token_features))
df["cwc_max"]= list(map(lambda x: x[1], token_features))
df["csc_min"]= list(map(lambda x: x[2], token_features))
df["csc_max"]= list(map(lambda x: x[3], token_features))
df["ctc_min"]= list(map(lambda x: x[4], token_features))
df["ctc_max"]= list(map(lambda x: x[5], token_features))
df["last_word_eq"]= list(map(lambda x: x[6], token_features))
df["first_word_eq"]= list(map(lambda x: x[7], token_features))

df.head()


# In[36]:


#length based features

def fetch_length_features(row):
    q1=row['question1']
    q2=row['question2']
    
    length_features=[0.0]*3
    
    #converting sentences into tokens
    q1_tokens= q1.split()
    q2_tokens=q2.split()
    
    if len(q1_tokens)==0 or len(q2_tokens)==0:
        return length_features
    
    #absolute length features
    length_features[0]=abs(len(q1_tokens)-len(q2_tokens))
    
    #average token length
    length_features[1]=(len(q1_tokens)+len(q2_tokens))/2
    
    return length_features


# In[37]:


length_features=df.apply(fetch_length_features, axis=1)

df['abs_len_diff']=list(map(lambda x:x[0], length_features))
df['mean_len']=list(map(lambda x:x[1], length_features))


# In[38]:


df.head()


# In[40]:


sns.pairplot(df[['ctc_min', 'cwc_min', 'csc_min', 'is_duplicate']], hue='is_duplicate')


# In[41]:


sns.pairplot(df[['ctc_max', 'cwc_max', 'csc_max', 'is_duplicate']], hue='is_duplicate')


# In[42]:


sns.pairplot(df[['last_word_eq', 'first_word_eq', 'is_duplicate']], hue='is_duplicate')


# In[44]:


sns.pairplot(df[['abs_len_diff','mean_len','is_duplicate']], hue='is_duplicate')


# In[ ]:





# In[45]:


ques_df=df[['question1','question2']]
ques_df.head()


# In[55]:


final_df=df.drop(columns=['id', 'qid1', 'qid2', 'question1', 'question2'])
print(final_df.shape)
final_df.head()


# In[48]:


from sklearn.feature_extraction.text import CountVectorizer


# In[50]:


#merge text

questions= list(ques_df['question1'])+list(ques_df['question2'])

cv=CountVectorizer(max_features=3000)
q1_arr, q2_arr=np.vsplit(cv.fit_transform(questions).toarray(),2)


# In[52]:


temp_df1=pd.DataFrame(q1_arr, index=ques_df.index)
temp_df2=pd.DataFrame(q2_arr, index=ques_df.index)
temp_df= pd.concat([temp_df1, temp_df2], axis=1)
temp_df.shape


# In[56]:


final_df=pd.concat([final_df,temp_df], axis=1)
print(final_df.shape)
final_df.head()


# #build model

# In[59]:


#split the data into train & test

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(final_df.iloc[:,1:].values, final_df.iloc[:,0].values, test_size=0.2, random_state=42, stratify=final_df.iloc[:,0].values)


# In[68]:


#call random forest classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
rf=RandomForestClassifier()


# In[69]:


#Fit the model & get accuracy
rf.fit(X_train, y_train)
y_pred=rf.predict(X_test)
print(confusion_matrix(y_test,y_pred))
accuracy_score(y_test, y_pred)


# In[63]:


#Call XGBoost classifier
from xgboost import XGBClassifier
xgb=XGBClassifier()


# In[71]:


#fit the model & get accuracy
xgb.fit(X_train, y_train)
y_pred=xgb.predict(X_test)
print(confusion_matrix(y_test,y_pred))
accuracy_score(y_test, y_pred)


# To increase more accuracy

# In[ ]:


#incease data- use google collab, use incremental learning
#apply preprocessing step like steaming
#apply more algorithms SVM, perceprton
#Hyperparameter tunning, Cross validation
#research create more features
#on behalf Bag of word use TDIDF, WORD2VEC, TFIDF weighted W2V

