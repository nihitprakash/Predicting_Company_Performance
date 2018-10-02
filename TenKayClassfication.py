import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from textblob import TextBlob
from nltk import stem
from sklearn.metrics import confusion_matrix
from __future__ import division
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from nltk import word_tokenize, FreqDist,classify, ConditionalFreqDist, pos_tag
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
import sklearn.metrics
import math
from sklearn.ensemble import RandomForestClassifier

# set wd
#wd = 'C:/Users/NihitPrakash/Documents/UCBerkeley/242-Data Analytics'
file_name = 'mda_extracted_255040.csv'

# read file 
dataset = pd.read_csv(file_name,sep=',')
dataset.describe()
dataset.head(8)
features = list(dataset.columns.values)

#==============================================================================
# Dataset desciption & baseline
#==============================================================================
M_baseline = confusion_matrix(dataset['Sentiment'], dataset['Sentiment'])
# 156: negative sentiment
# 163: positive sentiment
# quite balanced

TP_baseline = M_baseline[1,1]
# 'a' as accuracy - notation kept throughout the code
a_baseline = TP_baseline / np.trace(M_baseline)


#==============================================================================
# MDA's description
#==============================================================================
X = dataset['mda']
len(X)
num_letters = []
for i in range(len(X)):
    num_letters.append(len(X[i]))
#The average length of English words is 4.5 letters / 20 words per sentence
AVE_L = 4.5
AVE_W = 20
num_words_eval = [round(num_letters[i]/AVE_L,2) for i in range(len(num_letters))]
words_mean = round(sum (num_words_eval)/len(num_words_eval),1)
sentences_mean = round(words_mean/AVE_W,1)

#==============================================================================
# Functions
#==============================================================================

# Function to stem words     
def stemming(x):
    stemmer = stem.SnowballStemmer("english")
    words=x.split()
    doc=[]
    for word in words:
        word=stemmer.stem(word)
        doc.append(word)
    return " ".join(doc)
    
# Stemming the mda
mda_s = map (stemming , dataset['mda']) 
#for i in range(50):
#    print cmp(mda_s[i],dataset['mda'][i])
    
dataset['MDA_Text'] = mda_s

#function to export dataset to a csv file
#def export(col1, col2, col3, col4, outputname,dataset):
 #   header = [col1, col2, col3, col4]
  #  dataset.to_csv(outputname, columns = header)

#function to change the order of the columns
def order(frame,var):
    varlist =[w for w in frame.columns if w not in var]
    frame = frame[var+varlist]
    return frame

dataset = order(dataset,['ticker', 'date', 'Year'])
dataset.head(2)
    
#Function to count positive and negative words
def count_dic_words(lmcdic,dataset):
    #Modifying the Dictionary  
    lmcdic=lmcdic[['Word','Positive','Negative']]
    lmcdic['Sum']=lmcdic['Positive']+lmcdic['Negative']
    #remove rows where both +ve and -ve are not defined
	lmcdic=lmcdic[lmcdic.Sum != 0]
    lmcdic=lmcdic.drop(['Sum'],axis=1)
    #assign 1 to positive and -1 to negative
	lmcdic.loc[lmcdic['Positive']>0, 'Positive'] = 1
    lmcdic.loc[lmcdic['Negative']>0, 'Negative'] = -1
	#convert words to lowercase
	lmcdic['Word']=lmcdic['Word'].str.lower()
    #Counting the words in the MDA
    tf = CountVectorizer(analyzer='word', min_df = 0, stop_words = 'english')
    tfidf_matrix =  tf.fit_transform(dataset['MDA_Text'].values)
    feature_names = tf.get_feature_names() 
    tfidf_array = tfidf_matrix.toarray()
    tfidf_df = pd.DataFrame(tfidf_array)
    tfidf_df.columns = [i.lower() for i in feature_names] 
    tfidf_df = tfidf_df.T 
    tfidf_df['Word']=tfidf_df.index
    #Merging the results
    result_df = pd.merge(tfidf_df, lmcdic, how='inner',left_on='Word',right_on='Word')
    col_list=list(result_df)
    result_df_pos=result_df[result_df.Positive==1]
    result_df_neg=result_df[result_df.Negative==-1]
    result_df[col_list[0:len(dataset)]].sum(axis=0)
    #Counting the positive and negative words in a financial context per document
    pos_words_sum=result_df_pos[col_list[0:len(dataset)]].sum(axis=0)
    neg_words_sum=result_df_neg[col_list[0:len(dataset)]].sum(axis=0)
    #Adding new features to the master dataframe
    dataset['Tot_pos']=pos_words_sum.values
    dataset['Tot_neg']=neg_words_sum.values
    return dataset

# Function to add features to the train and test set based on vectorizer
def vect_features(vectorizer,train,test):
    features_train_transformed = vectorizer.fit_transform(train['MDA_Text'].values)
    feature_names = vectorizer.get_feature_names()
    features_train_transformed = features_train_transformed.toarray()
    train_data = pd.DataFrame(features_train_transformed)
    train_data.columns = feature_names
    train=pd.concat([train,train_data],axis=1)
    features_test_transformed = vectorizer.transform(test['MDA_Text'].values)
    features_test_transformed = features_test_transformed.toarray()
    test_data = pd.DataFrame(features_test_transformed)
    test_data.columns = feature_names
    test=pd.concat([test,test_data],axis=1)
    return train,test

#Function to create Classification Report   
def report(test,predictions):
    pd.crosstab(test['Sentiment'], predictions, rownames=['Actual'], colnames=['Predicted'], margins=True)
    a=accuracy_score(test['Sentiment'],predictions) # global accuracy t/total
    p=precision_score(test['Sentiment'],predictions, pos_label = "pos") # tp / (tp + fp)
    r=recall_score(test['Sentiment'].values,predictions, pos_label = "pos") #  tp / (tp + fn)
    f=f1_score(test['Sentiment'].values,predictions, pos_label = "pos") #  2 * (precision * recall) / (precision + recall)
    print "Accuracy = ",a,"\nPrecision =",p,"\nRecall = ",r,"\nF-Score = ",f 

#Function to define the model	
def model(classifier,train,test,column):
    targets = train['Sentiment'].values
    train_data=train.values
    predictors = train_data[0:,column:]
    classifier.fit(predictors,targets)
    test_data=test.values
    predictions=classifier.predict(test_data[0:,column:])
    report(test,predictions)
    return predictions

#==============================================================================
# Model 0: blob
#==============================================================================

blob = [TextBlob(i).sentiment[0] for i in X]
def binar (x): 
    if x>=0:
        return 1
    else:
        return 0
blob = map (binar, blob)
# blob: very simple positivity score
dataset['blob'] = blob
#print dataset.head(5)

#M_blob = confusion_matrix(dataset['Sentiment'], dataset['blob'])
#report (dataset['Sentiment'], dataset['blob'])
# accuracy 0.486: 

#==============================================================================
# Financial Dictionary
#==============================================================================
#for getting total positive and negative words
lmcdic = pd.read_excel("LoughranMcDonald_MasterDictionary_2014.xlsx") 

dataset=count_dic_words(lmcdic,dataset)

#for exporting to a csv file
#export("ticker", "date", "Tot_pos", "Tot_neg", "output.csv", dataset)

#==============================================================================
# Splitting the data
#==============================================================================
SPLIT_FACTOR = 0.30
train, test = train_test_split(dataset, test_size = SPLIT_FACTOR)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

#==============================================================================
# Model 1: 
#==============================================================================
#Model 1 - Baseline Model
#Algorithm: Bernoulli Naive Bayes 
print "Model 1"
vectorizer_1 = CountVectorizer(stop_words='english')
train_1,test_1 = vect_features(vectorizer_1,train,test)
classifier_1 = BernoulliNB(fit_prior=False)
predictions_1 = model(classifier_1,train_1,test_1,10)

#Model 2
#Algorithm: Multinomial Naive Bayes with with BiGrams
print "Model 2"
vectorizer_2 = CountVectorizer(stop_words='english',ngram_range=(1,2),min_df=5,max_df=0.8)
train_2,test_2 = vect_features(vectorizer_2,train,test)
classifier = MultinomialNB(fit_prior=False)
predictions = model(classifier,train_2,test_2,9)

#Model 3
#Algorithm: Multinomial Naive Bayes top 50 words 
print "Model 3"
vectorizer_3 = CountVectorizer(stop_words='english', max_features=50)
train_3,test_3 = vect_features(vectorizer_3,train,test)
classifier = MultinomialNB(fit_prior=False)
predictions = model(classifier,train_3,test_3,9)

#Model 4
#Algorithm: Multinomial Naive Bayes with BiGrams and top 50 words
print "Model 4"
vectorizer_4 = CountVectorizer(stop_words='english',max_features=100,ngram_range=(1,2),min_df=5,max_df=0.8)
train_4,test_4 = vect_features(vectorizer_4,train,test)
classifier = MultinomialNB(fit_prior=False)
predictions = model(classifier,train_4,test_4,9)

#Model 5
#Algorithm: Bernoulli Naive Bayes with BiGrams and top 50 words
print "Model 5"
vectorizer_5 = CountVectorizer(stop_words='english',max_features=100,ngram_range=(1,2),min_df=5,max_df=0.8)
train_5,test_5 = vect_features(vectorizer_5,train,test)
classifier = BernoulliNB(fit_prior=False)
predictions = model(classifier,train_5,test_5,10)

#Model 6
#Algorithm: Multinomial Naive Bayes with only TriGrams and top 50 words
print "Model 6"
vectorizer_6 = CountVectorizer(stop_words='english',max_features=50,ngram_range=(3,3),min_df=5,max_df=0.8)
train_6,test_6 = vect_features(vectorizer_6,train,test)
classifier = MultinomialNB(fit_prior=False)
predictions = model(classifier,train_6,test_6,9)

#Model 7
#Algorithm: Bernoulli Naive Bayes with BiGrams and top 50 words
print "Model 7"
vectorizer_7 = CountVectorizer(stop_words='english',max_features=50,ngram_range=(3,3),min_df=5,max_df=0.8)
train_7,test_7 = vect_features(vectorizer_7,train,test)
classifier = BernoulliNB(fit_prior=False)
predictions = model(classifier,train_7,test_7,9)

#Model 8
#Algorithm: Multinomial Naive Bayes
#Features: TfidfVectorizer of only top 50 words and 2-grams
print "Model 8"
vectorizer_8 = TfidfVectorizer(sublinear_tf=True,stop_words='english', max_features=100)
train_8,test_8 = vect_features(vectorizer_8,train,test)
test_8.head(2)
frames = [train_8, test_8]
result = pd.concat(frames)
result.head(2)
result = order(result,['mda', 'MDA_Text', 'ticker'])
result_new = result.ix[:,'ticker':]
result_new.head(2)
result_new.to_csv("output_fuller.csv")
classifier = MultinomialNB(fit_prior=False)
predictions = model(classifier,train_8,test_8,10)

#Model 9
#Algorithm: Uncalibrated Random Forest
#Features: TfidfVectorizer of only top 50 words and 2-grams, total positive words, total negative words, polarity score 
print "Model 9"
vectorizer_9 = TfidfVectorizer(sublinear_tf=True,stop_words='english',max_features=500,ngram_range=(1,2),min_df=5,max_df=0.8)
train_9,test_9 = vect_features(vectorizer_9,train,test)
classifier = RandomForestClassifier(fit_prior=False)
predictions = model(classifier,train_9,test_9,10)

