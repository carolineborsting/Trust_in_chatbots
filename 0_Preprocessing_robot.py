#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:57:13 2020

@author: Maria Abildtrup Madsen & Caroline Kjær Børsting 
"""


###--------------------------------------###
###---------------SET UP-----------------###
###--------------------------------------###

import pandas as pd
import os
import nltk
import re,string
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
ps = PorterStemmer() 

# Setting working directory 
os.chdir("/Users/mariaa.madsen/Google Drive/Human Computer Interaction/Analysis/Data")
#os.chdir("/Users/Caroline/Google Drev/Human Computer Interaction/Analysis/Data")


# Read data. 
df = pd.read_csv('data_nonhuman_13052020.csv', encoding='latin-1', sep=';')
df.head() 

###--------------------------------------###
###-----------PREPROCESSING--------------###
###--------------------------------------###


"""
def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

df['Text_chat'] = df['Text_chat'].apply(stem_sentences)

df['Text_questionnaire'] = df['Text_questionnaire'].apply(stem_sentences)
"""

# tokenize and lower all words 
df['Text_chat']=[[word.lower() for word in text.split()] for text in df['Text_chat']]  
df['Text_questionnaire']=[[word.lower() for word in text.split()] for text in df['Text_questionnaire']]  


# Remove stop words
nltk.download('stopwords')                  # Download NLTK list with stopwords
from nltk.corpus import stopwords           # Import stopwords
stop_words = stopwords.words('english')      # Make list with stopwords and set it to danish

# Remove stop words 
#df['Text_chat']=[[word for word in text if not word in stop_words] for text in df['Text_chat']] 
df['Text_questionnaire']=[[word for word in text if not word in stop_words] for text in df['Text_questionnaire']] 

# Remove non-alphabetic letters
#df['Text_chat']=[[word for word in text if word.isalpha()] for text in df['Text_chat']]  
df['Text_questionnaire']=[[word for word in text if word.isalpha()] for text in df['Text_questionnaire']]  


df['Text_chat'] = [' '.join(map(str, l)) for l in df['Text_chat']]
df['Text_questionnaire'] = [' '.join(map(str, l)) for l in df['Text_questionnaire']]

df.to_csv("nonhuman_df_cleaned.csv", sep=";")          # write csv file
