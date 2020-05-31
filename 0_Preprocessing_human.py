#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:57:13 2020

@author: mariaa.madsen
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

# Read data. 
human_df = pd.read_csv('data_human_13052020.csv', encoding='latin-1', sep=',')
human_df.head() 

###--------------------------------------###
###-----------PREPROCESSING--------------###
###--------------------------------------###

# Stemming
def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

#human_df['Text_chat'] = human_df['Text_chat'].apply(stem_sentences)
#human_df['Text_questionnaire'] = human_df['Text_questionnaire'].apply(stem_sentences)


# tokenize and lower all words 
human_df['Text_chat']=[[word.lower() for word in text.split()] for text in human_df['Text_chat']]  
human_df['Text_questionnaire']=[[word.lower() for word in text.split()] for text in human_df['Text_questionnaire']]  


# Remove stop words
nltk.download('stopwords')                  # Download NLTK list with stopwords
from nltk.corpus import stopwords           # Import stopwords
stop_words = stopwords.words('english')      # Make list with stopwords and set it to danish

# Remove stop words 
#human_df['Text_chat']=[[word for word in text if not word in stop_words] for text in human_df['Text_chat']] 
human_df['Text_questionnaire']=[[word for word in text if not word in stop_words] for text in human_df['Text_questionnaire']] 

# Remove non-alphabetic letters
human_df['Text_chat']=[[word for word in text if word.isalpha()] for text in human_df['Text_chat']]  
human_df['Text_questionnaire']=[[word for word in text if word.isalpha()] for text in human_df['Text_questionnaire']]  

human_df['Text_chat'] = [' '.join(map(str, l)) for l in human_df['Text_chat']]
human_df['Text_questionnaire'] = [' '.join(map(str, l)) for l in human_df['Text_questionnaire']]

human_df.to_csv("human_df_cleaned.csv", sep=";")          # write csv file
