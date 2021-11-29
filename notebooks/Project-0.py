# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 11:37:40 2021

@author: DZG5547@PSU.EDU
"""

### IMPORT LIBRARIES ###
import numpy as np  #Fundamental math library
import scipy  #Science, engineering and statistical routines
import pandas as pd  #Relational data tables - dataframes
import matplotlib as plt  #Core visualization library
import seaborn as sns  #Statistical visualizations. Pandas df as input
import statsmodels as sm  #Statistics functions and statistical testing
import sklearn  #machine learning

### LOAD, EXPLORE, VISUALIZE DATA ###
df = pd.read_csv("5-Minute Crafts.csv")

df.head()
df.describe()
df.columns
df.shape  #(4904,15)

### SELECT, CLEAN, PREPROCESS, AND TRANSFORM ###

# Missing values
df.isnull().values.any()
df.isnull().values.sum()  # none to drop or impute

# Define KPIs with measurable characteristics
df['view_rate'] = df['total_views'] / df['active_since_days']

cols_perf = [
    'active_since_days', 'duration_seconds', 'total_views', 'view_rate'
]
df[cols_perf].describe()

# Look at data
sns.boxplot(data=df[cols_perf[3]])  # use [0 to 3] for individual plot
sns.pairplot(df[cols_perf], diag_kind='hist')

# Remove outliers
from scipy import stats

z = np.abs(stats.zscore(df[cols_perf]))

df_o = df[(z < 3).all(axis=1)]
df_o.shape  #(4708, 16)

df_ox = df[(z >= 3).all(axis=1)]
df_ox.shape  #(, 16) #not working

sns.boxplot(data=df_o[cols_perf[3]])  # use [0 to 3] for individual plot
sns.pairplot(df_o[cols_perf], diag_kind='hist')

# Transform
active_check = df_o['active_since_days'].value_counts()
active_check  # By day to 6, then 10, 15, 20 bin, then 30 by 30 to 330, then 365 to 365 to 1460
df_o['active_since_years'] = df_o['active_since_days'].floordiv(
    365)  #bin by year

duration_check = df_o['duration_seconds'].value_counts()
duration_check  # Continuous to 899, 15 minutes then by minute. - BIN BY MINUTE.
df_o['duration_minutes'] = df_o['duration_seconds'].floordiv(
    60)  #bin by minute

views_check = df_o['total_views'].value_counts()
views_check  # Continuous - OKAY. Looks lognormal

rate_check = df_o['view_rate'].value_counts()
rate_check  # Continuous - OKAY. Looks lognormal

cols_perf_ubins = [
    'active_since_years', 'duration_minutes', 'total_views', 'view_rate'
]
sns.pairplot(df_o[cols_perf_ubins], diag_kind='hist')

from sklearn.preprocessing import minmax_scale

df_o['t_active_since_years'] = minmax_scale(df_o['active_since_years'])

# from sklearn.preprocessing import StandardScaler
# df_o['t_duration_minutes'] = StandardScaler().fit_transform(df_o['duration_minutes'])

# from sklearn.preprocessing import power_transform
# out = power_transform(df_o['total_views','view_rate'], method='box-cox')
# df_o['t_total_views','t_view_rate']

# cols_perf_t = ['t_active_since_years','t_duration_minutes','t_total_views','t_view_rate']
# sns.pairplot(df_o[cols_perf_t], diag_kind='hist')

### CLUSTER PERFORMANCE ###

### TEXT ANALYSIS ###
import nltk  #tools methods to process analyze text
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.probability import FreqDist

nltk.download('punkt')
nltk.download('stopwords')
import re
import string
# import pattern #tools for web mining, info retrevial, NLP, network analysis
# import gensim #sewmantic analysis, topic modeling, similarity analysis, Googles word2vec
# import textblob #text processing, phrase extraction, classification, POS tagging, sentiment
# import spacy #newer, industrial-strength NLP - focus on performance

# Change to lowercase
df_o['title'] = df_o['title'].map(lambda x: x.lower())

# Remove numbers
df_o['title'] = df_o['title'].map(lambda x: re.sub(r'\d+', '', x))

# Remove Punctuation
df_o['title'] = df_o['title'].map(
    lambda x: x.translate(x.maketrans('', '', string.punctuation)))

# Remove white spaces
df_o['title'] = df_o['title'].map(lambda x: x.strip())

# Tokenize into words
df_o['title'] = df_o['title'].map(lambda x: word_tokenize(x))

# Remove non alphabetic tokens
df_o['title'] = df_o['title'].map(
    lambda x: [word for word in x if word.isalpha()])

# filter out stop words
stop_words = set(stopwords.words('english'))
df_o['title'] = df_o['title'].map(
    lambda x: [w for w in x if not w in stop_words])

# Association rule bianary table
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(df_o['title']).transform(df_o['title'])
df_w = pd.DataFrame(te_ary, columns=te.columns_)
df_w

association_rules = apriori(df_w, min_support=0.01, use_colnames=True)

# Word Lemmatization
#lem = WordNetLemmatizer()
#df_o['title'] = df_o['title'].map(lambda x: [lem.lemmatize(word,"v") for word in x])

# Turn lists back to string
#df_o['title'] = df_o['title'].map(lambda x: ' '.join(x))

# Tokenization
# Tagging
# Chunking
# Stemming
# Lemmatization

# Text classification
# Text summarization
# Text clustering

# Sentiment analysis
# Entity extraction and recognition
# Similarity analysis and relation modeling
