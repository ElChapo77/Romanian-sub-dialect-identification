import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# reading the dataset for training and tasting
x_train = pd.read_csv('train_samples.txt', sep='\t', names=['id', 'text'])
y_train = pd.read_csv('train_labels.txt', sep='\t', names=['id', 'M/R'])
x_validation = pd.read_csv('validation_samples.txt', sep='\t', names=['id', 'text'])
y_validation = pd.read_csv('validation_labels.txt', sep='\t', names=['id', 'M/R'])

x_train = pd.concat([x_train, x_validation], ignore_index=True)
y_train = pd.concat([y_train, y_validation], ignore_index=True)

x_train.drop('id', axis='columns', inplace=True)
y_train.drop('id', axis='columns', inplace=True)
x_train = x_train.values.ravel()
y_train = y_train.values.ravel()

x_test = pd.read_csv('test_samples.txt', sep='\t', names=['id', 'text'])
df_to_write = pd.DataFrame(x_test)
df_to_write.drop('text', axis='columns', inplace=True)

x_test.drop('id', axis='columns', inplace=True)
x_test = x_test.values.ravel()

# creating the features that the model will use
v = TfidfVectorizer(ngram_range=(1,2), max_df=0.7)
x_train_count = v.fit_transform(x_train)
x_test_count = v.transform(x_test)

# creating the model, training it and predicting the result
classifier = MultinomialNB(alpha=0.09)
classifier.fit(x_train_count, y_train)
y_pred = classifier.predict(x_test_count)

df_to_write['label'] = y_pred
df_to_write.to_csv('submission.csv', index=False)
