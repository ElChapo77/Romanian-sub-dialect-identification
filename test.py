import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# reading the dataset for training and tasting

# train data
x_train = pd.read_csv('train_samples.txt', sep='\t', names=['id', 'text'])
y_train = pd.read_csv('train_labels.txt', sep='\t', names=['id', 'M/R'])
x_train.drop('id', axis='columns', inplace=True)
y_train.drop('id', axis='columns', inplace=True)
x_train = x_train.values.ravel()
y_train = y_train.values.ravel()

# validation data
x_validation = pd.read_csv('validation_samples.txt', sep='\t', names=['id', 'text'])
y_validation = pd.read_csv('validation_labels.txt', sep='\t', names=['id', 'M/R'])
x_validation.drop('id', axis='columns', inplace=True)
y_validation.drop('id', axis='columns', inplace=True)
x_validation = x_validation.values.ravel()
y_validation = y_validation.values.ravel()

# modeling the data
v = TfidfVectorizer(ngram_range=(1, 2), max_df=0.7)
v.fit(x_train)

x_train_count = v.transform(x_train)
x_validation_count = v.transform(x_validation)

# creating and testing the model

classifier = MultinomialNB(alpha=0.09)
classifier.fit(x_train_count, y_train)
y_pred = classifier.predict(x_validation_count)

print(classifier.score(x_validation_count, y_validation))
print("The accuracy is: ", accuracy_score(y_validation, y_pred))
print("The f1 score is: ", f1_score(y_validation, y_pred))

cm = confusion_matrix(y_validation, y_pred)
print(cm)

sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title("Confusion Matrix")
plt.show()
