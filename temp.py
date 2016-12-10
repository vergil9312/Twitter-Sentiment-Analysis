from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn import linear_model
import re
import string
from sklearn.model_selection import train_test_split
import csv
import matplotlib.pyplot as plt

# import tweets with positive or negative labels
with open('two_point_tweet.tsv') as tsv:
    reader = csv.reader(tsv,delimiter="\t")
    data = list(reader)

data = np.array(data)

# pattern for repeated character in a word
replacement_patterns = [
     (r'won\'t', 'will not'),
     (r'can\'t', 'cannot'),
     (r'i\'m', 'i am'),
     (r'ain\'t', 'is not'),
     (r'(\w+)\'ll', '\g<1> will'),
     (r'(\w+)n\'t', '\g<1> not'),
     (r'(\w+)\'ve', '\g<1> have'),
     (r'(\w+)\'s', '\g<1> is'),
     (r'(\w+)\'re', '\g<1> are'),
     (r'(\w+)\'d', '\g<1> would')
]

# Replace abbreviation
def replaceShort(s):
    for (pattern,repl) in replacement_patterns:
        s = re.sub(pattern,repl,s)
    return s

# Replace character reprition
def replaceRep(s):
    repl_word = re.sub(r'(\w*)(\w)\2(\w*)',r'\1\2\3',s)
    if repl_word != s:
        return replaceRep(repl_word)
    else:
        return repl_word

# Convert positive to 1, negative to 0
def classToNum(s):
    if s == 'negative':
        return 0
    elif s == 'positive':
        return 1

# Data preprocessing
# Extract labels and tweets
data = data[:,2:]
# Convert tweets to lower class
data[:,1] = [x.lower() for x in data[:,1]]
# Delete 'not available' tweets
data = np.array([x for x in data if x[1] != 'not available'])
# Delete '#' in a tweet
data[:,1] = [re.sub('#','',x) for x in data[:,1]]
# Delete '@' in a tweet
data[:,1] = [re.sub('@','',x) for x in data[:,1]]
# Delete URL address
data[:,1] = [re.sub(' http.*\w','',x) for x in data[:,1]]
data[:,1] = [re.sub(' http.*\d','',x) for x in data[:,1]]
# Convert abbreviation for number to proper English
data[:,1] = [re.sub('1st','first',x) for x in data[:,1]]
data[:,1] = [re.sub('2nd','second',x) for x in data[:,1]]
data[:,1] = [re.sub('3rd','third',x) for x in data[:,1]]
data[:,1] = [re.sub('4th','fourth',x) for x in data[:,1]]
data[:,1] = [re.sub('5th','fifth',x) for x in data[:,1]]
data[:,1] = [re.sub('6th','sixth',x) for x in data[:,1]]
data[:,1] = [re.sub('7th','seventh',x) for x in data[:,1]]
data[:,1] = [re.sub('8th','eighth',x) for x in data[:,1]]
data[:,1] = [re.sub('9th','nineth',x) for x in data[:,1]]
# Delete all numbers
data[:,1] = [re.sub('\d','',x) for x in data[:,1]]
# Delete all punctuation
data[:,1] = [''.join(c for c in x if c not in string.punctuation) for x in data[:,1]]
# Replace abbreviation
data[:,1] = [replaceShort(x) for x in data[:,1]]
# Replace character repetition with proper English
data[:,1] = [replaceRep(x) for x in data[:,1]]
# Delete excessive whitespace
data[:,1] = [re.sub('\s+',' ',x) for x in data[:,1]]

# Categories of tweets
categories = ['negative','positive']
# Convert labels of tweets to number
data[:,0] = [classToNum(x) for x in data[:,0]]
# Declare lists of accuracies and F-score for different classification method and size of training set
accuracy_NB = []
accuracy_SVM = []
accuracy_Mul = []
accuracy_ensem1 = []
accuracy_ensem2 = []
F_score_NB = []
F_score_SVM = []
F_score_Mul = []
F_score_ensem1 = []
F_score_ensem2 = []
holder = data

# Print out classification accuracy and F-score
def printScore(predicted,y_test,s):
    print(s)
    print('prediction accuracy:', np.mean(predicted == y_test))
    print('F-score:', f1_score(y_test,predicted))

# Loop from 10% to 100% training set size, append the result in respective list
for j in range(1,11):
    # X holds tweets, y holds labels
    X = data[:,1]
    y = data[:,0]
    y = [int(s) for s in y]
    
    # Split train, test set
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=45)
    X_train = X_train[:int(len(X_train)*j/10)]
    y_train = y_train[:int(len(y_train)*j/10)]
    
    # Naive Bayes
    text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
    text_clf = text_clf.fit(X_train, y_train)
    docs_test = X_test
    predicted_NB = text_clf.predict(docs_test)
    accuracy_NB.append(np.mean(predicted_NB == y_test))
    F_score_NB.append(f1_score(y_test,predicted_NB))
    
    # Support Vector Machine
    text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=30, random_state=42)),])
    text_clf = text_clf.fit(X_train, y_train)
    predicted_SVM = text_clf.predict(docs_test)
    accuracy_SVM.append(np.mean(predicted_SVM == y_test))
    F_score_SVM.append(f1_score(y_test,predicted_SVM))
    
    # Multinomial Logistic Regression
    text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', linear_model.LogisticRegression()),])
    text_clf = text_clf.fit(X_train, y_train)
    predicted_Mul = text_clf.predict(docs_test)
    accuracy_Mul.append(np.mean(predicted_Mul == y_test))
    F_score_Mul.append(f1_score(y_test,predicted_Mul))
    
    # Ensemble 1
    # Predict positive only if NB, SVM and Multinomial predicts positive
    predicted_ensem1 = []
    for i in range(len(predicted_NB)):
        if predicted_NB[i] == 1 and predicted_SVM[i] == 1 and predicted_Mul[i] ==1:
            predicted_ensem1.append(1)
        else:
            predicted_ensem1.append(0)
    predicted_ensem1 = np.array(predicted_ensem1)
    accuracy_ensem1.append(np.mean(predicted_ensem1 == y_test))
    F_score_ensem1.append(f1_score(y_test,predicted_ensem1))
        
    # Ensemble 2
    # Predict positive if any of NB, SVM and Multinomial predicts positive
    predicted_ensem2 = []
    for i in range(len(predicted_NB)):
        if predicted_NB[i] == 1 or predicted_SVM[i] == 1 or predicted_Mul[i] == 1:
            predicted_ensem2.append(1)
        else:
            predicted_ensem2.append(0)
    predicted_ensem2 = np.array(predicted_ensem2)
    accuracy_ensem2.append(np.mean(predicted_ensem2 == y_test))
    F_score_ensem2.append(f1_score(y_test,predicted_ensem2))

# Plot classification accuracy
plt.plot(accuracy_NB,marker='o',label='NB')
plt.plot(accuracy_SVM,marker='o',label='SVM')
plt.plot(accuracy_Mul,marker='o',label='Multi')
plt.plot(accuracy_ensem1,marker='o',label='Ensem1')
plt.plot(accuracy_ensem2,marker='o',label='Ensem2')
plt.ylabel('Classification Accuracy')
plt.xlabel('Percentage of Training Size')
plt.title('Classification Accuracy vs. Training Set Size')
plt.legend()
axes = plt.gca()
axes.set_ylim([0.8,0.90])
plt.show()
plt.close()

# Plot F-score
plt.plot(F_score_NB,marker='o',label='NB')
plt.plot(F_score_SVM,marker='o',label='SVM')
plt.plot(F_score_Mul,marker='o',label='Multi')
plt.plot(F_score_ensem1,marker='o',label='Ensem1')
plt.plot(F_score_ensem2,marker='o',label='Ensem2')
plt.ylabel('F-sore')
plt.xlabel('Percentage of Training Size')
plt.title('F-score vs. Training Set Size')
plt.legend()
axes = plt.gca()
axes.set_ylim([0.90,0.95])
plt.show()