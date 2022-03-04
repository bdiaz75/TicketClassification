
import numpy as np
import pandas as pd
import csv

import seaborn as sns
import matplotlib.pyplot as plt


#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
#import nltk
#dler = nltk.downloader.Downloader()
#dler._update_index()
#dler.download('all')
#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import gensim
from gensim.models import Word2Vec

''' Here we will implement the NLP algorithms using the training date.
In the training data, every ticket is labeled with a target. The target is 0 if
its a level 1 ticket, and the target is 1 if its a level 2 ticket. This target
is what we will need to predict. '''


#with open('./Train Data/testTrainData.csv','r') as file:
#    reader = csv.DictReader(file)

df_train = pd.read_csv('./Train Data/testTrainData.csv', encoding = "ISO-8859-1")

#df_train = pd.read_csv('./Train Data/trainData2 - Copy.csv', encoding = "ISO-8859-1")
#df_test = pd.read_csv('./Test Data/testTestData.csv',encoding = "ISO-8859-1")
df_test = pd.read_csv('./Test Data/testData1.csv',encoding = "ISO-8859-1")

x=df_train['Target'].value_counts()
print(x)
sns.barplot(x=x.index,y=x)
#plt.show()


df_test.dropna(subset=['Detail'], inplace=True)
df_train.dropna(subset=['Detail'], inplace=True)
f = df_train.isna().sum()
print('missing values')
print(f)

df_train['word_count'] = df_train['Detail'].apply(lambda x: len(str(x).split()))
print('word counts for level 2 tickets:')
print(df_train[df_train['Target']==1]['word_count'].mean()) #Level 2
print('word counts for level 1 tickets:')
print(df_train[df_train['Target']==0]['word_count'].mean()) #Level 1


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))
train_words=df_train[df_train['Target']==1]['word_count']
ax1.hist(train_words,color='red')
ax1.set_title('Level 2')
train_words=df_train[df_train['Target']==0]['word_count']
ax2.hist(train_words,color='green')
ax2.set_title('Level 1')
fig.suptitle('Words per ticket')
#plt.show()


nltk.download('punkt')
nltk.download('stopwords')
#nltk.download('corpus')


''' here we do preprocessing of the text to remove unnecessary words, punctuation, etc'''

def preprocess(text):
    text = text.lower()
    text=text.strip()
    text=re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]',' ',text)
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'\s+',' ',text)
    return text


# STOPWORD REMOVAL
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

#LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()

# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)



def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))




df_train['clean_text'] = df_train['Detail'].apply(lambda x: finalpreprocess(x))
#df_train=df_train.drop(columns=['word_count','char_count','unique_word_count'])
df_train.head()



#building Word2Vec model
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))
    def fit(self, X, y):
        return self
    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


df_train['clean_text_tok']=[nltk.word_tokenize(i) for i in df_train['clean_text']]
model = Word2Vec(df_train['clean_text_tok'],min_count=1)
w2v = dict(zip(model.wv.index_to_key, model.wv.vectors))


X_train, X_test, y_train, y_test = train_test_split(df_train["clean_text"],df_train["Target"],test_size=0.2,shuffle=True)


X_train_tok= [nltk.word_tokenize(i) for i in X_train]
X_test_tok= [nltk.word_tokenize(i) for i in X_test]


tfidf_vectorizer = TfidfVectorizer(use_idf=True)
X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)



modelw = MeanEmbeddingVectorizer(w2v)

# converting text to numerical data using Word2Vec
X_train_vectors_w2v = modelw.transform(X_train_tok)
X_val_vectors_w2v = modelw.transform(X_test_tok)


'''
FITTING THE CLASSIFICATION MODEL using Logistic Regression(tf-idf)
'''
lr_tfidf=LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2')
lr_tfidf.fit(X_train_vectors_tfidf, y_train)  #model

#Predict y value for test dataset
y_predict = lr_tfidf.predict(X_test_vectors_tfidf)
y_prob = lr_tfidf.predict_proba(X_test_vectors_tfidf)[:,1]


#print(classification_report(y_test,y_predict))
#print('Confusion Matrix:',confusion_matrix(y_test, y_predict))

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)

'''
FITTING THE CLASSIFICATION MODEL using Naive Bayes(tf-idf)
'''
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_vectors_tfidf, y_train)
#Predict y value for test dataset
y_predict = nb_tfidf.predict(X_test_vectors_tfidf)
y_prob = nb_tfidf.predict_proba(X_test_vectors_tfidf)[:,1]
print(classification_report(y_test,y_predict))
print('Confusion Matrix:',confusion_matrix(y_test, y_predict))

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)


'''
testing on new dataset with best model

'''

df_test['clean_text'] = df_test['Detail'].apply(lambda x: finalpreprocess(x)) #preprocess the data
X_test=df_test['clean_text']
X_vector=tfidf_vectorizer.transform(X_test) #converting X_test to vector
y_predict = lr_tfidf.predict(X_vector)      #use the trained model on X_vector
y_prob = lr_tfidf.predict_proba(X_vector)[:,1]
df_test['predict_prob']= y_prob
df_test['target']= y_predict
print(df_test.head())
final=df_test[['No.','predict_prob','clean_text','target']].reset_index(drop=True)
final.to_csv('submission1.csv')
