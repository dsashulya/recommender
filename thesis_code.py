# general
import pandas as pd
import numpy as np

# LDA
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim import corpora, models
import nltk
from gensim.models.coherencemodel import CoherenceModel

# SVD
from surprise import SVD, Dataset, Reader, dump
from surprise.accuracy import mae as MAE, rmse as RMSE
from sklearn.metrics import mean_absolute_error


# LDA
# loading the data
rev = pd.read_csv('reviews.csv')
reviews = []
for column in rev.columns:
    r = [x for x in rev[column].tolist() if type(x) != float]
    reviews.append(' '.join(r))
    
# preprocessing
stemmer = SnowballStemmer('english')
def lemmatize_stemming(text):
    ''' lemmatizes text'''
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    '''  breaks down texts into tokens
    removes stop words and words of less than 3 characters 
    returns list of lemmatized(stemmed) tokens '''
    result = []
    tokens = gensim.utils.simple_preprocess(text)
    l = len(tokens)
    for i in range(l):
        if tokens[i] not in gensim.parsing.preprocessing.STOPWORDS and len(tokens[i]) >= 3:
            result.append(lemmatize_stemming(tokens[i]))
    return result

# applying the functions to the documents -> list of stemmed tokens 
# and their frequencies in each document (bag of words)
docs = list(map(preprocess, reviews))
dictionary = gensim.corpora.Dictionary(docs)
dictionary.filter_extremes(no_below=10)
bow_corpus = [dictionary.doc2bow(doc) for doc in docs]

# choosing the optimal number of topics via coherence score
cs = []
for i in range(2, 21):
    model = gensim.models.LdaModel(bow_corpus, num_topics=i, id2word=dictionary, passes=10)
    
    cv = CoherenceModel(model=model, texts=list(docs), coherence='c_v').get_coherence()
    cs.append(cv)
    print(f'{i} topics, Coherence Score = {cv: .3f}')
    
# buiding the 6 topic model and saving the topics
model = gensim.models.LdaModel(bow_corpus, num_topics=6, id2word=dictionary, passes=50)

# saving the topics and the model
with open('topics.txt', 'w') as f:
    for l in range(6):
        f.write(f'TOPIC {l}:\n')
        for i in model.get_topic_terms(l):
            f.write(f'{dictionary[i[0]]}\n')
        f.write('\n')
gensim.models.LdaModel.save('six_topics')

# preparing the final dataset to use in the app
topics = []
for sight in bow_corpus:
    topics.append(model[sight])
    
top = pd.DataFrame(rev.columns, columns=['index'])
t0, t1, t2, t3, t4, t5 = [], [], [], [], [], []
for sight in topics:
    for topic in sight:
        if topic[0] == 0: t0.append(topic[1])
        if topic[0] == 1: t1.append(topic[1])
        if topic[0] == 2: t2.append(topic[1])
        if topic[0] == 3: t3.append(topic[1])
        if topic[0] == 4: t4.append(topic[1])
        if topic[0] == 5: t5.append(topic[1])
top['topic0'] = t0
top['topic1'] = t1
top['topic2'] = t2
top['topic3'] = t3
top['topic4'] = t4
top['topic5'] = t5

attractions = pd.read_csv('attractions.csv')
attractions = pd.merge(topic, attractions, how='left', on='index')
attractions['pop_score'] = attractions['pop_score'] * attractions['rating']
attractions['unpop_score'] = [norm(1/x, 1/max(r), 1/min(r)) for x in r] * attractions['rating']

for i in range(6):
    attractions[f'pop_topic{i}'] = attractions[f'topic{i}'] * attractions['pop_score']
    attractions[f'unpop_topic{i}'] = attractions[f'topic{i}'] * attractions['unpop_score']
attractions.to_csv('attractions.csv', index=False)


# FunkSVD
# loading the data
ratings = pd.read_csv('ratings_reshaped.csv')

# creating train and test sets
train = ratings.sample(frac=0.8)
test = ratings.drop(train.index.tolist())

train = Dataset.load_from_df(train, Reader(rating_scale=(1, 5)))
test = Dataset.load_from_df(test, Reader(rating_scale=(1, 5)))
train = train.build_full_trainset()
test = test.build_full_trainset()
test = test.build_testset()

# building the model
svd = SVD(n_factors=7, n_epochs=100, lr_all=0.005, reg_all=0.005, biased=True)
svd.fit(train)

# getting the MAE
pred = svd.test(test)
print(f"Test MAE: {MAE(pred, verbose=False): .2f}")

# saving the model
dump.dump('funksvd', predictions=pred, algo=svd, verbose=1)