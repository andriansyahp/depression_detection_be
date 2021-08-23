from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import re
import uvicorn
import numpy as np
import pandas as pd
import nltk
import pickle

# Token Counter
from sklearn.feature_extraction.text import CountVectorizer

# Classifier
from nltk.tag import CRFTagger
from collections import Counter

# Stemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter

nltk.download('punkt')

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Slang words Handler
colloquial_df = pd.read_csv('https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv')
colloquial_dict = dict(zip(colloquial_df['slang'], colloquial_df['formal']))

with open('model/tokenizer.pickle', 'rb') as t_pkl:
    loaded_tokenizer = pickle.load(t_pkl)

with open('model/model.pickle', 'rb') as m_pkl:
    loaded_model = pickle.load(m_pkl)


class Tweet(BaseModel):
    tweet: str


def remove_url(tweet):
    return re.sub(r"http\S+", "", tweet)


def remove_username(tweet):
    return re.sub('@[\w]+','',tweet)


def remove_emoji(tweet):
    re_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+", re.UNICODE)
    return re.sub(re_pattern, '', tweet)


def replace_slang(tweet):
    return ' '.join([colloquial_dict.get(i, i) for i in tweet.split()])


def cleaning(tweet):
    normal_tw = tweet.lower()  # lowercase
    normal_tw = re.sub('\s+', ' ', normal_tw)  # remove extra space
    normal_tw = normal_tw.strip()  # trim depan belakang
    normal_tw = re.sub(r'[^\w@\s]', '', normal_tw)  # buang punctuation
    # regex huruf yang berulang kaya haiiii (untuk fitur unigram)
    normal_regex = re.compile(r"(.)\1{1,}")
    # buang huruf yang berulang
    normal_tw = normal_regex.sub(r"\1\1", normal_tw)
    return normal_tw


def remove_stopwords(tweet):
    stopwords = pd.read_csv("util/stopwords.csv", header=None)[0].values
    special_list = ['username', 'url', 'sensitive-no']
    token = nltk.word_tokenize(tweet)
    token_afterremoval = []
    for k in token:
        if k not in stopwords and k not in special_list:
            token_afterremoval.append(k)

    str_clean = ' '.join(token_afterremoval)
    return str_clean


# Unigram
def EkstraksiBoW(tweet, vectorizer=None):
    if vectorizer is None:
        unigram = CountVectorizer(ngram_range=(1, 1), max_features=2000)
        unigram_matrix = unigram.fit_transform(np.array(tweet)).todense()
        return unigram_matrix, unigram
    else:
        unigram_matrix = vectorizer.transform(np.array(tweet)).todense()
        return unigram_matrix


# Lexicon-based
def EkstraksiSentimen(list_tweet):
    pos = pd.read_csv("util/positif_vania.txt", header=None, names=['pos'])
    list_pos = pos['pos'].tolist()
    neg = pd.read_csv("util/negatif_vania.txt", header=None, names=['neg'])
    list_neg = neg['neg'].tolist()

    fitur_sentimen_all = []
    for tweet in list_tweet:
        # inisiasi value
        emosi = ["positif", "negatif"]
        value = [0, 0]
        emosi_value = {}
        for i in range(len(emosi)):
            emosi_value[emosi[i]] = value[i]

        list_kata = tweet.split()
        for k in list_kata:
            if k in list_pos:
                emosi_value["positif"] += 1
            if k in list_neg:
                emosi_value["negatif"] += 1

        fitur_sentimen_perkalimat = list(emosi_value.values())
        fitur_sentimen_all.append(fitur_sentimen_perkalimat)

    return fitur_sentimen_all


# Part of Speech
def EkstraksiPOS(list_tweet):
    ct = CRFTagger()
    ct.set_model_file("util/all_indo_man_tag_corpus_model.crf.tagger")
    pos_feat_list = []
    count_tag = []
    for tweet in list_tweet:
        token = nltk.word_tokenize(tweet)
        tag = ct.tag_sents([token])
        flat_tag = [item for sublist in tag for item in sublist]
        pos_count = Counter([j for i, j in flat_tag])
        pos_feat = [pos_count['JJ'], pos_count['NEG']]
        pos_feat_list.append(pos_feat)
    return pos_feat_list


# Ortografi
def EkstraksiOrtografi(raw_tweet):
    all_orto_feat = []
    for tw in raw_tweet:
        capital_count = sum(1 for c in tw if c.isupper())
        exclamation_count = sum((1 for c in tw if c == "!"))
        word_len = len(nltk.word_tokenize(tw))
        char_len = len(tw)
        orto_feat = [capital_count, exclamation_count, word_len, char_len]
        all_orto_feat.append(orto_feat)
    return all_orto_feat


def preprocessing(df):
    df['clean_tweet'] = df['tweet'].apply(cleaning)
    df['clean_tweet'] = df['clean_tweet'].apply(remove_url)
    df['clean_tweet'] = df['clean_tweet'].apply(remove_username)
    df['clean_tweet'] = df['clean_tweet'].apply(remove_emoji)
    df['clean_tweet'] = df['clean_tweet'].apply(replace_slang)
    df['clean_tweet'] = df['clean_tweet'].apply(remove_stopwords)
    stemmer = StemmerFactory().create_stemmer()
    df['clean_tweet'] = df['clean_tweet'].swifter.apply(lambda tweet: stemmer.stem(tweet))


def feature_engineering(df, vectorizer=None):
    if vectorizer is None:
        tweets, vectorizer = EkstraksiBoW(df['clean_tweet'].tolist())
        df['unigram'] = tweets.tolist()
    else:
        df['unigram'] = EkstraksiBoW(df['clean_tweet'].tolist(), vectorizer).tolist()
    df['sentimen'] = EkstraksiSentimen(df['clean_tweet'].tolist())
    df['pos'] = EkstraksiPOS(df['clean_tweet'].tolist())
    df['ortografi'] = EkstraksiOrtografi(df['tweet'].tolist())
    return df, vectorizer


def detection_pipeline(txt, final_model, tokenizer):
    txt_df = pd.DataFrame({'tweet': [txt]})
    preprocessing(txt_df)
    df, vectorizer = feature_engineering(txt_df, tokenizer)
    return final_model.predict(np.hstack((df['unigram'].tolist(), df['sentimen'].tolist())))[0]


@app.get('/')
async def index():
    return {'message': 'This is an API for Depression Detection Classifier!'}


@app.post('/predict')
async def predict(data: Tweet):
    """ FastAPI
    Args:
        data (Reviews): json file
    Returns:
        prediction: probability of review being positive
    """
    data = data.dict()
    text = data['tweet']
    print(text)
    result = detection_pipeline(text, loaded_model, loaded_tokenizer)
    print(result)
    return {
        'prediction': int(result)
    }


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
