from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer as wnl
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer as cvec
import nltk
import string

def clean(comment, dtm = False, lemmatize = False, stop_words = False):

    no_http = re.sub(r'''https?://[\w/._-]+''', '', comment)
    no_at = re.sub(r'@\w+', '', no_http)
    no_punc = re.sub(f'[{string.punctuation}]','',no_at)
    no_nums = re.sub(r'[0-9]+','',no_punc)
    cleaned = nltk.word_tokenize(no_nums)

    if lemmatize:
        lemma = wnl()
        cleaned = [lemma.lemmatize(i) for i in cleaned]

    if stop_words:

        #lang = language(comment)
        stop_words = set(stopwords.words('english'))
        cleaned = [i for i in cleaned if i not in stop_words and len(i) > 1 ]

    if dtm:
        
        cv = cvec()
        cv_df = cv.fit_transform(cleaned)
        cleaned = pd.DataFrame(cv_df.toarray(),columns = cv.get_feature_names())

    return cleaned
