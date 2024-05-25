from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from collections import Counter

def read_text(work_id:str):
    '''
    Open text file in read-only mode.
    Input work_id to retrieve contents.
    '''
    with open(f'../data/texts/{work_id}.txt', 'r') as file:
        data = file.read()
    
    return data

def get_counter(text:str):
    '''Create counter object from text.'''
    stop_words = set(stopwords.words('english'))
    wnl = WordNetLemmatizer()
    text_counter = Counter([wnl.lemmatize(x.lower()) for x in regexp_tokenize(text, '[-\'\w]+') if x.lower() not in stop_words])

    return text_counter