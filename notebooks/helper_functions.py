from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from collections import Counter
import re

def get_stopwords():
    # additional stopwords specific to the corpus
    stop_words = stopwords.words('english')
    stop_words.extend(['chapter', 'mr', 'mrs', 'miss'])

    return stop_words

def remove_roman_numerals(text:str):
    '''
    Takes string as input and removes CHAPTER followed by Roman numerals and a period
    '''
    # Define the regex pattern to match "CHAPTER" followed by a Roman numeral up to 100 and a period
    roman_numeral_pattern = r'CHAPTER\s[IVXL]+\.'
    # Replace matched patterns with an empty string
    cleaned_text = re.sub(roman_numeral_pattern, '', text)
    
    return cleaned_text

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
    stop_words = get_stopwords()
    # wnl = WordNetLemmatizer()
    text_counter = Counter([wnl.lemmatize(x.lower()) for x in regexp_tokenize(text, '[-\'\w]+') if x.lower() not in stop_words])

    return text_counter