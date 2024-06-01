from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from collections import Counter
import re
import numpy as np
import random

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
    wnl = WordNetLemmatizer()
    text_counter = Counter([wnl.lemmatize(x.lower()) for x in regexp_tokenize(text, '[-\'\w]+') if x.lower() not in stop_words])

    return text_counter


def get_tokens(sent_text:str, token_size = 2000):
    '''
    Split text into tokens of specified size. Size is denoted in characters with a default of 2000.
    The assumption is that 4 characters per token holds true, with a customary maximum context window 
    of 512 tokens.
    '''
    token_count = 0  # iterates with each token
    token_text = []  # append token to list
    token_num = []  # token in numerical order
    token_char_count = []
    token_sent_count = []
    token_word_count = []
    token = ''
    sent_count = 0

    for sent in sent_text:
        if len(sent) > token_size:
            # Handle case where a single sentence is longer than the token size
            if token:  # If there's a current token, finalize it first
                token_count += 1
                token_text.append(token)
                token_num.append(token_count)
                token_char_count.append(len(token))
                token_sent_count.append(sent_count)
                token_word_count.append(np.char.count(token, ' ') + 1)
                
                # Reset token
                token = ''
                sent_count = 0

            # Add the long sentence as its own token
            token_count += 1
            token_text.append(sent)
            token_num.append(token_count)
            token_char_count.append(len(sent))
            token_sent_count.append(1)  # This token has one sentence
            token_word_count.append(np.char.count(sent, ' ') + 1)
        else:
            if len(token) + len(sent) + 1 <= token_size:  # +1 for the space between sentences
                if len(token) == 0:
                    token = sent
                else:
                    token = token + ' ' + sent
                sent_count += 1
            else:
                token_count += 1
                token_text.append(token)
                token_num.append(token_count)
                token_char_count.append(len(token))
                token_sent_count.append(sent_count)
                token_word_count.append(np.char.count(token, ' ') + 1)

                # Reset token and add the current sentence to the new token
                token = sent
                sent_count = 1

    # Append the last token if it exists
    if token:
        token_count += 1
        token_text.append(token)
        token_num.append(token_count)
        token_char_count.append(len(token))
        token_sent_count.append(sent_count)
        token_word_count.append(np.char.count(token, ' ') + 1)

    return token_text, token_num, token_char_count, token_sent_count, token_word_count


# get a percentage of a list with random choice
def split_list_by_percent(input_list, percent):
    # calculate number to pick 
    pick_num = int(len(input_list) * percent / 100)
    # pick random sample
    train_ids = random.sample(input_list, pick_num)
    # remaining values from original list
    test_ids = [item for item in input_list if item not in train_ids]

    return train_ids, test_ids


# split text ids for train-test split so that an id is either in train or test
def split_by_id(df, ja_split:int, ff_split:int):
    train_ids = []
    test_ids = []

    # list of Austen texts
    ja_ids = df[df['author'] == 'Austen, Jane']['text#'].tolist()

    # ja_id split
    ja_train, ja_test = split_list_by_percent(ja_ids, ja_split)

    train_ids.extend(ja_train)
    test_ids.extend(ja_test)

    # subset df for fanfic
    ff = df[df['author'] != 'Austen, Jane']

    for cat in ff['text_length'].unique():
        ff_ids = ff[ff['text_length'] == cat]['text#'].tolist()
        ff_train_ids, ff_test_ids = split_list_by_percent(ff_ids, ff_split)
        train_ids.extend(ff_train_ids)
        test_ids.extend(ff_test_ids)
    
    return train_ids, test_ids