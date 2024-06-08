from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize, word_tokenize
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from collections import Counter
import re
import numpy as np
import random

def get_stopwords():
    # additional stopwords specific to the corpus
    stop_words = stopwords.words('english')
    stop_words.extend(['chapter', 'mr', 'mrs', 'miss'])

    return stop_words

# Function to remove stop words
def remove_stop_words(text):
    words = word_tokenize(text)
    stop_words = get_stopwords()
    filtered_text = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_text)

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

# function to produce MultinomialNB models
def mnb_model(df, text_col, class_col, vect):
    # list to iterate over
    ids = df['text#'].unique().tolist()

    logo_list = []
    item_count = 0

    for id in ids:
        train_ids = list(filter(lambda x: x != id, ids))
        test_ids = [id]

        # subset dataframe for test id
        id_tokens = df[df['text#'] == id].copy()

        # train-test-split
        X_train = df[df['text#'].isin(train_ids)][text_col]
        X_test = df[df['text#'].isin(test_ids)][text_col]
        y_train = df[df['text#'].isin(train_ids)][class_col]
        y_test = df[df['text#'].isin(test_ids)][class_col]

        vect = vect

        X_train_vec = vect.fit_transform(X_train)
        X_test_vec = vect.transform(X_test)

        nb = MultinomialNB().fit(X_train_vec, y_train)

        y_pred = nb.predict(X_test_vec)
        y_proba = nb.predict_proba(X_test_vec)

        id_tokens['predictions'] = y_pred
        id_tokens['probabilities'] = y_proba[:,1]

        logo_list.append(id_tokens)
        item_count += 1
        print(f'text id: {id}, loop: {item_count}')

    logo_df = pd.concat(logo_list)

    return logo_df

# XGBoost function
def xgb_model(df, text_col, class_col, vect):
    # list to iterate over
    ids = df['text#'].unique().tolist()

    logo_list = []
    item_count = 0

    for id in ids:
        train_ids = list(filter(lambda x: x != id, ids))
        test_ids = [id]

        # subset dataframe for test id
        id_tokens = df[df['text#'] == id].copy()

        # train-test-split
        X_train = df[df['text#'].isin(train_ids)][text_col]
        X_test = df[df['text#'].isin(test_ids)][text_col]
        y_train = df[df['text#'].isin(train_ids)][class_col]
        y_test = df[df['text#'].isin(test_ids)][class_col]

        vect = vect

        X_train_vec = vect.fit_transform(X_train)
        X_test_vec = vect.transform(X_test)

        model = XGBClassifier(n_jobs = -1)
        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)
        y_proba = model.predict_proba(X_test_vec)

        id_tokens['predictions'] = y_pred
        id_tokens['probabilities'] = y_proba[:,1]

        logo_list.append(id_tokens)
        item_count += 1
        print(f'text id: {id}, loop: {item_count}')

    logo_df_xgb = pd.concat(logo_list)

    return logo_df_xgb

# function to produce test metrics
def get_metrics(df, actual, pred):
    actual_val = df[actual]
    pred_val = df[pred]
    
    conf_matrix = confusion_matrix(actual_val, pred_val)
    acc_score = accuracy_score(actual_val, pred_val)
    class_report = classification_report(actual_val, pred_val)

    return conf_matrix, acc_score, class_report