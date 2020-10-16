# import sys
# sys.path.append('/Users/wolfsinem/product-tagging')

# Data processing
import pandas as pd
import numpy as np
from collections import Counter

# Language processing
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import nltk
# nltk.download('stopwords') # if you haven't downloaded this yet.

# Machine Learning - model training
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.multiclass import OneVsRestClassifier

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# load in the dataframe
def load_df(filepath= '/Users/wolfsinem/product-tagging/data/flipkart_com-ecommerce_sample.csv'):
    """This simple function is just for reading purposes and returning a pandas df"""
    return pd.read_csv(filepath)
    

def model_dataframe():
    """
    This function will check if the loaded dataframe has both columns product_name
    and description. If yes, make a new model dataframe with an empty tags column.
    """
    df = load_df()
    if 'product_name' and 'description' in df.columns:
        model_df = df[['product_name','description']]
        pd.options.mode.chained_assignment = None 
        model_df['tags'] = ""
    else:
        raise ValueError("Dataframe columns product_name and description don't exist")
    return model_df

# print(model_dataframe().head(5))

def tokenize_model_df(df):
    """
    This function goes through the tokenization process for the description column.

    :param df: The final dataframe which the model_dataframe function returns.
    :type df: str.
    """
    df = model_dataframe()

    tokenizer = nltk.RegexpTokenizer(r"\w+")
    new_words = tokenizer.tokenize(sentence)
    new_words = [token.lower() for token in new_words] # set to a lower case
    
    stop_words = set(stopwords.words('english')) 

    filtered_sentence = [w for w in new_words if not w in stop_words]
    count_terms = Counter(filtered_sentence).most_common(10) # most common 10 terms
    # extract the first element of each sublist
    return [item[0] for item in count_terms] 
