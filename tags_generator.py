# Data processing
import pandas as pd
import numpy as np

# Data visualisation
import matplotlib.pyplot as plt

from collections import Counter
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import nltk
# nltk.download('stopwords') 

def create_df(filepath='/Users/wolfsinem/product-tagging/data/flipkart_com-ecommerce_sample.csv'):
    """This function creates a new dataframe based on the data in the csv file
    with an empty tags column 

    :param filepath: Path of chosen csv file.
    :type filepath: string
    """
    df = pd.read_csv((filepath))
    model_df = df[['product_name','description']]
    pd.options.mode.chained_assignment = None 
    model_df['tags'] = ""
    return model_df


def tokenize_string(sentence, size_tags=5):
    """This function splits a string into substrings using a regular expression
     using RegexpTokenizer. Additionally it counts the occurence of each word
     and returns the top 5 words which can be used as tags

    :param sentence: Text description of a product
    :type sentence: string
    """
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    new_words = tokenizer.tokenize(sentence)
    new_words = [token.lower() for token in new_words]
    
    stop_words = set(stopwords.words('english')) 
    manual_filtered_words = {'details','fabric','key','features','sales',
                            'number','contents','type','general', 
                            'specifications'}

    filter_tokens = [w for w in new_words if not w in stop_words and not w in manual_filtered_words]
    count_terms = Counter(filter_tokens).most_common(size_tags)
    return [item[0] for item in count_terms]


def tokenized_list():
    """This function creates a nested list with all the tokenized descriptions
    and adds them to the dataframe.
    """
    model_df = create_df()
    token_lists = []
    for i in model_df['description']:
        token_lists.append(
            [x for x in tokenize_string(str((i))) if not any(c.isdigit() for c in x)]
            )

    for i in range(len(model_df.index)):
        model_df.at[i,'tags'] = token_lists[i]
    return model_df
    

if __name__ == "__main__":
    df_with_tags = tokenized_list()
    print(df_with_tags.head())
    