# Data processing
import pandas as pd
from collections import Counter

# Language processing
from nltk.corpus import stopwords 
import nltk
# nltk.download('stopwords') # if you haven't downloaded this yet.

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def load_df(filepath= '/Users/wolfsinem/product-tagging/data/flipkart_com-ecommerce_sample.csv'):
    """This simple function is just for reading purposes and returning a pandas 
    df
    """
    return pd.read_csv(filepath)
    

def model_dataframe():
    """This function will check if the loaded dataframe has both columns 
    product_name and description. If yes, make a new model dataframe with an 
    empty tags column.
    """

    df = load_df()
    df.dropna(inplace=True)
    if 'product_name' and 'description' in df.columns:
        model_df = df[['product_name','description']]
        pd.options.mode.chained_assignment = None 
        model_df['tags'] = ""
    else:
        raise ValueError("Columns product_name and description don't exist")
    return model_df


def tokenize_model(sentence):
    """This function goes through the tokenization process for the description 
    column.

    :param df: The final dataframe which the model_dataframe function returns.
    :type df: str.
    """

    tokenizer = nltk.RegexpTokenizer(r"\w+")
    new_words = tokenizer.tokenize(sentence)
    new_words = [token.lower() for token in new_words] # set to a lower case
    
    stop_words = set(stopwords.words('english')) 
    # these are manually filtered words, not the most efficient way 
    manually_filtered = ['product', 'type', 'fabric', 'material', 'warranty', 
                        'key', 'details', 'use', 'avoid']

    filtered_sentence = [w for w in new_words if not w in stop_words 
                        and not w in manually_filtered
                        ]
    count_terms = Counter(filtered_sentence).most_common(10)
    # extract the first element of each sublist
    return [item[0] for item in count_terms] 


def extend_tokenized_model():
    """ This function uses a for-loop to create a set of tags for each product
    description in the dataframe.
    """
    
    model_df = model_dataframe()

    token_lists = []
    for i in model_df['description']:
        token_lists.append(
            [x for x in tokenize_model(str((i))) if not any(c.isdigit() for c in x)
            ])

    for i in range(len(model_df.index)):
        model_df.at[i,'tags'] = token_lists[i]
    return model_df


# print(extend_tokenized_model().head())
# this returns a newly made dataframe with an extra 'tags' column for each
# column 