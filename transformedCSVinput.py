import os
import sys

# Data processing
import pandas as pd
from collections import Counter

# Language processing
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
import nltk
# nltk.download('stopwords') # if you haven't downloaded this yet.

sys.path.append('/Users/wolfsinem/product-tagging/static/data')


def retrieve_csv(path="/Users/wolfsinem/product-tagging/static/data/uploads"):
    """returns the latest uploaded file in the given folder path with the 
    conditions of it being a csv file and starting with csv.
    
    :param path: path of the directory in which the uploads are saved.
    :type path: string.
    """
    
    files = os.listdir(path)
    for fname in files:
        if fname.startswith('sub') and fname.endswith(".csv"):
            paths = [os.path.join(path, basename) for basename in files]
            return max(paths, key=os.path.getctime)
        else:
            raise ValueError("The file we are looking for isnt't there")


def model_dataframe():
    """This function will check if the dataset which is uploaded by the user has 
    both columns product_name and description. If yes, make a new model dataframe 
    with an empty tags column so we can later add new tags to it.
    """

    df = pd.read_csv(retrieve_csv())
    
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['description'],inplace=True)
    
    if 'product_name' in df.columns and 'description' in df.columns: 
        model_df = df[['product_name','description']]
        pd.options.mode.chained_assignment = None 
        model_df['tags'] = ""
    else:
        raise ValueError("Columns product_name and description don't exist, please rename the column names")
    return model_df


def tokenize_user_text_input(sentence, size_tags=10):
    """This function splits a string into substrings using a regular expression
     using RegexpTokenizer. Additionally it counts the occurence of each word
     and returns the top x words which can be used as tags

    :param sentence: Text description of a product
    :type sentence: string
    """
    
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    new_words = tokenizer.tokenize(str(sentence))
    new_words = [token.lower() for token in new_words]
    
    stop_words = set(stopwords.words('english')) 

    filter_tokens = [w for w in new_words if not w in stop_words]
    count_terms = Counter(filter_tokens).most_common(size_tags)
    count_terms = [item[0] for item in count_terms]

    token_lists = []
    for i in count_terms:
        token_lists.append(i)
    
    token_lists = [item for item in token_lists if not item.isdigit()]
    
    return token_lists


def lemma_tag(sentence): 
    """This function uses the NLTK lemmatizer function in the first part. 
    Lemmatization, unlike Stemming, reduces the inflected words properly ensuring 
    that the root word belongs to the language See: 
    https://www.datacamp.com/community/tutorials/stemming-lemmatization-python

    To reduce the amount of duplicates in a set of tags we will thus use 
    lemmatization. Words like 'weight' and 'weights' will be considered the same 
    and be saved as 'weight'. In addition to that we have a few other conditions 
    to clean the set of tags.

    :param sentence: A single sentence e.g. product description
    :type sentence: string
    """

    lemmatizer = WordNetLemmatizer()

    lemm_set = []
    for word in tokenize_user_text_input(sentence):
        tag = lemmatizer.lemmatize(word)
        lemm_set.append(tag)
    
    lemm_set = list(set(lemm_set))
    lemm_set = [x for x in lemm_set if not x[-3:] == "ing"]
    
    return [i for i in lemm_set if len(i) > 1]


def extend_df(df=model_dataframe()):
    """This function extends the original dataframe with an extra column 'tags'.
    This function uses both the lemma_tag() and tokenize_user_text_input() 
    function to tokenize and clean the set of tags.
    
    :param df: This would be the orginal df imported by the user.
    :type df: string.
    """
    
    for i in df.index:
        df.at[i,'tags'] = lemma_tag(df.loc[i]['description'])
        
    return df


def export_extendedDF(path="/Users/wolfsinem/product-tagging/static/data/exports", df=model_dataframe()):
    """This function exports the created extended subset into the right folder.
    From this folder the file can be exported to the right user again.

    :param path: This is the path which the file is exported to.
    :type path: string.

    :param df: This would be the orginal df imported by the user.
    :type df: string.
    """
    
    if os.path.exists(path):
        outname = "extended-" + os.path.basename(os.path.normpath(retrieve_csv()))
        full_path = os.path.join(path, outname) 
        df.to_csv(full_path)
        
    else:
        raise ValueError("This {} path file does not exist".format(path))


if __name__ == "__main__":
    export_extendedDF()