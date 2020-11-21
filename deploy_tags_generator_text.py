"""
This file is only for the deployment of the tags generator based on the input 
text given by the user. We will use the NLTK library for this http://www.nltk.org/howto/
"""

from collections import Counter
from nltk.corpus import stopwords 
import nltk

# First we import the tokenize_string function we made in the tags_generator.py 
# file and use this to split the given input string into substrings using regular
# expression using RegexpTokenizer. Additionally it counts the occurence of each
# word and returns the top x words which can be used as tags

# The second function we use is the tokenized_list() function.
# This is almost the same as the original one in our tags_generator.py file
# but since we only take in the user text input rather than a CSV file its slightly
# different.

def tokenize_user_text_input(sentence, size_tags):
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


if __name__ == "__main__":

    # user_input = """The legend continues to live in the Nike Air Force 1 '07 - Men's, a 
    #             modern version of the iconic AF1, combining classic style and modern 
    #             details. The low design offers optimum soil adhesion and a classic 
    #             look. This version of the Nike Air Force 1 features rippled leather 
    #             edges for a cleaner, slimmer line and more refined details. The 
    #             leather and fabric upper features external layers positioned at 
    #             strategic points for a lifetime durability and support. The 
    #             perforated inserts favor the breathability to keep the foot always 
    #             fresh and dry.")"""

    user_input = input("Enter a (product) description here: \n")
    print("\n")

    N = 10
    generator = tokenize_user_text_input(user_input,N)
    print("The generated set of tags are: \n")
    for tag in generator:
        print(tag)

    print("\n")