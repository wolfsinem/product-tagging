import sys
import nltk 
# nltk.download('averaged_perceptron_tagger') # download once

# python built in library to calculate the similarity
import difflib
from extended_df import model_dataframe
from nltk.stem import WordNetLemmatizer 

sys.path.append('/Users/wolfsinem/product-tagging/product_tagging')
from tags_generator import tokenize_string


df = model_dataframe()


def similar(string1, string2):
    """Function to calculate the similarity of two (product) descriptions.
    In a large dataset with product descriptions, product descriptions are most 
    likely to be similar to each other. 

    :param string1: This would be the first textual description. 
    :type string1: string
    
    :param string2: This would be the second textual description.
    :type string2: string
    """

    sequence = difflib.SequenceMatcher(None, string1, string2)
    score = sequence.ratio()
    return score

# print(similar(df.loc[19970]['description'],df.loc[19967]['description'])) 
# gives a similarity rate of 0.9954954954954955


def pos_tagger(product_description):
    """This function is part of the NLTK library which is a part-of-speach tagger.
    The POS-tagger processes a sequence of words, and attached a part of speach 
    tag to each word. See: http://www.nltk.org/book/ch05.html

    Use nltk.help.upenn_tagset('NN') for a description of each pos tag. First 
    download nltk.download('tagsets')

    :param product_description: Description of a prodcut. Could be any kind of text
    :type product_description: string
    """

    tags = tokenize_string(product_description)
    POS = nltk.pos_tag(tags)
    return POS

# print(pos_tagger(df.loc[5]['description']))
# [('paper', 'NN'), 
# ('crystal', 'NN'), 
# ('weight', 'VBD'), 
# ('gandhi', 'JJ'), 
# ('finish', 'JJ'), 
# ('silver', 'NN'), 
# ('eternal', 'JJ'), 
# ('super', 'JJ'), 
# ('series', 'NN'), 
# ('weights', 'NNS'), 
# ('dimensions', 'NNS'), 
# ('8cm', 'CD'), 
# ('x', 'JJ'), 
# ('beautiful', 'JJ'), 
# ('set', 'VBN'), 
# ('1', 'CD'), 
# ('clear', 'JJ'), 
# ('message', 'NN'), 
# ('5cm', 'CD'), 
# ('price', 'NN')]


def similarity_rate(description):
    """This function calculates the similarity score between words that are 
    similar to eachother in a list of tags.
    
    :param description: A product description. 
    :type description: string
    """
    
    tagged_list = tokenize_string(description)
    
    diff_set = []
    for i in tagged_list:
        diff_set.append(difflib.get_close_matches(i, tagged_list))
    
    scores = []
    for i in range(len(diff_set)):
        if not len(diff_set[i]) <= 1:
            firstW = diff_set[i][0]
            secondW = diff_set[i][1]
            similarityScore = similar(diff_set[i][0],diff_set[i][1])
            scores.append([firstW, secondW, similarityScore])
#             print("Score of similarity for {} and {} is: {}".format(firstW, secondW, similarityScore))
    
    return scores


def lemma_tag(set_tags): 
    """This function uses the NLTK lemmatizer function. Lemmatization, unlike Stemming, 
    reduces the inflected words properly ensuring that the root word belongs to the language
    See: https://www.datacamp.com/community/tutorials/stemming-lemmatization-python

    To reduce the amount of duplicates in a set of tags we will thus use lemmatization.
    Words like 'weight' and 'weights' will be considered the same and be saved
    as 'weight'. 
    """

    lemmatizer = WordNetLemmatizer()

    lemm_set = []
    for word in tokenize_string(set_tags):
        tag = lemmatizer.lemmatize(word)
        lemm_set.append(tag)
    
    lemm_set = list(set(lemm_set))
    lemm_set = [x for x in lemm_set if not any(c.isdigit() for c in x)] # remove digits
    
    return [i for i in lemm_set if len(i) > 1] # remove words with single character


# set of tags we used to generate
product_description = 'You will be bombarded with complimenting glances as you walk out wearing this black coloured analog watch. Featuring a stylish dial and attractive leather strap, this watch will be a classy touch to your look. This stylish accessory is a fine pick to flaunt with casuals as well as with formals too.'

# new set of tags
new_set = lemma_tag(product_description)
print(new_set)