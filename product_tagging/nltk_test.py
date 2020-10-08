import nltk
from nltk.collocations import *

# nltk.download('genesis')
# bigram_measures = nltk.collocations.BigramAssocMeasures()
# trigram_measures = nltk.collocations.TrigramAssocMeasures()

# # change this to read in your data
# finder = BigramCollocationFinder.from_words(
#     nltk.corpus.genesis.words('english-web.txt'))

# # only bigrams that appear 3+ times
# finder.apply_freq_filter(3)

# # return the 10 n-grams with the highest PMI
# a = finder.nbest(bigram_measures.pmi, 10)
# print(a)

from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')
# nltk.download('punkt')
wordnet_lemmatizer = WordNetLemmatizer()

sentence = "Key Features of Alisha Solid Women’s Cycling Shorts Cotton Lycra Navy, Red, Navy, Specifications "
punctuations="?:!.,';"
sentence_words = nltk.word_tokenize(sentence)
for word in sentence_words:
    if word in punctuations:
        sentence_words.remove(word)

sentence_words
print("{0:20}{1:20}".format("Word","Lemma"))
for word in sentence_words:
    print ("{0:20}{1:20}".format(word,wordnet_lemmatizer.lemmatize(word, pos="v")))