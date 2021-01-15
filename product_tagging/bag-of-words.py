from tensorflow.python.keras.preprocessing.text import Tokenizer
from tokenizer import tokenize, TOK

sentence = ["Key Features of Alisha Solid Women's Cycling Shorts Cotton Lycra Navy, Red, Navy,Specifications of Alisha Solid Women's Cycling Shorts Shorts Details Number of Contents in Sales Package Pack of 3 Fabric Cotton Lycra Type Cycling Shorts General Details Pattern Solid Ideal For Women's Fabric Care Gentle Machine Wash in Lukewarm Water, Do Not Bleach Additional Details Style Code ALTHT_3P_21 In the Box 3 shorts"]

def print_bow(sentence: str) -> None:
    tokenizer = Tokenizer(
        num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
        split=' ', char_level=False, oov_token=None, document_count=0
    )
    tokenizer.fit_on_texts(sentence)
    sequences = tokenizer.texts_to_sequences(sentence)
    word_index = tokenizer.word_index 
    bow = {}
    for key in word_index:
        bow[key] = sequences[0].count(word_index[key])
    print(bow)
    print(f"Bag of word sentence 1 :\n{bow}")
    print(f'We found {len(word_index)} unique tokens.')

print_bow(sentence)

for token in tokenize(sentence):
    print("{0}: '{1}' {2}".format(
            TOK.descr[token.kind],
             token.txt or "-",
            token.val or ""))