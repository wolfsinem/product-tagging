# Machine Learning - model training
from sklearn.preprocessing import MultiLabelBinarizer

import pickle
import sys

sys.path.append('/Users/wolfsinem/product-tagging')
from product_tagging.tags_generator import tokenized_list

# Limit total records so it doesn't take long to process.
N = 2000

# Load in our dataframe from the tags_generator file
MODEL = tokenized_list()

# Preprocessing 
model_df = MODEL[:N]
target_variable = model_df['tags']
mlb = MultiLabelBinarizer()
target_variable = mlb.fit_transform(target_variable)

# Open our classifier and vectorizer pickle files
with open('text_classifier.pkl', 'vect.pkl', 'rb') as training_model:
    model, vectorizer = pickle.load(training_model)

with open('tfidfvectorizer.pkl', 'rb') as tfvectorizer:
    vectorizer = pickle.load(tfvectorizer)


user_input_string = ["""Key Features of Alisha Solid Womens Cycling Shorts Cotton 
                    'Lycra Navy, Red, Navy,Specifications of Alisha Solid Womens 
                    Cycling Shorts Shorts Details Number of Contents in Sales 
                    Package Pack of 3 Fabric Cotton Lycra Type Cycling Shorts 
                    General Details Pattern Solid Ideal For Womens Fabric Care 
                    Gentle Machine Wash in Lukewarm Water, Do Not Bleach Additional 
                    Details Style Code ALTHT_3P_21 In the Box 3 shorts"""]

user_input_string = vectorizer.transform(user_input_string).toarray()
label = model.predict(user_input_string)

# Output the generated tags by the machine learning model
print(mlb.inverse_transform(label))