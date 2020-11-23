"""This is an example file of how you can deploy your Machine Learning model
after you've trained it and saved it into pickle files.
At the end of this process you would have 2 different pickle files:
- classifier; which is the LinearSVC pipeline model
- vectorizer; which is the TfIdfVectorizer model
See: Jupyter Notebook file 'Deployment' for more information. 
"""

import pickle
import sys

sys.path.append('/Users/wolfsinem/product-tagging')
from product_tagging.tags_generator import tokenized_list

# Machine Learning - model training
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

n = 2000 # number of rows
model_df = tokenized_list()
model_df = model_df[:n]

target_variable = model_df['tags']

mlb = MultiLabelBinarizer()
target_variable = mlb.fit_transform(target_variable)


vectorizer = TfidfVectorizer(strip_accents='unicode', 
                             analyzer='word', 
                             ngram_range=(1,3), 
                             stop_words='english',
                             token_pattern=r'\w{3,}'
                            )

# fit the independent features
independent_variable = vectorizer.fit_transform(model_df['description'])

print('Independent variable shape: {}'.format(independent_variable.shape))
print('Target variable shape: {}'.format(target_variable.shape))

X_train, X_test, y_train, y_test = train_test_split(
                                        independent_variable, 
                                        target_variable, 
                                        test_size=0.2, 
                                        random_state=42, 
                                        )

Linear_pipeline = Pipeline([
                ('clf', OneVsRestClassifier(LinearSVC(
                                                class_weight='balanced',
                                                random_state=42,
                                                tol=1e-1,
                                                C=8.385), 
                                                n_jobs=-1)),
            ])

Linear_pipeline.fit(X_train, y_train)

prediction = Linear_pipeline.predict(X_test)
print('Accuracy for LinearSVC is {}'.format(accuracy_score(y_test, prediction)))

# saving everything into pickle file
with open('classifier2', 'wb') as picklefile:
    pickle.dump(Linear_pipeline,picklefile)

with open('vect2', 'wb') as picklefile:
    pickle.dump(vectorizer,picklefile)

# open the pickle files 
with open('/Users/wolfsinem/product-tagging/classifier2', 'rb') as training_model:
    model = pickle.load(training_model)

with open('/Users/wolfsinem/product-tagging/vect2', 'rb') as training_model:
    vectorizer = pickle.load(training_model)


user_input_string = ["""Key Features of Alisha Solid Womens Cycling Shorts Cotton 
                    'Lycra Navy, Red, Navy,Specifications of Alisha Solid Womens 
                    Cycling Shorts Shorts Details Number of Contents in Sales 
                    Package Pack of 3 Fabric Cotton Lycra Type Cycling Shorts 
                    General Details Pattern Solid Ideal For Womens Fabric Care 
                    Gentle Machine Wash in Lukewarm Water, Do Not Bleach Additional 
                    Details Style Code ALTHT_3P_21 In the Box 3 shorts"""]

user_input_string = vectorizer.transform(user_input_string).toarray()
label = model.predict(user_input_string)
tags = mlb.inverse_transform(label)
print(tags)