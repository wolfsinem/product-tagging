from flask import Flask, render_template, request, flash, redirect, make_response
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import sys
import os 
import csv

sys.path.append('/Users/wolfsinem/product-tagging')

from product_tagging.tags_generator import tokenized_list
from deploy_tags_generator_text import tokenize_user_text_input


N = 2000 

 # Load in our dataframe from the tags_generator file
MODEL = tokenized_list()

# Preprocessing 
model_df = MODEL[:N]
target_variable = model_df['tags']
mlb = MultiLabelBinarizer()
target_variable = mlb.fit_transform(target_variable)

# Open our classifier and vectorizer pickle files
with open('text_classifier.pkl', 'rb') as training_model:
    model = pickle.load(training_model)
with open('tfidfvectorizer.pkl', 'rb') as tfvectorizer:
    vectorizer = pickle.load(tfvectorizer)


# Initializing the Flask app 
app = Flask(__name__)
app.config.from_object("config.DevelopmentConfig")


@app.route('/', methods=['POST', 'GET'])
def index():
    """Setting up the main route"""
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    """This predict function will load the persisted model into memory when the 
    application starts and create an API endpoint that takes input variables, 
    transforms them into the appropiate format and return predictions; set of tags.
    """

    if request.method == 'POST':

        user_input_string = request.form.values()

        user_input_string = vectorizer.transform(user_input_string).toarray()
        label = model.predict(user_input_string)
        generated_tags = mlb.inverse_transform(label)

        # Output the generated tags by the machine learning model
        return render_template("index.html", generated_tags = generated_tags)


# move to config file
# app.config["FILE_UPLOADS"] = '/Users/wolfsinem/product-tagging/static/img/uploads'


@app.route('/read_csv', methods=['POST', 'GET'])
def read_csv():
    """This predict function will load the persisted model into memory when the 
    application starts and create an API endpoint that takes the input CSV file, 
    transforms it into the appropiate format and returns a new CSV file with a
    new column; tags. 
    """
    
    if request.method == "POST":

        if request.files:

            image = request.files["image"]
            image.save(os.path.join(app.config["FILE_UPLOADS"], image.filename))
            print('The uploaded file: {} has been saved into the directory'.format(image.filename))
            return redirect(request.url)


    return render_template("prediction.html")



@app.route('/generated', methods=['POST', 'GET'])
def generate_tags():
    """This product generator doesn't use any machine learning to create tags 
    based on the text input. It only uses the NLTK library.
    """

    if request.method == 'POST':

        user_input_size = request.form.get('tags_size')
        if user_input_size:
            tags_size = int(user_input_size) + 1
        else:
            tags_size = 20

        # TODO if the user input for tags_size is bigger than the amount of words
        # in a user input string, return an error
        user_input_string = request.form.get('product_description')
        tags_set = tokenize_user_text_input(user_input_string, tags_size)

        # TODO output should be converted 'nicer' on the UI
        # when u open the web for the first time it gives 'none' as tags set
        # this looks ugly on the UI so delete

        # Output the generated tags by the machine learning model
        return render_template("algorithm.html", tags_set = tags_set)
    
    return render_template("algorithm.html")




if __name__ == "__main__":
    app.run(debug=True)