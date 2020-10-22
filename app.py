from flask import Flask, render_template, request
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import sys

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


@app.route('/')
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
        return render_template("index.html", generated_tags = 'The set of predicted tags are {}'.format(generated_tags))


@app.route('/generated', methods=['POST', 'GET'])
def generate_tags():
    """This product generator doesn't use any machine learning to create tags 
    based on the text input. It only uses the NLTK library.
    """

    if request.method == 'POST':
        # TODO ask user how many tags they want to generate
        tags_size = 20
        user_input_string = request.form.get('product_description')
        tags_set = tokenize_user_text_input(user_input_string, tags_size)

        # Output the generated tags by the machine learning model
        return render_template("algorithm.html", tags_set = 'The set of generated tags are {}'.format(tags_set))
    
    return render_template("algorithm.html")




if __name__ == "__main__":
    app.run(debug=True)