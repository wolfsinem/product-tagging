from flask import Flask, render_template, request, redirect
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import sys
import os 

sys.path.append('/Users/wolfsinem/product-tagging')

from product_tagging.tags_generator import tokenized_list
from similarityRate import lemma_tag

N = 5000 
MODEL = tokenized_list()

# Preprocessing 
model_df = MODEL[:N]
target_variable = model_df['tags']

mlb = MultiLabelBinarizer()
target_variable = mlb.fit_transform(target_variable)

# Open our classifier and vectorizer pickle files
with open('/Users/wolfsinem/product-tagging/data/classifier2', 'rb') as training_model:
    model = pickle.load(training_model)

with open('/Users/wolfsinem/product-tagging/data/vect2', 'rb') as tfvectorizer:
    vectorizer = pickle.load(tfvectorizer)

# Initializing the Flask app 
app = Flask(__name__)
app.config.from_object("config.DevelopmentConfig")


def allowed_file(filename):
    """To make sure the user can't upload any type of file we will use this
    function to limit the input. 

    :param filename: 
    :type filename:
    """ 
    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_FILE_EXTENSIONS"]:
        return True
    else:
        return False


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

        return render_template("index.html", generated_tags = generated_tags)


@app.route('/warning', methods=['POST', 'GET'])
def send_warning():
    """Function to send out the warning template to user."""
    return render_template('warning.html')


@app.route('/read_csv', methods=['POST', 'GET'])
def read_csv():
    """This predict function will load the persisted model into memory when the 
    application starts and create an API endpoint that takes the input CSV file, 
    transforms it into the appropiate format and returns a new CSV file with a
    new column; tags. 
    """
    
    if request.method == "POST":

        if request.files:

            file = request.files["file"]

            if file.filename == "":
                print("There is no file")
                return redirect(request.url)

            if allowed_file(file.filename):
                file.save(os.path.join(app.config["FILE_UPLOADS"], file.filename))
                print('The uploaded file: {} has been saved into the directory'.format(file.filename))
                return redirect(request.url)

            else:
                print("That file extension is not allowed")
                return send_warning()

    return render_template("prediction.html")


@app.route('/generated', methods=['POST', 'GET'])
def generate_tags():
    """This product generator doesn't use any machine learning to create tags 
    based on the text input. It only uses the NLTK library.
    """

    if request.method == 'POST':

        user_input_size = request.form.get('tags_size')
        if user_input_size:
            tags_size = int(user_input_size) # + 1
        else:
            tags_size = 20

        user_input_string = request.form.get('product_description')
        tags_set = lemma_tag(user_input_string, tags_size)

        return render_template("algorithm.html", tags_set = tags_set)
    
    return render_template("algorithm.html")


if __name__ == "__main__":
    app.run(debug=True)