from flask import Flask, render_template, request, redirect, send_file
from werkzeug.utils import secure_filename

from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import sys
import os 

# Data processing
import pandas as pd
from collections import Counter

# Language processing
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
import nltk

sys.path.append('/Users/wolfsinem/product-tagging')
from product_tagging.tags_generator import tokenized_list
from similarityRate import lemma_tag


N = 5000 
MODEL = tokenized_list() # sys.append =----> export-user-input for new extended df


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
    function to limit the input to just the formats as described in config.

    :param filename: user input file.
    :type filename: string.
    """ 

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_FILE_EXTENSIONS"]:
        return True
    else:
        return False


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


def extend_df(df):
    """This function extends the original dataframe with an extra column 'tags'.
    This function uses both the lemma_tag() and tokenize_user_text_input() 
    function to tokenize and clean the set of tags.
    
    :param df: This would be the orginal df imported by the user.
    :type df: string.
    """
    
    for i in df.index:
        df.at[i,'tags'] = lemma_tag(df.loc[i]['description'])
        
    return df


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
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['FILE_UPLOADS'], filename)
                file.save(file_path)
                print('The uploaded file: {} has been saved into the directory'.format(filename))

                df = pd.read_csv(file_path)
                df.dropna(inplace=True)
                df.drop_duplicates(subset=['description'],inplace=True)

                if 'product_name' in df.columns and 'description' in df.columns: 
                    model_df = df[['product_name','description']]
                    pd.options.mode.chained_assignment = None 
                    model_df['tags'] = ""
                    file_path = os.path.join(app.config['FILE_UPLOADS'], "extended-" +filename)

                    extended_dataset = extend_df(model_df)
                    extended_dataset.to_csv(file_path)
                    print('The transformed file: {} has been saved into the directory'.format("extended-"+filename))

                # return redirect('/uploads/'+ filename)
                    return render_template("download.html", value="extended-"+filename)
            else:
                print("File =---> {} has no valid file format".format(file.filename))
                return send_warning()

    return render_template("upload_csv.html")


@app.route('/uploads/<filename>', methods=['GET'])
def download_csv(filename):
    """In the previous function we have read the user input's csv file
    and this function will transform this dataset to give user a new
    extended csv file back.
    """
    return render_template('download.html',value="extended-"+filename)


@app.route('/return-files/<filename>')
def return_files(filename):
    file_path = os.path.join(app.config["FILE_EXPORTS"], filename)
    return send_file(file_path, as_attachment=True)


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