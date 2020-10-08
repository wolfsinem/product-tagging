# Machine Learning - model training
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.svm import LinearSVC

from sklearn.multiclass import OneVsRestClassifier

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from tags_generator import tokenized_list


# created dataframe with new tags column 
n = 2000 # number of rows
model_df = tokenized_list()


def preprocessing_df():
    """This function preprocesses the newly created dataframe with the tags 
    column. It takes the target variable, which are the tags and uses the
    MultiLabelBinarizer. This allows us to encode multiple labels per instance
    and fit the label sets binarizer and transform the given label sets.  
    """

    target_variable = model_df['tags'][:n]
    mlb = MultiLabelBinarizer()
    target_variable = mlb.fit_transform(target_variable)

    return target_variable


def tfidfvec():
    """This function uses the TfidfVectorizer method to tokenize texts, 
    learn the vocabulary and inverse the document frequency weighting and 
    allows you to encode new texts.  
    """

    vectorizer = TfidfVectorizer(
                             strip_accents='unicode', 
                             analyzer='word', 
                             ngram_range=(1,3), 
                             stop_words='english',
                             token_pattern=r'\w{3,}'
                            )
    
    product_description = model_df['description'][:n]
    independent_variable = vectorizer.fit_transform(product_description)
    return independent_variable


def traintestSplit(test_size=0.3, random_state=42):
    """This function splits the independent variable and target variable
    data into a test and train set. 
    """
    
    independent_variable = tfidfvec()
    target_variable = preprocessing_df()

    X_train, X_test, y_train, y_test = train_test_split(
                                            independent_variable,
                                            target_variable,
                                            test_size = test_size,
                                            random_state = random_state                   
    )

    return X_train, X_test, y_train, y_test


def linearSVC_pipeline(random_state=42, tol=1e-1, C=8.385, n_jobs=-1):
    """This function uses the TfidfVectorizer method to tokenize texts, 
    learn the vocabulary and inverse the document frequency weighting and 
    allows you to encode new texts.
    """

    X_train, X_test, y_train, y_test = traintestSplit()

    Linear_pipeline = Pipeline([
                ('clf', OneVsRestClassifier(LinearSVC(
                                                class_weight = 'balanced',
                                                random_state = random_state,
                                                tol = tol,
                                                C = C ), 
                                                n_jobs = n_jobs)),
            ])
    
    svcpipeline = Linear_pipeline.fit(X_train, y_train)
    prediction = Linear_pipeline.predict(X_test)
    accScore = accuracy_score(y_test, prediction)
    return svcpipeline, prediction,accScore


svcpipeline, prediction, accScore = linearSVC_pipeline()
print(accScore)