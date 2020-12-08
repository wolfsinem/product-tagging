# Automated Product Tagging

This project of Automated Product Tagging is part of my internal project for my internship: [Onestdata](https://onestdata.com/). 

Every product is made up of several tags that are set to describe its characteristics. These tags can include anything about the product, e.g. color, size and type.
These tags allow visitors to filter products based on the categories they want to explore.

The algorithm is largely based on the NLTK library. The NLTK (Natural Language Toolkit) library is a leading platform for building Python programs to work with human language data. Since we work with a dataset which has a description column, containing human language, this package is really useful in producing tags for products. For more documentation you can click on this link: [NLTK](https://www.nltk.org/)

The machine learning model on the other hand is based on the TfIdfVectorizer. This method tokenizes documents/texts, learns the vocabulary and inverses the document frequency weighting and allows you to encode new documents. For more documentation you can click on this link: [TFIDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

Alongside the model I chose for the LinearSVC (Linear Support Vector Classification). The purpose of this model is to fit to the data you provide, returning a "best fit" hyperplane that divides, or categorizes, your data. From there, after getting the hyperplane, you can then feed some features to your classifier to see what the "predicted" class is. See: [NLTK](https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/). Because we are dealing with products that can carry multiple tags, this is a good multilabel classification model.

# Workflow
![workflow]((https://github.com/wolfsinem/product-tagging/blob/master/img/workflow.png))

## UI Home page to use the machine learning model
![alt text](https://github.com/wolfsinem/product-tagging/blob/master/img/UI3.png)

## UI Upload CSV page to upload a file
![alt text](https://github.com/wolfsinem/product-tagging/blob/master/img/UI3.2.png)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the needed libraries.

```bash
pip install -r requirements.txt
```

## Run

```python
flask run
```
or 

```python
python app.py
```