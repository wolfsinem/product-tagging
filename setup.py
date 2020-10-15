import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="product-tagging",
    version="1.0",
    author="Sinem Ertem",
    author_email="sinem.ertem@student.hu.nl",
    description='Product tagging using Machine Learning',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wolfsinem/product-tagging",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)