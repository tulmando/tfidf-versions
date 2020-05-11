import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="example-pkg-tulmando",
    version="0.0.1",
    author="Ido Tulman",
    author_email="tulmando@gmail.com",
    description="A simulation for few versions of tf-idf",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tulmando/tfidf-versions",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
