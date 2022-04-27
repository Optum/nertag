from setuptools import setup

name = "nertag"
version = "0.0.0"
url = ""

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name=name,
    packages=[name],
    version=version,
    author="bhsu",
    author_email="brandon.hsu@optum.com",
    license="Apache-2.0",
    url=url,
    description="Automated named-entity recognition tagger",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    install_requires=["pandas", "nltk"],
    keywords=[
        "nlp",
        "ner",
        "dataset",
        "labeling",
    ],
)
