# Information Retrieval Search Engine using PyTerrier

This project implements a complete **Information Retrieval (IR) system** for searching and ranking tweets using **PyTerrier**.

The system processes a dataset containing 40,000 tweets, creates an inverted index, retrieves relevant documents using **TF-IDF and BM25**, improves queries using **RM3 Query Expansion**, and provides a simple Flask-based search interface.

The project demonstrates the major stages of a traditional Information Retrieval pipeline:

* Data collection
* Text preprocessing
* Stopword removal
* Text cleaning
* Stemming
* Document indexing
* Query preprocessing
* TF-IDF retrieval
* BM25 retrieval
* Query expansion
* Search result ranking
* User interface development
* Retrieval evaluation

---

## Project Overview

The goal of this project is to build a search engine capable of retrieving relevant tweets based on a user's query.

The system follows this pipeline:

```text
Tweet Dataset
      |
      v
Text Preprocessing
      |
      v
Stopword Removal
      |
      v
Text Cleaning
      |
      v
Porter Stemming
      |
      v
PyTerrier Indexing
      |
      v
User Query
      |
      v
Query Preprocessing
      |
      +-------------------+
      |                   |
      v                   v
   TF-IDF               BM25
                          |
                          v
                  RM3 Query Expansion
                          |
                          v
                  Ranked Documents
                          |
                          v
                    Search Interface
```

---

## Dataset

The project uses:

```text
tweet_emotions.csv
```

The dataset contains:

```text
40,000 tweets
```

with the following columns:

| Column      | Description                              |
| ----------- | ---------------------------------------- |
| `tweet_id`  | Unique identifier for each tweet         |
| `sentiment` | Sentiment category assigned to the tweet |
| `content`   | Original tweet text                      |

Example:

```text
tweet_id: 1956967341
sentiment: empty
content: @tiffanylue i know i was listenin to bad habit earlier...
```

The tweet ID is also converted into a document identifier:

```python
df["docno"] = df["tweet_id"].astype(str)
```

This identifier is used by PyTerrier during indexing and retrieval.

---

# Text Preprocessing

Before indexing, the tweet text is processed to reduce noise and improve retrieval quality.

The preprocessing pipeline includes:

```text
Original Tweet
      |
      v
Tokenization
      |
      v
Lowercasing
      |
      v
Stopword Removal
      |
      v
URL Removal
      |
      v
Twitter Mention Removal
      |
      v
Special Character Removal
      |
      v
Whitespace Cleaning
      |
      v
Porter Stemming
      |
      v
Processed Document
```

---

## Stopword Removal

The project uses the English stopword list provided by **NLTK**.

```python
from nltk.corpus import stopwords

stop_words = set(
    stopwords.words("english")
)
```

Words such as:

```text
the
is
a
an
and
of
to
```

are removed because they usually provide limited value for document retrieval.

The function used is:

```python
def remove_stopwords(text):

    tokens = word_tokenize(text)

    filtered_tokens = [
        word.lower()
        for word in tokens
        if word.lower() not in stop_words
    ]

    return " ".join(filtered_tokens)
```

---

## Text Cleaning

A custom cleaning function removes unnecessary Twitter and formatting information.

```python
def clean(text):

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"RT ", " ", text)
    text = re.sub(r"@[\w]*", " ", text)
    text = re.sub(
        r"[\.\,\#_\|\:\?\?\/\=]",
        " ",
        text
    )

    text = re.sub(r"\t", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()
```

The preprocessing removes:

* URLs
* Retweet markers
* Twitter usernames
* Selected special characters
* Tabs
* Line breaks
* Extra whitespace

---

# Stemming

The project uses the **Porter Stemmer** from NLTK.

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
```

The stemming function converts words into simplified root forms.

```python
def Steem_text(text):

    tokens = word_tokenize(text)

    stemmed_tokens = [
        stemmer.stem(word)
        for word in tokens
    ]

    return " ".join(stemmed_tokens)
```

For example:

```text
running -> run
connected -> connect
empty -> empti
```

Stemming helps retrieve documents containing different grammatical forms of similar words.

---

# Query Preprocessing

User queries go through the same main preprocessing operations as documents.

```python
def preprocess(sentence):

    sentence = remove_stopwords(sentence)
    sentence = clean(sentence)
    sentence = Steem_text(sentence)

    return sentence
```

For example:

```text
Original Query:

empty

Processed Query:

empti
```

Applying similar preprocessing to documents and queries improves consistency during retrieval.

---

# Document Indexing

The project uses **PyTerrier** to construct the document index.

```python
indexer = pt.DFIndexer(
    "./DatasetIndex",
    overwrite=True
)

index_ref = indexer.index(
    df["processed_text"],
    df["docno"]
)
```

The resulting index contains:

```text
Number of Documents: 40,000
Number of Terms:     39,621
Number of Postings: 277,379
Number of Tokens:   285,788
```

The index does not store positional information.

---

## Inverted Index

The PyTerrier index stores relationships between:

```text
Terms
  |
  v
Documents containing those terms
```

This makes document retrieval significantly faster than checking every document individually for every query.

The notebook also explores the index lexicon and document postings.

---

# TF-IDF Retrieval

The first retrieval model implemented is **TF-IDF**.

```python
tfidf_retr = pt.BatchRetrieve(
    index,
    controls={
        "wmodel": "TF_IDF"
    },
    num_results=10
)
```

The system returns the top 10 documents ranked according to their TF-IDF score.

Example query:

```text
empty
```

After stemming:

```text
empti
```

Example results:

| Rank | Document ID |  Score |
| ---: | ----------- | -----: |
|    1 | 1962056440  | 7.8881 |
|    2 | 1963821958  | 7.6949 |
|    3 | 1966076425  | 7.2959 |
|    4 | 1695546329  | 7.2959 |
|    5 | 1962256777  | 6.7864 |

Higher scores represent documents considered more relevant to the query.

---

# TF-IDF

TF-IDF stands for:

```text
Term Frequency - Inverse Document Frequency
```

It combines two ideas.

### Term Frequency

Measures how frequently a term occurs inside a document.

```text
TF(t,d)
```

### Inverse Document Frequency

Reduces the importance of terms appearing in many documents.

Conceptually:

```text
IDF(t) = log(N / df(t))
```

where:

```text
N     = Total number of documents
df(t) = Number of documents containing term t
```

A term receives a high TF-IDF value when it:

* Appears frequently in a particular document
* Does not appear frequently across the entire collection

---

# BM25 Retrieval

The second retrieval model is **BM25**.

```python
bm25 = pt.BatchRetrieve(
    index,
    wmodel="BM25",
    num_results=10
)
```

For the processed query:

```text
empti
```

the notebook produced results such as:

| Rank | Document ID | BM25 Score |
| ---: | ----------- | ---------: |
|    1 | 1962056440  |    14.4282 |
|    2 | 1963821958  |    14.0749 |
|    3 | 1966076425  |    13.3450 |
|    4 | 1695546329  |    13.3450 |
|    5 | 1962256777  |    12.4131 |

Example retrieved tweets include:

```text
Room is so empty
```

```text
i have a empty house and no ine to share it with
```

```text
Gutted. The kitchen is empty literally EMPTY.
No even kidding. I'm so hungry
```

```text
back to Roseburg...and an empty apartment
```

```text
Excited about having an empty apartment to ourselves
for a little while
```

---

# BM25

BM25 is a probabilistic ranking algorithm widely used in Information Retrieval.

Unlike basic TF-IDF, BM25 includes mechanisms for handling:

* Term frequency
* Document frequency
* Document length
* Term-frequency saturation

This generally provides more controlled document-ranking behavior than directly increasing the score whenever a term appears repeatedly.

---

# Query Expansion

The project implements **RM3 Query Expansion**.

Query expansion attempts to improve retrieval by adding related terms to the original query.

The RM3 configuration is:

```python
rm3_expander = pt.rewrite.RM3(
    index,
    fb_terms=10,
    fb_docs=100
)
```

This means the algorithm uses:

```text
Feedback Documents = 100
Expansion Terms    = 10
```

---

## RM3 Workflow

```text
Original Query
      |
      v
BM25 Retrieval
      |
      v
Top Retrieved Documents
      |
      v
Pseudo-Relevance Feedback
      |
      v
Identify Related Terms
      |
      v
Create Expanded Query
      |
      v
Run BM25 Again
      |
      v
Updated Ranking
```

---

## Query Expansion Example

Original query:

```text
empti
```

The RM3 model generated an expanded query containing terms such as:

```text
feel
empti
todai
hou
in
room
back
excit
littl
sad
```

Each expansion term receives a weight.

For example:

```text
empti^0.779976130
room^0.033506181
sad^0.024571197
feel^0.022337455
```

The original term remains the most strongly weighted term.

---

# Retrieval Before and After Expansion

The notebook compares BM25 rankings before and after RM3 expansion.

Example:

| Rank | Before Expansion | After Expansion |
| ---: | ---------------: | --------------: |
|    1 |         Doc 8158 |        Doc 8158 |
|    2 |        Doc 13031 |       Doc 13031 |
|    3 |        Doc 18784 |       Doc 26825 |
|    4 |        Doc 26825 |       Doc 18784 |
|    5 |         Doc 8677 |        Doc 8677 |

The rankings of some documents change after the expanded terms are added.

This demonstrates how pseudo-relevance feedback can modify the ranking produced by the original query.

---

# Search Interface

The project also includes a simple **Flask web interface**.

The interface provides a search box where the user can enter a term and retrieve matching tweets.

Technologies used for the interface include:

* Flask
* HTML
* CSS
* Google Colab
* flask-ngrok / Colab port proxy

---

## Simple Search Function

The notebook also implements a basic manual search function:

```python
def sui(df2, que):

    quer = preprocess(que)

    docs_id = []

    # Search processed documents
    ...

    return docs_id
```

Example query:

```text
bad
```

Example returned tweets include:

```text
@tiffanylue i know i was listenin to bad habit earlier
and i started freakin at his part =[
```

and:

```text
@raaaaaaek oh too bad! I hope it gets better.
I've been having sleep issues lately too
```

---

# Evaluation

The notebook includes an evaluation experiment using the **Vaswani Information Retrieval dataset** provided by PyTerrier.

```python
vaswani_dataset = pt.datasets.get_dataset(
    "vaswani"
)
```

The Vaswani index contains:

```text
Number of Documents: 11,429
Number of Terms:      7,756
Number of Postings: 224,573
Number of Tokens:   271,581
```

The notebook also loads the dataset's:

```text
Topics
Qrels
Index
```

using PyTerrier.

---

## Evaluation Metrics

The notebook demonstrates two common IR evaluation metrics:

### Mean Average Precision

Mean Average Precision evaluates how effectively a retrieval system ranks relevant documents across queries.

```text
MAP
```

Higher values indicate better ranking performance.

### Normalized Discounted Cumulative Gain

NDCG evaluates ranking quality while assigning greater importance to relevant results appearing near the top of the ranking.

```text
NDCG
```

Higher values indicate better retrieval ranking.

---

## Evaluation Code

The notebook performs a demonstration search:

```python
retr = pt.BatchRetrieve(
    index2,
    controls={
        "wmodel": "TF_IDF"
    }
)

res = retr.search(
    "mathematical"
)
```

and then evaluates the run using:

```python
pt.Evaluate(
    res,
    qrels
)
```

The saved notebook reports:

```text
MAP  = 0.000004796
NDCG = 0.000228919
```

However, these numbers should **not be treated as the final retrieval performance of the system**.

The manually entered query `"mathematical"` is given query ID `1` by the search call, while Vaswani's relevance judgments for query ID `1` correspond to its own predefined topic.

Therefore, the query and relevance judgments are not correctly aligned.

A valid evaluation should retrieve results for the original Vaswani topics and evaluate those rankings against the corresponding Vaswani qrels.

---

# Correct Evaluation Approach

A stronger version of the project should evaluate the retrieval model using all predefined Vaswani topics.

For example:

```python
tfidf = pt.BatchRetrieve(
    index2,
    wmodel="TF_IDF"
)

bm25 = pt.BatchRetrieve(
    index2,
    wmodel="BM25"
)
```

The complete topic set should then be processed and compared against the provided qrels.

This would allow a valid comparison of:

```text
TF-IDF
vs
BM25
vs
BM25 + RM3
```

using metrics such as:

```text
MAP
NDCG
Precision
Recall
MRR
```

---

# Technologies Used

* Python
* PyTerrier
* Terrier
* NLTK
* Pandas
* NumPy
* Porter Stemmer
* TF-IDF
* BM25
* RM3 Query Expansion
* Flask
* HTML
* CSS
* Google Colab
* Maven
* Terrier PRF

---

# Installation

Install the main dependencies:

```bash
pip install python-terrier
pip install nltk
pip install flask
pip install flask-ngrok
pip install pyngrok
```

The notebook also installs Terrier's pseudo-relevance feedback package.

---

# Running the Project

## 1. Clone the Repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
```

---

## 2. Open the Notebook

Open:

```text
information_retrieval_search_engine.ipynb
```

using Google Colab or Jupyter Notebook.

Google Colab is recommended because the current notebook contains Colab-specific commands.

---

## 3. Add the Dataset

Upload:

```text
tweet_emotions.csv
```

to the notebook environment.

The notebook currently expects the file at:

```text
/content/tweet_emotions.csv
```

---

## 4. Install Dependencies

Run the first installation cells to install:

* PyTerrier
* NLTK
* Terrier PRF
* Flask dependencies

---

## 5. Preprocess the Tweets

Run the preprocessing cells to:

1. Tokenize tweets.
2. Convert text to lowercase.
3. Remove stopwords.
4. Remove URLs.
5. Remove Twitter handles.
6. Remove special characters.
7. Apply Porter stemming.

---

## 6. Create the Index

Run:

```python
index_ref = indexer.index(
    df["processed_text"],
    df["docno"]
)
```

The complete tweet collection will then be indexed by PyTerrier.

---

## 7. Search with TF-IDF

Create the TF-IDF retriever:

```python
tfidf_retr = pt.BatchRetrieve(
    index,
    controls={
        "wmodel": "TF_IDF"
    },
    num_results=10
)
```

and search using:

```python
results = tfidf_retr.search(query)
```

---

## 8. Search with BM25

Run:

```python
bm25 = pt.BatchRetrieve(
    index,
    wmodel="BM25",
    num_results=10
)

results = bm25.search(query)
```

---

## 9. Apply Query Expansion

Create the RM3 query expansion component:

```python
rm3_expander = pt.rewrite.RM3(
    index,
    fb_terms=10,
    fb_docs=100
)
```

Then combine BM25 with RM3:

```python
rm3_qe = bm25 >> rm3_expander
```

---

## 10. Compare Rankings

Compare document rankings:

```text
Before RM3 Expansion
vs
After RM3 Expansion
```

to observe how query expansion changes retrieval results.

---

## 11. Run the Search Interface

Run the Flask section of the notebook and open the generated application URL.

The user can then enter search terms through a graphical interface.

---

# Information Retrieval Concepts Demonstrated

This project demonstrates several fundamental Information Retrieval concepts.

### Document Collection

A collection of 40,000 tweets acts as the searchable corpus.

### Tokenization

Documents and queries are divided into individual terms.

### Stopword Removal

Common words with limited retrieval value are removed.

### Stemming

Related grammatical forms are converted into similar roots.

### Inverted Index

Terms are mapped to the documents in which they occur.

### Query Processing

Queries are processed using the same general normalization pipeline as the documents.

### TF-IDF

Documents are ranked according to term importance.

### BM25

Documents are ranked using a probabilistic relevance model.

### Query Expansion

RM3 adds related terms based on pseudo-relevant retrieved documents.

### Ranking

Documents are ordered according to estimated relevance.

### Evaluation

Retrieval effectiveness can be measured using metrics such as MAP and NDCG.

---

# Limitations

The current implementation has several limitations.

### Evaluation Query Mismatch

The Vaswani evaluation currently searches manually for:

```text
mathematical
```

while evaluating the results against the relevance judgments for Vaswani query ID 1.

Because these are not the same query, the resulting MAP and NDCG values are not a valid measurement of retrieval quality.

### Manual UI Search Uses Only 50 Tweets

The notebook creates:

```python
df2 = df.head(50)
```

for the custom search interface.

Therefore, that part of the application searches only the first 50 tweets instead of the full 40,000-document collection.

### Manual Search Supports Simple Matching

The `sui()` function performs token matching rather than using the ranked PyTerrier retrieval pipeline.

The web interface would be stronger if it called BM25 or BM25 + RM3 directly.

### Stemming Can Affect Manual Comparisons

Porter stemming converts:

```text
empty
```

into:

```text
empti
```

Some notebook cells later check explicitly for `"empty"` in already-stemmed text, which can produce incorrect frequency counts.

### Empty Documents

During indexing, PyTerrier reports:

```text
43 empty documents
```

after preprocessing.

These documents contain no indexable terms after text cleaning.

### Dataset Path

The dataset location is currently hard-coded as:

```text
/content/tweet_emotions.csv
```

which reduces portability outside Google Colab.

---

# Future Improvements

Possible improvements include:

* Correctly evaluating all Vaswani topics
* Comparing TF-IDF, BM25, and BM25 + RM3 quantitatively
* Reporting MAP, NDCG, MRR, Precision, and Recall
* Connecting the Flask interface directly to PyTerrier
* Searching all 40,000 tweets through the web interface
* Displaying ranked BM25 scores
* Displaying the original tweet text with each result
* Adding result pagination
* Adding sentiment filters
* Supporting multi-word queries
* Adding phrase search
* Adding autocomplete
* Adding spell correction
* Adding semantic search using Transformer embeddings
* Comparing lexical search with semantic search
* Adding Sentence-BERT embeddings
* Building a hybrid BM25 + semantic retrieval system
* Removing empty documents before indexing
* Improving Twitter-specific text preprocessing
* Making dataset paths configurable
* Deploying the search engine as a standalone web application

---

# Conclusion

This project demonstrates an end-to-end **Information Retrieval search engine** using PyTerrier.

The system covers the complete traditional retrieval pipeline:

```text
Data Collection
Text Preprocessing
Stopword Removal
Text Cleaning
Stemming
Indexing
Query Processing
TF-IDF Retrieval
BM25 Retrieval
RM3 Query Expansion
Ranking
Search Interface
Evaluation
```

A collection of **40,000 tweets** is indexed using PyTerrier, and users can retrieve relevant documents using TF-IDF and BM25.

The project also demonstrates pseudo-relevance feedback through **RM3 query expansion**, showing how related terms can modify document rankings.

Overall, the project provides practical experience with **Information Retrieval, document indexing, ranking algorithms, query processing, relevance feedback, and search-engine development**.
