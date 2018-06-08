# Toxic Comment Classifier

## Data Description:
You are provided with a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:

toxic
severe_toxic
obscene
threat
insult
identity_hate
You must create a model which predicts a probability of each type of toxicity for each comment.

File descriptions
train.csv - the training set, contains comments with their binary labels
test.csv - the test set, you must predict the toxicity probabilities for these comments. To deter hand labeling, the test set contains some comments which are not included in scoring.
sample_submission.csv - a sample submission file in the correct format
test_labels.csv - labels for the test data; value of -1 indicates it was not used for scoring; (Note: file added after competition close!)

## Approach :


Feature engineering:

I've broadly classified my feature engineering ideas into the following three groups
Direct features:

Features which are a directly due to words/content.We would be exploring the following techniques

    Word frequency features
        Count features
        Bigrams
        Trigrams
    Vector distance mapping of words (Eg: Word2Vec)
    Sentiment scores

Indirect features:

Some more experimental features.

    count of sentences
    count of words
    count of unique words
    count of letters
    count of punctuations
    count of uppercase words/letters
    count of stop words
    Avg length of each word

Leaky features:

From the example, we know that the comments contain identifier information (eg: IP, username,etc.). We can create features out of them but, it will certainly lead to overfitting to this specific Wikipedia use-case.

    toxic IP scores
    toxic users

Note: Creating the indirect and leaky features first. There are two reasons for this,

    Count features(Direct features) are useful only if they are created from a clean corpus
    Also the indirect features help compensate for the loss of information when cleaning the dataset


Direct features:
1)Count based features(for unigrams):

Lets create some features based on frequency distribution of the words. Initially lets consider taking words one at a time (ie) Unigrams

Python's SKlearn provides 3 ways of creating count features.All three of them first create a vocabulary(dictionary) of words and then create a sparse matrix of word counts for the words in the sentence that are present in the dictionary. A brief description of them:

    CountVectorizer
        Creates a matrix with frequency counts of each word in the text corpus
    TF-IDF Vectorizer
        TF - Term Frequency -- Count of the words(Terms) in the text corpus (same of Count Vect)
        IDF - Inverse Document Frequency -- Penalizes words that are too frequent. We can think of this as regularization
    HashingVectorizer
        Creates a hashmap(word to number mapping based on hashing technique) instead of a dictionary for vocabulary
        This enables it to be more scalable and faster for larger text coprus
        Can be parallelized across multiple threads

Using TF-IDF here. Note: Using the concatenated dataframe "merge" which contains both text from train and test dataset to ensure that the vocabulary that we create does not missout on the words that are unique to testset.
Topic Modeling:

Topic modeling can be a useful tool to summarize the context of a huge corpus(text) by guessing what the "Topic" or the general theme of the sentence.

This can also be used as inputs to our classifier if they can identify patterns or "Topics" that indicate toxicity.

Let's find out!

The steps followed in this kernel:

    Preprocessing (Tokenization using gensim's simple_preprocess)
    Cleaning
        Stop word removal
        Bigram collation
        Lemmatization
    Creation of dictionary (list of all words in the cleaned text)
    Topic modeling using LDA
    Visualization with pyLDAviz
    Convert topics to sparse vectors
    Feed sparse vectors to the model

