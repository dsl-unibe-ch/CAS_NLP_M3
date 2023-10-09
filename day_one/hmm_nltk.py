
# Simple script to train Hidden Markov Model for Part of Speech tagging using NLTK


#import the models
import nltk
from nltk import HiddenMarkovModelTagger as hmm # do not use nltk.tag.hmm
from nltk.corpus import treebank
import warnings
import dill


warnings.filterwarnings('ignore')


# Download the data. Run only once
#nltk.download('treebank')


# Prepare the data. We'll use the Penn Treebank which is an English Corpus that includes pos tagging. For information on the tagset: https://www.sketchengine.eu/penn-treebank-tagset/
# We split the data into training and testing. Try to change the data size and experiment with the accuracy change.

print(f'The number of tagged examples in the dataset is: {len(treebank.tagged_sents())}')

train_data = treebank.tagged_sents()[:2000]
test_data = treebank.tagged_sents()[-500:]

print(train_data[0])

# Extracting unique tags from train_data
unique_tags = set(tag for sent in train_data for _, tag in sent)

print(unique_tags)


# Define the trainer and train the model
tagger = hmm.train(train_data, verbose=True)



# Evaluate the model's accuracy on the test data
accuracy = tagger.accuracy(test_data)
print(f"Accuracy: {accuracy:.2f}")

#Generate true tags list and model prediction to get more detailed stats on where the model performed better and where it didn't perform so well


# Generate Predictions
true_tags = [tag for sent in test_data for _, tag in sent]
predicted_tags = [tag for sent in tagger.tag_sents([[word for word, _ in sent] for sent in test_data]) for _, tag in sent]


# Compute accuracy for each label
labels = list(set(true_tags))
for label in labels:
    correct_predictions = sum(1 for t, p in zip(true_tags, predicted_tags) if t == label and p == label)
    total_predictions = sum(1 for t in true_tags if t == label)
    wrong_predictions = total_predictions - correct_predictions
    label_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print(f"Label: {label}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Wrong Predictions: {wrong_predictions}")
    print(f"Accuracy: {label_accuracy:.2f}\n")


# Save the trained model to a file
with open('hmm_tagger.pkl', 'wb') as f:
    dill.dump(tagger, f)


# Load the trained model from the file
with open('hmm_tagger.pkl', 'rb') as f:
    loaded_tagger = dill.load(f)


sentence = 'I took the train from Zurich to Italy last night'

tokens = nltk.word_tokenize(sentence)

# Tag the tokenized sentence
tagged_sentence = loaded_tagger.tag(tokens)

print(tagged_sentence)


