# Program to generate data using NLTK to test the language identifier. The program generates 300 examples of (en,de,fr) and saves them in a txt file
import nltk
from random import shuffle

# Download the necessary corpora
nltk.download('reuters')
nltk.download('europarl_raw')

from nltk.corpus import reuters, europarl_raw


#print(reuters.categories())

# Extract sentences (using 'acq' as an example category from reuters)
english_sentences = reuters.sents(categories='acq')[:100]
german_sentences = europarl_raw.german.sents()[:100]
french_sentences = europarl_raw.french.sents()[:100]

# Convert sentences to desired format
formatted_english = [" ".join(sent) + ",en" for sent in english_sentences]
formatted_german = [" ".join(sent) + ",de" for sent in german_sentences]
formatted_french = [" ".join(sent) + ",fr" for sent in french_sentences]

# Combine all sentences
all_sentences = formatted_english + formatted_german + formatted_french

# Shuffle the sentences to mix them up
shuffle(all_sentences)

# Write the sentences to a file
with open('sentences.txt', 'w') as f:
    for sentence in all_sentences:
        f.write(sentence + '\n')
