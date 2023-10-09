from collections import Counter
import re

def extract_ngrams(text, n):
    return [text[i:i+n] for i in range(len(text)-n+1)]

def get_top_ngrams(filename, n, top=100):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().lower()
        text = re.sub(r'[^a-z]', '', text)  # remove non-alphabetic characters
        ngrams = extract_ngrams(text, n)
        return [item[0] for item in Counter(ngrams).most_common(top)]

def calculate_score(sentence, ngrams_list, n):
    sentence_ngrams = extract_ngrams(sentence, n)
    return sum([1 for ng in sentence_ngrams if ng in ngrams_list])

def test_accuracy(filename):
    en_bigram = get_top_ngrams('english.txt', 2)
    en_trigram = get_top_ngrams('english.txt', 3)
    de_bigram = get_top_ngrams('german.txt', 2)
    de_trigram = get_top_ngrams('german.txt', 3)
    fr_bigram = get_top_ngrams('french.txt', 2)
    fr_trigram = get_top_ngrams('french.txt', 3)

    correct_predictions = 0
    total_sentences = 0

    with open(filename, 'r') as f:
        for line in f:
            sentence, actual_language = line.strip().rsplit(',', 1)
            sentence = sentence.lower()
            sentence = re.sub(r'[^a-z]', '', sentence)  # remove non-alphabetic characters

            en_score = calculate_score(sentence, en_bigram, 2) + calculate_score(sentence, en_trigram, 3)
            de_score = calculate_score(sentence, de_bigram, 2) + calculate_score(sentence, de_trigram, 3)
            fr_score = calculate_score(sentence, fr_bigram, 2) + calculate_score(sentence, fr_trigram, 3)

            scores = {'en': en_score, 'de': de_score, 'fr': fr_score}
            detected_language = max(scores, key=scores.get)

            if detected_language == actual_language:
                correct_predictions += 1
            total_sentences += 1

    accuracy = correct_predictions / total_sentences
    print(f"Accuracy: {accuracy:.2f}")

def main():
    test_accuracy('sentences.txt')
    en_bigram = get_top_ngrams('english.txt', 2)
    en_trigram = get_top_ngrams('english.txt', 3)
    de_bigram = get_top_ngrams('german.txt', 2)
    de_trigram = get_top_ngrams('german.txt', 3)
    fr_bigram = get_top_ngrams('french.txt', 2)
    fr_trigram = get_top_ngrams('french.txt', 3)

    while True:
        sentence = input("Enter a sentence (or 'exit' to quit): ").lower()
        if sentence == 'exit':
            break

        sentence = re.sub(r'[^a-z]', '', sentence)  # remove non-alphabetic characters

        en_score = calculate_score(sentence, en_bigram, 2) + calculate_score(sentence, en_trigram, 3)
        de_score = calculate_score(sentence, de_bigram, 2) + calculate_score(sentence, de_trigram, 3)
        fr_score = calculate_score(sentence, fr_bigram, 2) + calculate_score(sentence, fr_trigram, 3)

        scores = {'English': en_score, 'German': de_score, 'French': fr_score}
        detected_language = max(scores, key=scores.get)

        print(f"The detected language is: {detected_language}")
        print(scores)

if __name__ == "__main__":
    main()
