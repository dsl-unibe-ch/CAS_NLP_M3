import string

def tokenize_text(text):
    # Rule 1: Replace every punctuation with whitespace then the punctuation
    for punct in string.punctuation:
        text = text.replace(punct, f' {punct}')
    
    # Rule 2: Replace every whitespace with new line character
    text = text.replace(' ', '\n')
    
    return text

def main():
    # Read the text from the file
    with open('english.txt', 'r') as file:
        text = file.read()
    
    # Tokenize the text
    tokenized_text = tokenize_text(text)
    
    # Write the tokenized text to a new file
    with open('english_tokenized.txt', 'w') as file:
        file.write(tokenized_text)

if __name__ == "__main__":
    main()
