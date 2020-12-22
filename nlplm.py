try: # Use the default NLTK tokenizer.
    from nltk import word_tokenize, sent_tokenize 
    # Testing whether it works. 
    # Sometimes it doesn't work on some machines because of setup issues.
    word_tokenize(sent_tokenize("This is a foobar sentence. Yes it is.")[0])
except: # Use a naive sentence tokenizer and toktok.
    import re
    from nltk.tokenize import ToktokTokenizer
    # See https://stackoverflow.com/a/25736515/610569
    sent_tokenize = lambda x: re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', x)
    # Use the toktok tokenizer that requires no dependencies.
    toktok = ToktokTokenizer()
    word_tokenize = word_tokenize = toktok.tokenize


import os
import requests
import io #codecs


if os.path.isfile('corpus.txt'):
    with io.open('corpus.txt', encoding='utf8') as fin:
        text = fin.read()
else:
    print("error corpus file not found")       
# Tokenize the text.
tokenized_text = [list(map(str.lower, word_tokenize(sent))) 
                  for sent in sent_tokenize(text)]

# Preprocess the tokenized text for 3-grams language modelling
from nltk.lm.preprocessing import padded_everygram_pipeline
n = 2
train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)

from nltk.lm import MLE
model = MLE(n) 
model.fit(train_data, padded_sents)
print(model.vocab)

word = input("Enter the word: ")
history = input("Enter the history word: ")
print ("The MLE of P(word|history) is ", model.score(word, history.split()))