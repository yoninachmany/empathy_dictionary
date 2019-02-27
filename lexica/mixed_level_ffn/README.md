See the modified version of the code in the `yoni-empathy-dictionary` branch.

A FFN trained on all the messages predicted the ratings for all the tokens.
Note that the NLTK tokenizer is used here (for consistency within the project),
which produces a slightly different number of words than DLATK's Happier Fun
Tokenizer (9,357 lowercase tokens vs. 10,097 for empathy, 15,399 vs. 17,1314
for vad).

