The predictive lexica were created with DLATK's `--regression_to_lexicon`
switch (https://dlatk.wwbp.org/fwinterface/fwflag_regression_to_lexicon.html),
which extracts the coefficients from a regression model with linear
coefficients and the `--no_standardize` flag (in order for the lexicon
extraction equation/model to make sense). The weights are unbounded, may be
negative, and include an extra row for the intercept.

Note: Since the call to create a lexica was causing errors on the DLATK
server, I ran the code locally with Docker
(https://dlatk.wwbp.org/tutorials/tut_docker.html).

Paper using same approach to create lexica:
http://wwbp.org/papers/emnlp2014_developingLexica.pdf

GH issue with more description on how to create lexica (notes that you are
limited to only using 1gram features): https://github.com/dlatk/dlatk/issues/5

P.S. The sql tables were exported to CSV using a bash script in the
tools/usefulScripts directory in WWBP.
