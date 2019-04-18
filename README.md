# Empathy Dictionary

This repository contains the datasets, experimental code and results presented in our ACL 2019.


## Dataset
Our dataset comprises 1860 short texts together with ratings for two kinds of empathic states, empathic concern and personal distress. It is, to our knowledge, the first publicly available gold standard for NLP-based empathy prediction. The `csv`-formatted data can be found [here](data/responses/data/messages.csv). For details regarding our annotation methodology please refer to the [paper](https://arxiv.org/pdf/1808.10399.pdf).

## Re-Running the Experiments
We ran our code under Ubuntu 16.04.4. Our `conda` environment is specified in `environment.yaml`. To re-run our experiments, you have to add the root directory of the repository to you python path and setup an environment variable `VECTORS`. Details can be found in the script `activate_project_environment` and `constants.py`. Before running the script, make sure that you have a properely named `conda` environment set-up on your machine (default name is `emnlp18empathy`).

Please note that re-running our code will produce varying results due to racing conditions caused by multi-threading.

The necessary FastText word vectors can be found [here](https://fasttext.cc/docs/en/english-vectors.html).

Once everything is set up, executing `run_experiments.sh` will re-run our cross-validation experiment. The results will be stored in `modeling/main/crossvalidation/results`.

## Paper

You can find our Overleaf document [here](https://www.overleaf.com/9126915636cgjnxkdzmhkv).

## Contact
I am happy to give additional information or get feedback about our work via email: nachmany@seas.upenn.edu
