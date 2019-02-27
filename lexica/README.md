**Outline of the paper**: We want to create an empathy dictionary because
psychologists found this to be difficult to do from direct subject responses.
Instead we want to infer word level empathy ratings using available text level
ratings. From a more abstract point of view, this task can be paraphrased as
deriving word level ratings from higher level supervision. A small number of
papers has addressed special cases of this problem in isolation, however a
systematic comparison of techniques is missing. To make a justifed choice for
a particular method, we will first conduct a systematic comparison of existing
methods. Since, again, no empathy dictionary exist, we will use emotion
ratings (words and sentence level) as best possible surrogate (data is
available and emotion is strongly related to empathy).

**Evaluation**:

* I would suggest to evaluate the methods by trying to derive word level
ratings for Valence, Arousal and Dominance from EmoBank
(https://github.com/JULIELab/EmoBank) and comparing those against the gold
ratings in Warriner's emotion lexicon (http://crr.ugent.be/archives/1003 ).
* It would be good to at least partially read the respective papers (the
Behavior Research Methods and the EACL ones, not the LAW-workshop paper) to
get some background on the datasets and VAD in general. References are given
on the websites. Please get in touch if you feel unsure about technical details.
* Reasons for choosing those datasets: Warriner's lexicon I think is a rather
obvious choice. It's large and well known. EmoBank would be a good fit because
it focuses on domain-balanced standard English and hence should go well
together with standard emotion lexicons. The Stanford Sentiment Treebank might
 be another good choice. But then, the resulting word ratings would be biased
 towards the movie review domain. Also, it only covers valence/polarity. So I
 think you should focus on EmoBank first.
* 10-fold crossvalidation