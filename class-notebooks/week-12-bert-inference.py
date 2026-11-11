# first, install libraries
# see more on torch installs: https://pytorch.org/get-started/locally/
# see more on transformers installs: https://huggingface.co/docs/transformers/installation
#
# !conda activate 656-ml
# !pip install torch torchvision
# !conda install transformers datasets

# the pipeline function for inference
from transformers import pipeline

# the sentiment analysis task (common NLP task)
sentiment_pipe = pipeline("sentiment-analysis")
out = sentiment_pipe("I love my machine learning class.")

# see that it automatically loads up a version of bert

# see the output
out

# multiple
outs = sentiment_pipe(["I love my machine learning class.", "My machine learning class is meh.", "I am bored in my machine learning class."])
outs

# let's try with a different model
pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")
outs = sentiment_pipe(["I love my machine learning class.", "My machine learning class is meh.", "I am bored in my machine learning class."])
outs

# BERT-like models (also called auto-encoding Transformer models) is
# associate with "Masked language modeling (MLM)" randomly masks some
# tokens in the input and trains the model to predict the original
# tokens based on the surrounding context. This allows the model to
# learn bidirectional context (looking at words both before and after
# the masked word). These models use the bidirectional approach to
# understand context from both directions.

# They’re best suited for tasks that require deep understanding of
# text, such as classification, named entity recognition, and question
# answering.

# BERT is the first model to effectively implement deep
# bidirectionality to learn richer representations of the text by
# attending to words on both sides.

# let's try wiht another model, meant to classify toxic text
toxic = pipeline("text-classification", model="unitary/toxic-bert")
outs = toxic(["I love my machine learning class.", "My machine learning class is meh.", "I am bored in my machine learning class."])
outs
outs = toxic(["I hate rainy days.", "I would die to go there."])
outs

# now we are going pull in a ready-made dataset for inference
from datasets import load_dataset

# movie reviews
imdb = load_dataset("stanfordnlp/imdb")

# check out the dataset object
imdb["test"][0]
imdb["test"][:10]

# There are two fields in this dataset:
## text: the movie review text.
## label: a value that is either 0 for a negative review or 1 for a positive review.

emotions = pipeline("sentiment-analysis", model="SamLowe/roberta-base-go_emotions")
emotions(imdb["test"][0]['text'])
emotions(imdb["test"][1]['text'])

# looping through the first ten
for i in imdb["test"]['text'][:10]:
    out = emotions(i)
    print(i, out)
    print('\n')
    
# now with another dataset from the hub, on political bias
bias_data = load_dataset("cajcodes/political-bias")

# see the datset object
bias_data

for i in bias_data['train']['text'][:10]:
    print(i)
    
for i in bias_data["train"]['text'][:10]:
    out = emotions(i)
    print(i, out)
    print('\n')
for i in bias_data["train"]['text'][:10]:
    out = emotions(i)
    print(i, out)
    print('\n')
