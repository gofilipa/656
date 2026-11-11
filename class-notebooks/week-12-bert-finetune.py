## first install libraries, uncomment the line below
# !pip install transformers datasets evaluate accelerate
## if error with loading trl:
# !pip install git+https://github.com/huggingface/trl.git

########################
### LOADING THE DATA ###
########################

# load the movie reviews
from datasets import load_dataset

imdb = load_dataset("stanfordnlp/imdb")

# examine the data, which has two fields:
#    text: the movie review text.
#    label: a value that is either 0 for a negative review or 1 for a positive review.
imdb["test"][0]

##########################
### PREPARING THE DATA ###
##########################

# now we will tokenize the data

# load distilbert tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Randomly sample 2,000 training examples
train_dataset = imdb["train"].shuffle(seed=42).select(range(2000))

# define our processing function for tokenization
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

# now, tokenize
tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True
)

# load our data collator, for padding the data to equal lengths
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# finally, create a map of the expected ids to their labels with id2label and label2id:
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

################
### TRAINING ###
################

# load the training libraries
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# load the training model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

# define our training arguments
training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
)

# define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    processing_class=tokenizer,
    data_collator=data_collator,
)

# train the model!
trainer.train()

# save the model locally
trainer.save_model('./bert-finetuned')

#################
### INFERENCE ###
#################

# create a text for testing it out (whatever you want can go in the quotes)
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

# load the pipeline for inference
from transformers import pipeline

# define our classifier with the newest model
classifier = pipeline("sentiment-analysis", model="./bert-finetuned")

# run the classifer
classifier(text)

# try out different prompts
text1 = "What a boring film. Waste of time."
classifier(text1)

texts = ['Loved it', 'It was meh', 'I have seen better', 'It was too far long', 'It stimulated me', 'Visuals', 'Dialogue', 'Character development']
for i in texts:
    print(classifier(i))
    
# see the entire tutorial here (with evaluation): https://huggingface.co/docs/transformers/tasks/sequence_classification
