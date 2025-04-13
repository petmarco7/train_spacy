import spacy
import random
from spacy.training.example import Example

TRAIN_DATA = [
    ("I love eating pizza and pasta.", {"entities": [(14, 19, "FOOD"), (24, 29, "FOOD")]}),
    ("The chef's special is sushi.", {"entities": [(22, 27, "FOOD")]}),
    ("Can I have a cheeseburger and fries?", {"entities": [(13, 25, "FOOD"), (30, 35, "FOOD")]}),
    ("I'd like to order a burrito bowl.", {"entities": [(20, 32, "FOOD")]}),
]

def print_entities(train_data):
    for text, annotations in train_data:
        for ent in annotations.get("entities"):
            print(text[ent[0]:ent[1]], ent[2])

print_entities(TRAIN_DATA)

nlp = spacy.load("en_core_web_lg")

test_text = "I want to order pasta with pesto. I also want a salad."

print("Before training:")
doc = nlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.label_)

print()
print()

print("Training the model...")
def train_ner(nlp, train_data, n_iter):
    # Add the NER component if not already in the pipeline
    if "ner" not in nlp.pipe_names:
        nlp.add_pipe("ner", last=True)

    ner = nlp.get_pipe("ner")

    # Add the labels to the NER component
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Disable other components in the pipeline
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    # Train the NER model
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            
            for text, annotations in train_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)

            print("Iteration {} Loss: {}".format(itn, losses))

    return nlp

nlp = train_ner(nlp, TRAIN_DATA, 30)

print()
print()

print("After training:")
doc = nlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.label_)