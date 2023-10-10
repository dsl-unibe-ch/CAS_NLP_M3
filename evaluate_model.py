import spacy
from spacy.tokens import DocBin
from spacy.training import Example

#load the model
nlp = spacy.load("path/to/model") # replace with actual path to model

#load the evaluation data
eval_data_file = "path/to/dev.spacy" # replace with actual path to dev
doc_bin = DocBin().from_disk(eval_data_file)
eval_docs = list(doc_bin.get_docs(nlp.vocab))


#Prepare the evaluation data as a list of Example objects
examples = []

for gold_doc in eval_docs:
    text = gold_doc.text
    pred_doc = nlp(text)
    example = Example(gold_doc, pred_doc)
    examples.append(example)


scores = nlp.evaluate(examples)


print("Token-level scores:", scores["token_acc"])
print("Entity-level scores:")
print(" - Precision:", scores["ents_p"])
print(" - Recall:", scores["ents_r"])
print(" - F1-score:", scores["ents_f"])