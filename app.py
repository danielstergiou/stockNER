import spacy
from spacy.training import Example
from spacy.tokens import DocBin
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random
import json

# Load annotations from the JSON file
def load_annotations(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Process the annotations
def prepare_data(json_file):
    annotations = load_annotations(json_file)
    data = []
    for item in annotations["annotations"]:
        # Ensure each annotation is structured correctly
        if isinstance(item, list) and len(item) == 2:
            text, entity_data = item
            if "entities" in entity_data and isinstance(entity_data["entities"], list):
                try:
                    entities = [
                        tuple(entity) for entity in entity_data["entities"]
                        if len(entity) == 3
                    ]
                    data.append((text, {"entities": entities}))
                except Exception as e:
                    print(f"Skipping malformed entity: {entity_data['entities']}, Error: {e}")
    return data

# Load and split the data
data = prepare_data("annotations.json")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)

print(f"Training Set: {len(train_data)}, Validation Set: {len(val_data)}, Test Set: {len(test_data)}")

# Initialize the blank NLP model
nlp = spacy.load("en_core_web_sm")

# Add NER pipeline
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Add labels to the NER pipeline
for _, annotations in train_data:
    for _, _, label in annotations["entities"]:
        ner.add_label(label)

# Train the NER model
def train_ner_model(nlp, train_data, epochs=50):
    optimizer = nlp.resume_training()  # Use resume_training for pretrained models
    for epoch in range(epochs):
        random.shuffle(train_data)
        losses = {}
        for text, annotations in train_data:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.1, losses=losses)  # Lower dropout rate
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {losses['ner']:.4f}")

# Train the model
train_ner_model(nlp, train_data, epochs=50)

def evaluate_model(nlp, data):
    y_true = []
    y_pred = []

    for text, annotations in data:
        # Get gold entities
        gold_entities = [(start, end, label) for start, end, label in annotations["entities"]]

        # Get predicted entities
        doc = nlp(text)
        pred_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

        # Align predictions with true labels
        gold_labels = [label for _, _, label in gold_entities]
        predicted_labels = []

        for start, end, label in gold_entities:
            # Check if the gold entity exists in predictions
            match = [pred for pred in pred_entities if pred[0] == start and pred[1] == end]
            if match:
                predicted_labels.append(match[0][2])  # Append the predicted label
            else:
                predicted_labels.append("O")  # "O" for no prediction

        # Extend the results
        y_true.extend(gold_labels)
        y_pred.extend(predicted_labels)

    # Print the classification report
    print(classification_report(y_true, y_pred, zero_division=0))


# Evaluate on validation and test sets
print("Validation Set Evaluation:")
evaluate_model(nlp, val_data)

print("Test Set Evaluation:")
evaluate_model(nlp, test_data)