import json, spacy
from spacy.tokens import DocBin

INPUT_NDJSON = "./data/json/dataset5-corrected.ndjson"
OUT_SPACY = "./training/train.spacy"
LABELS = {"TICKER", "COMPANY"}

def readJson(path):
    examples = []
    with open(path, encoding="utf8") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("data_row", {}).get("row_data")
            entities = []
            projects = row.get("projects", {})
            for project_data in projects.values():
                for label in project_data.get("labels", []):
                    for obj in label.get("annotations", {}).get("objects", []):
                        name = obj.get("name")
                        if name in LABELS:
                            loc = obj.get("location", {})
                            start = loc.get("start")
                            end = loc.get("end")
                            if start is not None and end is not None:
                                entities.append((start, end, name))
            if text and entities:
                examples.append((text, {"entities": entities}))
    return examples

def makeDocBin(examples, nlp, out_path):
    docBin = DocBin()
    for text, annot in examples:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label)
            if span:
                ents.append(span)
        doc.ents = ents
        docBin.add(doc)
    docBin.to_disk(out_path)
    print(f"Final doc count: {len(docBin)}")

if __name__ == "__main__":
    nlp = spacy.blank("en")
    examples = readJson(INPUT_NDJSON)
    print(f"Loaded {len(examples)} examples")
    makeDocBin(examples, nlp, OUT_SPACY)
    print(f"Wrote {OUT_SPACY}")


