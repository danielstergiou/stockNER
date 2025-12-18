import re, csv, os, uuid, spacy
from dotenv import load_dotenv 

from typing import List, Set, Dict, Any
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc

from labelbox import Client
from labelbox.schema.ontology_kind import OntologyKind
from labelbox.schema.media_type import MediaType
from labelbox.schema.annotation_import import LabelImport
import labelbox.data.annotation_types as lb_types


stockData = "./data/stockslist.csv"
etfData = "./data/etfs.csv"
commodityData = "./data/commodities.csv"
textData = "./data/dataset300.txt"

RE_CASHTAG  = re.compile(r'\$([A-Za-z]{1,4})\b')
RE_EXCHANGE = re.compile(r'\b(?:NYSE|NASDAQ|AMEX):([A-Za-z]{1,4}(?:\.[AB])?)\b')

FIN_KEYWORDS = {
    "stock","shares","earnings","volume","call","put","market",
    "option","dividend","buy","bought","sell","sold","position"
}
STOP_TOKENS = {
    "IT","ON","FOR","AND","ARE","BE","ALL","IN","TO","OF","THE","LL",
    "YOU","NOW","HAS","HAVE","IVE","I","ME"
}

load_dotenv()
LABELBOX_API_KEY = os.getenv('LABELBOX_API')
LABELBOX_PROJECT = "StockNER"
LABELBOX_DATASET = "StockNER-Dataset300"
ONTOLOGY_NAME = "StockNER-ontology"

def load_data(stockData: str, etfData: str):
    ticker_to_name, name_to_tickers = {}, {}
    ticker_to_sector, ticker_to_industry = {}, {}
    for path in (stockData, etfData):
        with open(path, newline='', encoding='utf8') as f:
            for row in csv.DictReader(f):
                t = row['Ticker'].strip().upper()
                n = row['Name'].strip()
                ticker_to_name[t] = n
                ticker_to_sector[t]   = row.get('Sector','').strip()
                ticker_to_industry[t] = row.get('Industry','').strip()
                for alias in {n, row.get('OtherName','').strip()}:
                    if alias:
                        name_to_tickers.setdefault(alias.lower(), set()).add(t)
    return ticker_to_name, name_to_tickers, ticker_to_sector, ticker_to_industry

def build_nlp_components(name_to_tickers: Dict[str,Set[str]], all_tickers: List[str]):
    nlp = spacy.load("en_core_web_trf", disable=["ner"])
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    ticker_patterns = [nlp.make_doc(t.lower()) for t in all_tickers]
    matcher.add("TICKER", ticker_patterns)

    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents":True}, before="parser")
    patterns = []
    for alias_lower, ticks in name_to_tickers.items():
        toks = alias_lower.split()
        if not toks:
            continue
        patterns.append({"label":"COMPANY",
                         "pattern":[{"LOWER": tok} for tok in toks]})
    ruler.add_patterns(patterns)
    return nlp, matcher

def extract_candidates(text: str, doc: Doc,
                       matcher: PhraseMatcher,
                       name_to_tickers: Dict[str,Set[str]]
                       ) -> List[Dict[str,Any]]:
    cands = []
    for m in RE_CASHTAG.finditer(text):
        cands.append({"span":(m.start(1),m.end(1)),
                      "text":m.group(1).upper(),"signal":"cashtag"})
    for m in RE_EXCHANGE.finditer(text):
        cands.append({"span":(m.start(1),m.end(1)),
                      "text":m.group(1).upper(),"signal":"exchange"})
    for _, start, end in matcher(doc):
        span = doc[start:end]
        cands.append({"span":(span.start_char,span.end_char),
                      "text":span.text.upper(),"signal":"phrase"})
    for ent in doc.ents:
        if ent.label_=="COMPANY":
            for t in name_to_tickers.get(ent.text.lower(),[]):
                cands.append({"span":(ent.start_char,ent.end_char),
                              "text":t,"signal":"entity"})
    for tok in doc:
        key = tok.text.strip(".,$").lower()
        for t in name_to_tickers.get(key,[]):
            cands.append({"span":(tok.idx, tok.idx+len(tok)),
                          "text":t,"signal":"alias"})
    seen=set(); uniq=[]
    for c in cands:
        key=(c["span"],c["text"],c["signal"])
        if key not in seen:
            seen.add(key); uniq.append(c)
    return uniq

def featurize(cand: Dict[str,Any], doc: Doc,
              ticker_to_sector: Dict[str,str],
              ticker_to_industry: Dict[str,str]) -> Dict[str,float]:
    feats={}
    for sig in ("cashtag","exchange","phrase","entity","alias"):
        feats[f"sig_{sig}"] = 1.0 if cand["signal"]==sig else 0.0
    txt = cand["text"]
    feats["len"]       = float(len(txt))
    feats["all_upper"] = float(txt.isupper())
    feats["has_dot"]   = float("." in txt)
    idx=None
    for i,t in enumerate(doc):
        if t.idx==cand["span"][0]:
            idx=i; break
    window=[]
    if idx is not None:
        window=[w.text.lower() for w in doc[max(0,idx-3):idx+4]]
    feats["ctx_fin_kw"]    = sum(w in FIN_KEYWORDS for w in window)
    sector = ticker_to_sector.get(txt,"").lower().split()
    industry = ticker_to_industry.get(txt,"").lower().split()
    feats["ctx_sector"]   = float(any(w in window for w in sector))
    feats["ctx_industry"] = float(any(w in window for w in industry))
    return feats

def split_long_text(text, max_length=400):
    words = text.split()
    chunks = []
    while len(words) > max_length:
        chunks.append(' '.join(words[:max_length]))
        words = words[max_length:]
    if words:
        chunks.append(' '.join(words))
    return chunks

class HeuristicClassifier:
    def predict_proba(self, feats: List[Dict[str,float]], cands: List[Dict[str,Any]]) -> List[float]:
        probs=[]
        for f,c in zip(feats,cands):
            txt = c["text"]
            l   = len(txt)
            p=0.0
            if f["sig_cashtag"] or f["sig_exchange"]:
                p=1.0
            else:
                if l<=2:
                    if f["ctx_fin_kw"]>0:
                        p=0.9
                else:
                    if f["sig_entity"]:
                        p=0.9
                    elif (f["sig_phrase"] or f["sig_alias"]) and (f["ctx_fin_kw"]>0 or f["ctx_sector"]>0 or f["ctx_industry"]>0):
                        p=0.8
            if txt in STOP_TOKENS:
                p=0.0
            probs.append(p)
        return probs

class FullExtractor:
    def __init__(self, stockData, etfData):
        self.t2n, self.n2t, self.t2s, self.t2i = load_data(stockData, etfData)
        self.nlp, self.matcher = build_nlp_components(self.n2t, list(self.t2n.keys()))
        self.clf = HeuristicClassifier()

    def extract(self, text: str) -> Set[str]:
        doc = self.nlp(text)
        cands = extract_candidates(text, doc, self.matcher, self.n2t)
        feats = [featurize(c, doc, self.t2s, self.t2i) for c in cands]
        probs = self.clf.predict_proba(feats, cands)
        return {
            c["text"] for c,p in zip(cands,probs)
            if p >= 0.5}

    def annotate_line(self, text: str) -> List[str]:
        return sorted(self.extract(text))

def labelbox_bootstrap(stockData, etfData, textData):
    client = Client(api_key=LABELBOX_API_KEY)
    print("Connected to Labelbox:", client.get_user().email)

    ds = next((d for d in client.get_datasets() if d.name == LABELBOX_DATASET), None)
    if not ds:
        ds = client.create_dataset(name=LABELBOX_DATASET)
        print("Created dataset", LABELBOX_DATASET)
    else:
        print("Found dataset", LABELBOX_DATASET)

    pr = next((p for p in client.get_projects() if p.name == LABELBOX_PROJECT), None)
    if not pr:
        pr = client.create_project(name=LABELBOX_PROJECT, media_type=MediaType.Text, dataset_ids=[ds.uid])
        print("Created project", LABELBOX_PROJECT)
    else:
        print("Found project", LABELBOX_PROJECT)

    ontology_json = {
        "tools": [
            {"tool": "named-entity", "name": "TICKER", "color": "#1CE6FF"},
            {"tool": "named-entity", "name": "COMPANY", "color": "#1565C0"},
            {"tool": "named-entity", "name": "COMMODITY", "color": "#FF4A46"},
            {"tool": "named-entity", "name": "INDUSTRY", "color": "#008941"},
        ]
    }
    ontology = next((o for o in client.get_ontologies(name_contains=ONTOLOGY_NAME) if o.name == ONTOLOGY_NAME), None)
    if not ontology:
        ontology = client.create_ontology(
            name=ONTOLOGY_NAME,
            normalized=ontology_json,
            media_type=MediaType.Text,
            ontology_kind=OntologyKind.ResponseCreation
        )
        print("Created ontology", ONTOLOGY_NAME)
    else:
        print("Found ontology", ONTOLOGY_NAME)

    for tool in ontology.tools():
        print("Ontology tool:", tool.name, tool.feature_schema_id)
    tid = next(tool.feature_schema_id for tool in ontology.tools() if tool.name == "TICKER")
    cid = next(tool.feature_schema_id for tool in ontology.tools() if tool.name == "COMPANY")

    with open(textData, encoding="utf8") as f:
        lines = [line.strip() for line in f if line.strip()]
    uploaded = {dr.external_id for dr in ds.data_rows()}
    batch = []
    for i, line in enumerate(lines, 1):
        if str(i) not in uploaded:
            batch.append({"row_data": line, "global_key": str(i)})
    if batch:
        ds.create_data_rows(batch)
        print(f"Uploaded {len(batch)} new data rows.")
    else:
        print("All data rows already uploaded.")

    ext = FullExtractor(stockData, etfData)
    MAX_LEN = 64000
    MAX_ANN = 150
    valid_labels = []

    for i, line in enumerate(lines, 1):
        if len(line) > MAX_LEN:
            print(f"⚠️ Skipping row {i} (too long: {len(line)} characters)")
            continue

        anns = []
        for chunk in split_long_text(line, 400):
            doc = ext.nlp(chunk)
            cands = extract_candidates(chunk, doc, ext.matcher, ext.n2t)
            feats = [featurize(c, doc, ext.t2s, ext.t2i) for c in cands]
            probs = ext.clf.predict_proba(feats, cands)

            for c, p in zip(cands, probs):
                if p < 0.5:
                    continue
                start, end = c["span"]
                if not isinstance(start, int) or not isinstance(end, int):
                    continue
                if start >= end or end > len(line):
                    continue
                label = "TICKER" if c["signal"] != "entity" else "COMPANY"
                anns.append(
                    lb_types.ObjectAnnotation(
                        name=label,
                        value=lb_types.TextEntity(start=start, end=end)
                    )
                )

        if len(anns) > MAX_ANN:
            print(f"⚠️ Row {i} has {len(anns)} annotations, truncating to {MAX_ANN}")
            anns = anns[:MAX_ANN]

        valid_labels.append(
            lb_types.Label(
                data={"global_key": str(i)},
                annotations=anns
            )
        )

    print(f"Prepared predictions for {len(valid_labels)} rows.")
    
    total_anns = sum(len(lbl.annotations) for lbl in valid_labels)
    print(f"Total annotations to import: {total_anns}")

    if valid_labels:
        import_job = LabelImport.create_from_objects(
            client=client,
            project_id=pr.uid,
            name="NER prelabels " + str(uuid.uuid4())[:8],
            labels=valid_labels
        )
        import_job.wait_till_done()
        print("Import State:", import_job.state)

        try:
            print("Error Summary:", import_job.errors)
        except Exception as e:
            print("⚠️ Could not fetch errors:", e)

        try:
            print("Failed data rows:", import_job.failed_data_rows)
        except Exception as e:
            print("⚠️ Could not fetch failed data rows:", e)

        print("Error Details URL:", getattr(import_job, "error_details_url", "N/A"))
    else:
        print("No predictions to upload.")

if __name__=="__main__":
    labelbox_bootstrap(stockData, etfData, textData)