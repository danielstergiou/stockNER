import re, csv, os, json, spacy
from typing import List, Dict, Any
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
STOCK_CSV        = ("./data/stocklist.csv")
ETF_CSV          = ("./data/etfs.csv")
COMMODITY_CSV    = ("./data/commodities.csv")
INDUSTRY_CSV     = ("./data/sectorIndustries.csv")
DERIVATIVE_CSV   = ("./data/derivatives.csv")
MACRO_CSV        = ("./data/macro.csv")

FINANCIAL_KEYWORDS = {
    "stock", "shares", "price", "market", "trade", "buy", "sell", "invest", "investment", 
    "ticker", "ipo", "dividend", "portfolio", "equity", "etf", "earnings", "profit", "loss",
    "long", "short", "call", "put", "bullish", "bearish", "hodl", "yolo"
}

STOPWORDS = {
    "a", "i", "has", "is", "am", "on", "of", "for", "do", "did", "so", "the", "it", "was", "in", "to"
}

RE_CASHTAG  = re.compile(r'\$([A-Za-z]{1,5})\b')
RE_EXCHANGE = re.compile(r'\b(?:NYSE|NASDAQ|AMEX):([A-Za-z]{1,5}(?:\.[AB])?)\b')

# ─── LOADERS ─────────────────────────────────────────────────────────────────────
def load_instruments(*paths: str):
    ticker_to_name, name_to_tickers = {}, {}
    for path in paths:
        with open(path, newline='', encoding='utf8') as f:
            for row in csv.DictReader(f):
                t = row['Ticker'].strip().upper()
                n = row.get('Name','').strip()
                other = row.get('OtherName') or ''
                ticker_to_name[t] = n
                for alias in {n, other.strip()}:
                    if alias:
                        name_to_tickers.setdefault(alias.lower(), set()).add(t)
    return ticker_to_name, name_to_tickers

def load_commodities(path: str):
    name_to_comm = {}
    with open(path, encoding='utf8') as f:
        for row in csv.DictReader(f):
            canon = row['Name'].strip()
            for alias in {canon} | set((row.get('OtherName') or '').split('|')):
                alias = alias.strip()
                if alias:
                    name_to_comm[alias.lower()] = canon
    return name_to_comm

def load_industries(path: str):
    alias_to_ind = {}
    with open(path, encoding='utf8') as f:
        for row in csv.DictReader(f):
            ind = row['Industry'].strip()
            sub = row.get('Sub-Industry','').strip()
            for alias in {ind, sub}:
                if alias:
                    alias_to_ind[alias.lower()] = ind
    return alias_to_ind

def load_derivatives(path: str):
    name_to_deriv = {}
    with open(path, encoding='utf8') as f:
        for row in csv.DictReader(f):
            canon = row['Derivative'].strip()
            for alias in {canon} | set(row.get('OtherNames','').split('|')):
                alias = alias.strip()
                if alias:
                    name_to_deriv[alias.lower()] = canon
    return name_to_deriv

def load_macros(path: str):
    name_to_macro = {}
    with open(path, encoding='utf8') as f:
        for row in csv.DictReader(f):
            canon = row['Name'].strip()
            for alias in {canon} | set((row.get('OtherName') or '').split('|')):
                alias = alias.strip()
                if alias:
                    name_to_macro[alias.lower()] = canon
    return name_to_macro

# ─── MATCHER BUILDER ─────────────────────────────────────────────────────────────
def build_matchers(nlp, dicts):
    matchers = {}
    for label, alias_dict in dicts.items():
        matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
        patterns = [nlp.make_doc(alias) for alias in alias_dict.keys()]
        matcher.add(label, patterns)
        matchers[label] = matcher
    return matchers

# ─── EXTRACTOR ────────────────────────────────────────────────────────────────────
def extract_candidates(text: str, doc: Doc, matchers: Dict[str, PhraseMatcher], tickers: set, name_to_tickers: dict, regex_tickers: bool = True) -> List[Dict[str, Any]]:
    cands = []
    if regex_tickers:
        for m in RE_CASHTAG.finditer(text):
            t = m.group(1)
            if t.upper() in tickers and t.lower() not in STOPWORDS:
                cands.append({'span': (m.start(1), m.end(1)), 'text': t, 'label': 'TICKER'})
        for m in RE_EXCHANGE.finditer(text):
            t = m.group(1)
            if t.upper() in tickers and t.lower() not in STOPWORDS:
                cands.append({'span': (m.start(1), m.end(1)), 'text': t, 'label': 'TICKER'})

    for label, matcher in matchers.items():
        for _, start, end in matcher(doc):
            span = doc[start:end]
            txt = span.text
            txt_key = txt.lower()
            if len(txt.strip()) <= 1 or txt_key in STOPWORDS:
                continue
            if label == 'TICKER':
                if txt.upper() not in tickers:
                    continue
                if not any(fin_kw in text.lower() for fin_kw in FINANCIAL_KEYWORDS):
                    continue
                if span.root.pos_ not in {"NOUN", "PROPN"}:
                    continue
            elif label == 'COMPANY':
                if txt_key not in name_to_tickers:
                    continue
                if span.root.pos_ not in {"NOUN", "PROPN"}:
                    continue
                if not any(fin_kw in text.lower() for fin_kw in FINANCIAL_KEYWORDS):
                    continue
            cands.append({'span': (span.start_char, span.end_char), 'text': txt, 'label': label})

    seen, uniq = set(), []
    for c in cands:
        key = (c['span'], c['text'].lower(), c['label'])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq

# ─── MAIN EXTRACTOR CLASS ─────────────────────────────────────────────────────────
class FullExtractor:
    def __init__(self):
        self.t2n, self.n2t = load_instruments(STOCK_CSV, ETF_CSV)
        self.commodities = load_commodities(COMMODITY_CSV)
        self.industries = load_industries(INDUSTRY_CSV)
        self.derivatives = load_derivatives(DERIVATIVE_CSV)
        self.macros = load_macros(MACRO_CSV)

        self.nlp = spacy.load("en_core_web_trf")

        self.matchers = build_matchers(self.nlp, {
            'TICKER': {t: t for t in self.t2n.keys()},
            'COMPANY': self.n2t,
            'COMMODITY': self.commodities,
            'INDUSTRY': self.industries,
            'DERIVATIVE': self.derivatives,
            'MACRO': self.macros
        })

    def extract(self, text: str) -> List[Dict[str, Any]]:
        doc = self.nlp(text)
        return extract_candidates(text, doc, self.matchers, set(self.t2n.keys()), self.n2t)

    def label_to_ls_format(self, text: str, labels: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "data": {"text": text},
            "predictions": [
                {
                    "result": [
                        {
                            "from_name": "label",
                            "to_name": "text",
                            "type": "labels",
                            "value": {
                                "start": item["span"][0],
                                "end": item["span"][1],
                                "text": text[item["span"][0]:item["span"][1]],
                                "labels": [item["label"]]
                            }
                        }
                        for item in labels
                    ]
                }
            ]
        }

    def process_file(self, input_path: str, output_path: str):
        export_data = []
        with open(input_path, encoding='utf8') as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                labels = self.extract(text)
                print(f"\ntext:\n{text}\n")
                print("labels:")
                for l in labels:
                    print(f"{l['label']}: {l['text']} [{l['span'][0]}:{l['span'][1]}]")
                export_data.append(self.label_to_ls_format(text, labels))
        with open(output_path, 'w', encoding='utf8') as out:
            json.dump(export_data, out, indent=2)

# ─── EXECUTE ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    extractor = FullExtractor()
    extractor.process_file("./data/datasets/dataset40.txt", "./data/json/dataset40labels.json")

