#!/usr/bin/env python3
"""
Financial NER Labeler (Fast Version)
=====================================
Uses word-based lookup for speed instead of regex per term.
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    matched_by: str = ""


class FinancialNERLabeler:
    """Fast financial entity labeler using word lookup."""
    
    AMBIGUOUS_WORDS = {
        'a', 'an', 'the', 'be', 'to', 'of', 'and', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with',
        'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if',
        'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just',
        'him', 'know', 'take', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see',
        'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back',
        'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want',
        'because', 'any', 'these', 'give', 'day', 'most', 'us', 'is', 'are', 'was', 'been', 'has', 'had',
        'may', 'very', 'much', 'too', 'own', 'same', 'tell', 'need', 'feel', 'high', 'low', 'old', 'big',
        'man', 'men', 'best', 'must', 'home', 'both', 'end', 'does', 'made', 'find', 'here', 'many', 'such',
        'way', 'each', 'next', 'got', 'few', 'real', 'those', 'ever', 'sure', 'why', 'post',
        'true', 'free', 'life', 'fund', 'half', 'hold', 'plan', 'risk', 'rate', 'run', 'tax',
        'pay', 'play', 'open', 'fast', 'gain', 'loss', 'term', 'cap', 'set', 'keep',
        'lot', 'hit', 'bit', 'beat', 'hope', 'idea', 'mind', 'hear', 'near', 'else', 'move', 'turn',
        'live', 'late', 'save', 'left', 'fall', 'kind', 'fact', 'part', 'case', 'week', 'head', 'hand',
        'line', 'side', 'help', 'name', 'land', 'food', 'care', 'house', 'word', 'game', 'class',
        'self', 'rule', 'power', 'order', 'group', 'human', 'area', 'money', 'point', 'world',
        'place', 'thing', 'state', 'never', 'still', 'today', 'read', 'last', 'city', 'start', 'ago',
        'buy', 'sell', 'cash', 'debt', 'loan', 'bond', 'bear', 'bull', 'peak', 'dip', 'rally', 'crash',
        'earn', 'cost', 'price', 'value', 'share', 'stock', 'trade', 'market', 'index', 'bank', 'return',
        'ai', 'am', 'by', 'go', 'if', 'is', 'it', 'me', 'my', 'no', 'of', 'on', 'or', 'so', 'to', 'up', 'we',
        'tv', 'uk', 'eu', 'dd', 'op', 'pm', 'hr', 'min', 'sec', 'usa', 'nyc',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
        'edit', 'update', 'tldr', 'imo', 'imho', 'fyi', 'btw', 'lol', 'omg', 'wtf', 'idk',
        'gap', 'net', 'sky', 'sun', 'key', 'yes', 'yet', 'bar', 'cup', 'top', 'tip', 'pop',
        'mom', 'dad', 'kid', 'pet', 'job', 'car', 'sea', 'pro', 'con', 'fan', 'app', 'web',
        'hot', 'cold', 'dry', 'wet', 'raw', 'fix', 'mix', 'win', 'lost', 'sit', 'fit',
    }
    
    POPULAR_TICKERS = {
        'aapl', 'msft', 'googl', 'goog', 'amzn', 'nvda', 'meta', 'tsla',
        'jpm', 'v', 'unh', 'ma', 'hd', 'pg', 'cvx', 'mrk', 'abbv', 'ko', 'pep', 'cost',
        'wmt', 'dis', 'csco', 'vz', 'adbe', 'crm', 'nflx', 'intc', 'amd', 'qcom', 'txn',
        'gme', 'amc', 'bb', 'pltr', 'sofi', 'nio', 'lcid', 'rivn', 'f', 'gm',
        'hood', 'coin', 'sq', 'pypl', 'shop', 'roku', 'snap', 'pins', 'uber',
        'abnb', 'docu', 'zm', 'pton', 'rblx', 'snow', 'crwd', 'ddog', 'mdb',
        'bac', 'wfc', 'c', 'gs', 'ms', 'xom', 'bp', 'cop', 'slb', 'oxy',
    }
    
    POPULAR_ETFS = {
        'spy', 'qqq', 'iwm', 'dia', 'voo', 'vti', 'vxus', 'bnd', 'agg',
        'gld', 'slv', 'uso', 'ung', 'arkk', 'arkw', 'arkg', 'arkf',
        'tqqq', 'sqqq', 'spxu', 'spxl', 'uvxy', 'svxy', 'vxx',
        'xlf', 'xlk', 'xle', 'xlv', 'xli', 'xlc', 'xly', 'xlp', 'xlu', 'xlb',
        'smh', 'soxx', 'xbi', 'ibb', 'kweb', 'fxi', 'eem', 'vwo', 'efa',
        'schd', 'dgro', 'vym', 'tlt', 'ief', 'shy', 'hyg', 'jnk',
        'vix', 'uvix', 'svix', 'jepi', 'jepq', 'splg', 'itot',
    }

    def __init__(self, data_dir: str = '/mnt/user-data/uploads'):
        self.data_dir = Path(data_dir)
        
        # Single word lookups
        self.tickers: Set[str] = set()
        self.etfs: Set[str] = set()
        self.commodities: Set[str] = set()
        self.derivatives: Set[str] = set()
        self.macro_terms: Set[str] = set()
        self.sectors: Set[str] = set()
        self.industries: Set[str] = set()
        
        # Multi-word phrase lookups (phrase -> label)
        self.phrases: Dict[str, str] = {}
        
        self.missing_suggestions: Dict[str, Set[str]] = defaultdict(set)
        
        self._load_reference_data()
    
    def _load_reference_data(self):
        print("Loading reference data...")
        self._load_stocks()
        self._load_etfs()
        self._load_commodities()
        self._load_derivatives()
        self._load_macro()
        self._load_sectors()
        
        print(f"  Tickers: {len(self.tickers)}")
        print(f"  ETFs: {len(self.etfs)}")
        print(f"  Commodities: {len(self.commodities)}")
        print(f"  Derivatives: {len(self.derivatives)}")
        print(f"  Macro terms: {len(self.macro_terms)}")
        print(f"  Sectors: {len(self.sectors)}")
        print(f"  Industries: {len(self.industries)}")
        print(f"  Multi-word phrases: {len(self.phrases)}")
    
    def _add_term(self, term: str, label: str):
        """Add a term to appropriate lookup."""
        term = term.lower().strip()
        if not term:
            return
        
        if ' ' in term:
            self.phrases[term] = label
        else:
            if label == 'TICKER':
                self.tickers.add(term)
            elif label == 'ETF':
                self.etfs.add(term)
            elif label == 'COMMODITY':
                self.commodities.add(term)
            elif label == 'DERIVATIVE':
                self.derivatives.add(term)
            elif label == 'MACRO':
                self.macro_terms.add(term)
            elif label == 'SECTOR':
                self.sectors.add(term)
            elif label == 'INDUSTRY':
                self.industries.add(term)
    
    def _load_stocks(self):
        filepath = self.data_dir / 'stocklist.csv'
        if not filepath.exists():
            filepath = self.data_dir / 'stockslist.csv'
        if not filepath.exists():
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = (row.get('Ticker') or '').strip().lower()
                if ticker and len(ticker) <= 5:
                    self.tickers.add(ticker)
        
        self.tickers.update(self.POPULAR_TICKERS)
    
    def _load_etfs(self):
        filepath = self.data_dir / 'etfs.csv'
        if not filepath.exists():
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = (row.get('Ticker') or '').strip().lower()
                if ticker:
                    self.etfs.add(ticker)
        
        self.etfs.update(self.POPULAR_ETFS)
    
    def _load_commodities(self):
        filepath = self.data_dir / 'commodities.csv'
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (row.get('Name') or '').strip().lower()
                    other = (row.get('OtherName') or '').strip().lower()
                    if name:
                        self._add_term(name, 'COMMODITY')
                    if other:
                        self._add_term(other, 'COMMODITY')
        
        for c in ['gold', 'silver', 'platinum', 'palladium', 'copper', 'aluminum', 'zinc', 'nickel',
                  'oil', 'crude', 'wti', 'brent', 'gasoline', 'corn', 'wheat', 'soybeans', 'soybean',
                  'coffee', 'sugar', 'cotton', 'cocoa', 'lumber', 'timber', 'uranium', 'lithium', 'cobalt']:
            self.commodities.add(c)
        
        for p in ['crude oil', 'natural gas', 'heating oil', 'rare earth']:
            self.phrases[p] = 'COMMODITY'
    
    def _load_derivatives(self):
        filepath = self.data_dir / 'derivatives.csv'
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (row.get('Derivative') or '').strip().lower()
                    others = (row.get('OtherNames') or '').strip()
                    if name:
                        self._add_term(name, 'DERIVATIVE')
                    for term in others.split('|'):
                        self._add_term(term, 'DERIVATIVE')
        
        for d in ['call', 'calls', 'put', 'puts', 'option', 'options', 'strike', 'expiry', 'expiration',
                  'itm', 'otm', 'atm', 'long', 'short', 'bullish', 'bearish', 'bull', 'bear',
                  'future', 'futures', 'forward', 'forwards', 'swap', 'swaps',
                  'spread', 'spreads', 'straddle', 'strangle', 'condor', 'butterfly',
                  'delta', 'gamma', 'theta', 'vega', 'rho', 'iv', 'premium', 'contract', 'contracts',
                  'exercise', 'assignment', 'margin', 'leap', 'leaps', 'weekly', 'weeklies',
                  'fd', 'fds', 'yolo', 'squeeze', 'hedge', 'hedging', 'hedged']:
            self.derivatives.add(d)
        
        for p in ['in the money', 'out of the money', 'at the money', 'iron condor',
                  'covered call', 'naked put', 'cash secured put', 'short squeeze', 
                  'gamma squeeze', 'gamma ramp', 'diamond hands', 'paper hands', 'bag holder',
                  'implied volatility', 'put to call ratio', 'put call ratio']:
            self.phrases[p] = 'DERIVATIVE'
    
    def _load_macro(self):
        filepath = self.data_dir / 'macro.csv'
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (row.get('Name') or '').strip().lower()
                    others = (row.get('OtherNames') or '').strip()
                    if name:
                        self._add_term(name, 'MACRO')
                    for term in others.split('|'):
                        self._add_term(term, 'MACRO')
        
        for m in ['inflation', 'deflation', 'stagflation', 'cpi', 'ppi', 'pce',
                  'gdp', 'gnp', 'recession', 'depression', 'recovery', 'expansion',
                  'unemployment', 'jobless', 'payroll', 'bps',
                  'fed', 'fomc', 'powell', 'treasury', 'yield', 'bond',
                  'qe', 'qt', 'tapering', 'stimulus', 'tariff', 'tariffs',
                  'deficit', 'surplus', 'ism', 'pmi', 'manufacturing', 'services',
                  'ecb', 'boe', 'boj', 'dollar', 'euro', 'yen', 'pound', 'yuan',
                  'currency', 'forex', 'fx', 'pivot', 'hawkish', 'dovish']:
            self.macro_terms.add(m)
        
        for p in ['interest rate', 'rate hike', 'rate cut', 'basis points',
                  'federal reserve', 'jerome powell', 'yield curve', 'inverted yield curve',
                  'quantitative easing', 'quantitative tightening', 'fiscal policy', 'monetary policy',
                  'trade deficit', 'trade surplus', 'debt ceiling', 'national debt',
                  'consumer confidence', 'consumer sentiment', 'retail sales',
                  'housing starts', 'building permits', 'existing home sales',
                  'nonfarm payroll', 'jobs report', 'central bank', 'central banks',
                  'soft landing', 'hard landing', 'inflation rate']:
            self.phrases[p] = 'MACRO'
    
    def _load_sectors(self):
        filepath = self.data_dir / 'sectorIndustries.csv'
        if not filepath.exists():
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sector = (row.get('Sector') or '').strip().lower()
                industry = (row.get('Industry') or '').strip().lower()
                sub = (row.get('Sub-Industry') or '').strip().lower()
                
                if sector:
                    self._add_term(sector, 'SECTOR')
                if industry:
                    self._add_term(industry, 'INDUSTRY')
                if sub:
                    self._add_term(sub, 'INDUSTRY')
    
    def _is_valid_ticker(self, word: str, has_dollar: bool) -> bool:
        if has_dollar:
            return True
        if word in self.POPULAR_TICKERS or word in self.POPULAR_ETFS:
            return True
        if word in self.AMBIGUOUS_WORDS:
            return False
        if len(word) <= 2:
            return False
        return True
    
    def find_entities(self, text: str) -> List[Entity]:
        entities = []
        used_positions = set()
        text_lower = text.lower()
        
        def add_entity(start: int, end: int, label: str, matched_text: str, matched_by: str):
            for pos in range(start, end):
                if pos in used_positions:
                    return False
            entities.append(Entity(text=matched_text, label=label, start=start, end=end, matched_by=matched_by))
            for pos in range(start, end):
                used_positions.add(pos)
            return True
        
        # 1. Find $TICKER patterns first
        for match in re.finditer(r'\$([a-zA-Z]{1,5})\b', text):
            ticker = match.group(1).lower()
            start, end = match.start(), match.end()
            
            if ticker in self.etfs:
                add_entity(start, end, 'ETF', match.group(0), f'${ticker}')
            elif ticker in self.tickers or ticker not in self.AMBIGUOUS_WORDS:
                add_entity(start, end, 'TICKER', match.group(0), f'${ticker}')
                if ticker not in self.tickers:
                    self.missing_suggestions['stocklist.csv'].add(ticker.upper())
        
        # 2. Find multi-word phrases
        for phrase, label in sorted(self.phrases.items(), key=lambda x: -len(x[0])):
            pattern = r'\b' + re.escape(phrase) + r'\b'
            for match in re.finditer(pattern, text_lower):
                add_entity(match.start(), match.end(), label, text[match.start():match.end()], phrase)
        
        # 3. Find single-word entities
        for match in re.finditer(r'\b([a-zA-Z][a-zA-Z0-9\.]*)\b', text):
            word = match.group(1).lower()
            start, end = match.start(), match.end()
            
            if start in used_positions:
                continue
            
            if word in self.etfs and self._is_valid_ticker(word, False):
                add_entity(start, end, 'ETF', text[start:end], word)
            elif word in self.tickers and self._is_valid_ticker(word, False):
                add_entity(start, end, 'TICKER', text[start:end], word)
            elif word in self.commodities:
                add_entity(start, end, 'COMMODITY', text[start:end], word)
            elif word in self.derivatives:
                add_entity(start, end, 'DERIVATIVE', text[start:end], word)
            elif word in self.macro_terms:
                add_entity(start, end, 'MACRO', text[start:end], word)
            elif word in self.sectors:
                add_entity(start, end, 'SECTOR', text[start:end], word)
            elif word in self.industries:
                add_entity(start, end, 'INDUSTRY', text[start:end], word)
        
        entities.sort(key=lambda e: e.start)
        return entities
    
    def tokenize_and_tag(self, text: str, entities: List[Entity]) -> List[Tuple[str, str]]:
        tokens = []
        for match in re.finditer(r'\S+', text):
            token = match.group()
            start, end = match.start(), match.end()
            
            tag = 'O'
            for entity in entities:
                if start >= entity.start and end <= entity.end:
                    tag = f'B-{entity.label}' if start == entity.start else f'I-{entity.label}'
                    break
            tokens.append((token, tag))
        return tokens
    
    def label_file(self, input_path: str, output_path: str) -> List[dict]:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"\nLabeling {len(lines)} samples...")
        samples = []
        
        for i, line in enumerate(lines):
            entities = self.find_entities(line)
            tokens = self.tokenize_and_tag(line, entities)
            
            samples.append({
                'text': line,
                'entities': [{'text': e.text, 'label': e.label, 'start': e.start, 'end': e.end, 'matched_by': e.matched_by} for e in entities],
                'tokens': tokens
            })
            
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(lines)}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"Saved to {output_path}")
        
        conll_path = output_path.replace('.jsonl', '.conll')
        with open(conll_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                for token, tag in sample['tokens']:
                    f.write(f"{token}\t{tag}\n")
                f.write('\n')
        print(f"Saved to {conll_path}")
        
        return samples
    
    def print_stats(self, samples: List[dict]):
        entity_counts = defaultdict(int)
        samples_with_entities = 0
        
        for sample in samples:
            if sample['entities']:
                samples_with_entities += 1
            for e in sample['entities']:
                entity_counts[e['label']] += 1
        
        total = sum(entity_counts.values())
        
        print("\n" + "=" * 60)
        print("LABELING STATISTICS")
        print("=" * 60)
        print(f"Total samples: {len(samples)}")
        print(f"Samples with entities: {samples_with_entities} ({100*samples_with_entities/len(samples):.1f}%)")
        print(f"Total entities: {total}")
        print(f"\nEntity breakdown:")
        for label, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
            print(f"  {label}: {count}")
        
        if self.missing_suggestions.get('stocklist.csv'):
            items = sorted(self.missing_suggestions['stocklist.csv'])[:20]
            print(f"\nSuggested additions to stocklist.csv:")
            for item in items:
                print(f"  - {item}")
        
        print("=" * 60)


def main():
    labeler = FinancialNERLabeler(data_dir='/mnt/user-data/uploads')
    
    input_file = '/mnt/user-data/uploads/allRedditPosts.txt'
    output_file = '/mnt/user-data/outputs/labeled_reddit_posts.jsonl'
    
    samples = labeler.label_file(input_file, output_file)
    labeler.print_stats(samples)
    
    print(f"\n✓ Ready for review!")


if __name__ == '__main__':
    main()
