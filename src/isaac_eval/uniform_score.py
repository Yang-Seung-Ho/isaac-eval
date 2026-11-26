"""Uniform final-answer scorer used for all results in Section VII (common scoring rule, Section V-D).

Usage (from the repository root):
    ./.venv/bin/python3 -m isaac_eval.uniform_score


Every system is scored from its FINAL ANSWER TEXT with one rule:
  (a) value check   : every numeric value in the gold answer appears in the response
                      (|r-g| <= 0.5 for |g| >= 10, else <= half a unit of the gold's last decimal);
  (b) referent check: every gold string that is part of the answer (i.e. not already given
                      in the query) appears in the response (station, month, taxon, ...).
Surplus numbers are allowed (containment rule), identically for all systems.
"""
import sys, re, json, glob, collections
from pathlib import Path
RUNS = Path("results/runs")  # run from the repository root
from decimal import Decimal
from isaac_eval.numeric import extract_numbers, flatten_facts
import isaac_eval.numeric as _num
# Fix: a response that STARTS with a decimal number ("74914.17입니다") was read as a list marker "74914."
_num._LIST_MARKER_RE = re.compile(r"^[ \t]*(?:[*\-+][ \t]*)?(?:\*\*)?(\d+)[.)](?=\s|\*|$)", re.MULTILINE)
from isaac_eval.metrics import answer_accuracy, answer_accuracy_from_text, wilson_interval

MONTHS = {'Jan.':(1,'January'),'Feb.':(2,'February'),'Mar.':(3,'March'),'Apr.':(4,'April'),'May':(5,'May'),
          'Jun.':(6,'June'),'Jul.':(7,'July'),'Aug.':(8,'August'),'Sep.':(9,'September'),'Oct.':(10,'October'),
          'Nov.':(11,'November'),'Dec.':(12,'December')}
META_KEYS = {'field','rank','mode'}

def gold_numbers(g):
    out=[]
    def walk(v):
        if isinstance(v,dict): [walk(x) for x in v.values()]
        elif isinstance(v,(list,tuple)): [walk(x) for x in v]
        elif isinstance(v,bool): pass
        elif isinstance(v,(int,float)): out.append(Decimal(str(v)))
    walk(g); return out

def gold_strings(g):
    if not isinstance(g,dict): return []
    return [v for k,v in g.items() if isinstance(v,str) and k not in META_KEYS]

def tol(g):
    if abs(g) >= 10: return Decimal('0.5')
    exp = -g.as_tuple().exponent if g.as_tuple().exponent < 0 else 0
    return Decimal('0.5') * (Decimal(10) ** -exp) if exp else Decimal('0.5')

def referent_present(s, text, query):
    if s in MONTHS:
        n, full = MONTHS[s]
        pats = [re.escape(s.rstrip('.')), full, f'{n}월']
        return any(re.search(p + (r'(?![0-9])' if p.endswith('월') else ''), text, re.I) for p in pats)
    if re.fullmatch(r'J\d+', s):
        return re.search(re.escape(s) + r'(?!\d)', text) is not None
    return s.lower() in text.lower()


KO_ORD = {'첫':1,'한':1,'두':2,'둘':2,'세':3,'셋':3,'네':4,'넷':4,'다섯':5,'여섯':6,'일곱':7,'여덟':8,'아홉':9,'열한':11,'열두':12,'열':10}
EN_ORD = {'first':1,'second':2,'third':3,'fourth':4,'fifth':5,'sixth':6,'seventh':7,'eighth':8,'ninth':9,'tenth':10,'eleventh':11,'twelfth':12}

SINO = {'일':1,'이':2,'삼':3,'사':4,'오':5,'육':6,'유':6,'칠':7,'팔':8,'구':9,'시':10}
def sino_tokens(q):
    out=set()
    for m in re.finditer(r'(십?[일이삼사오육칠팔구]?|[일이삼사오육유칠팔구시])\s*(월|번|째)', q):
        t=m.group(1)
        if not t: continue
        if t.startswith('십'): v = 10 + (SINO.get(t[1:],0) if len(t)>1 else 0)
        else: v = SINO.get(t)
        if v: out.add((v, m.group(2)))
    return out

def given_in_query(s, q):
    if referent_present(s, q, ''): return True
    if s in MONTHS and (MONTHS[s][0],'월') in sino_tokens(q): return True
    m = re.fullmatch(r'J(\d+)', s)
    if m:
        n = int(m.group(1))
        nums = {int(x) for x in re.findall(r'\d+', q)}
        if n in nums: return True
        for w,v in sorted(KO_ORD.items(), key=lambda x:-len(x[0])):
            if v==n and re.search(w + r'\s*(번째|번|째)', q): return True
        if any(v==n and u in ('번','째') for v,u in sino_tokens(q)): return True
        for w,v in EN_ORD.items():
            if v==n and re.search(w, q, re.I): return True
    return False

def score(rec):
    text = rec.get('final_response') or ''
    q = rec.get('query','')
    g = rec['gold_value']
    nums = extract_numbers(text)
    val_ok = all(any(abs(r-x) <= tol(x) for r in nums) for x in gold_numbers(g))
    refs = [s for s in gold_strings(g) if not given_in_query(s, q)]
    ref_ok = all(referent_present(s, text, q) for s in refs)
    return val_ok and ref_ok, val_ok, ref_ok

def load(prefix):
    fs = sorted(f for f in glob.glob(str(RUNS / f'{prefix}_*.json'))
                if re.fullmatch(re.escape(prefix)+r'_[A-Z]+\d+\.json', f.split('/')[-1]))
    return [json.load(open(f)) for f in fs]


GROUPS = ["proposed", "standard_rag", "self_rag", "crag", "pot_pal_jeju",
          "proposed_asv_bg11", "proposed_asv_nies", "proposed_asv_ohm",
          "pot_pal_asv_bg11", "pot_pal_asv_nies", "pot_pal_asv_ohm",
          "standard_rag_asv_bg11", "standard_rag_asv_nies", "standard_rag_asv_ohm",
          "pot_pal_adv40_v2", "proposed_unmasked_adv40_v2", "proposed_adv40_v2", "hybrid_e2e"]

if __name__ == "__main__":
    for g in GROUPS:
        R = load(g)
        if not R:
            continue
        k = sum(score(r)[0] for r in R)
        print(f"{g:32s} {k:3d}/{len(R):3d} = {k/len(R):.3f}  Wilson95 {wilson_interval(k, len(R))}")
