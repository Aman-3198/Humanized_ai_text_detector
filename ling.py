import numpy as np, re
from collections import Counter
import spacy, wordfreq
NLP=spacy.load('en_core_web_sm',disable=['ner','lemmatizer'])
POS=['ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','PUNCT','SCONJ','VERB','X']
W=re.compile(r"[A-Za-z]+")
def depth(t):
    d=0
    while t.head is not t and d<30: t=t.head; d+=1
    return d
def ling_feats(texts):
    out=[]
    for doc in NLP.pipe(texts,batch_size=64):
        toks=[t for t in doc if not t.is_space]; n=max(len(toks),1)
        pc=Counter(t.pos_ for t in toks); f=[pc.get(p,0)/n for p in POS]
        bg=Counter((a.pos_,b.pos_) for a,b in zip(toks,toks[1:])); nb=max(sum(bg.values()),1)
        for pr in [('DET','NOUN'),('ADJ','NOUN'),('NOUN','ADP'),('ADP','DET'),('AUX','VERB'),('PRON','VERB'),('ADV','VERB'),('NOUN','NOUN'),('PUNCT','PRON'),('CCONJ','PRON'),('NOUN','PUNCT'),('VERB','ADP')]: f.append(bg.get(pr,0)/nb)
        sents=list(doc.sents); sl=[len([t for t in s if not t.is_punct]) for s in sents] or [0]
        dp=[depth(t) for t in toks] or [0]; dc=Counter(t.dep_ for t in toks)
        f+=[np.mean(dp),np.max(dp),dc.get('nsubjpass',0)/n,dc.get('auxpass',0)/n,dc.get('relcl',0)/n,dc.get('advcl',0)/n,dc.get('conj',0)/n,dc.get('appos',0)/n,dc.get('prep',0)/n,dc.get('neg',0)/n,dc.get('agent',0)/n,dc.get('mark',0)/n,
            np.mean([max(depth(t) for t in s) for s in sents]) if sents else 0, sum(1 for s in sents if s.root.pos_ not in('VERB','AUX'))/max(len(sents),1)]
        ws=[w.lower() for w in W.findall(doc.text)] or ['x']
        z=np.array([wordfreq.zipf_frequency(w,'en') for w in ws])
        f+=[z.mean(),z.std(),np.percentile(z,10),np.percentile(z,25),np.median(z),(z<3).mean(),(z<4).mean(),(z<5).mean(),(z>=6).mean(),(z==0).mean()]
        # sentence-to-sentence variation of mean rarity
        out.append(f)
    return np.nan_to_num(np.array(out,float))
