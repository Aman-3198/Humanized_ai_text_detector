"""Detector v5 - 3 classes: plain_ai / humanized_ai / human.
Text is cleaned (detokenised: no 'word , word' spacing shortcut), cut into ~300-char windows, each window scored by
word TF-IDF LR + char TF-IDF LR + grammar/word-rarity LR; log-probs averaged over models and windows.
Needs: scikit-learn, spacy + en_core_web_sm, wordfreq."""
import re, unicodedata, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from ling import ling_feats
CLASSES=['plain_ai','humanized_ai','human']
ZW=re.compile('[\u200b\u200c\u200d\u2060\ufeff\u00ad]'); SENT=re.compile(r'(?<=[.!?])\s+')
def clean(t):
    t=str(t).replace('\r',' '); t=ZW.sub('',t); t=unicodedata.normalize('NFKC',t)
    t=t.translate({0x2018:"'",0x2019:"'",0x201c:'"',0x201d:'"',0x2013:'-',0x2014:' - ',0x2026:'...'})
    t=re.sub(r'\s+',' ',t).strip()
    t=re.sub(r"\s+([,.;:!?%)\]}])",r"\1",t); t=re.sub(r"([(\[{$])\s+",r"\1",t)
    t=re.sub(r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b",r"\1",t); t=re.sub(r'"\s*([^"]*?)\s*"',r'"\1"',t)
    return re.sub(r"\s+-\s+"," - ",t)
def chunks(text,target=300,minlen=120):
    if len(text)<=target*1.6: return [text]
    out=[];cur=''
    for s in SENT.split(text):
        cur=(cur+' '+s).strip()
        if len(cur)>=target: out.append(cur); cur=''
    if len(cur)>=minlen or not out: out.append(cur)
    elif cur: out[-1]=out[-1]+' '+cur
    return [c for c in out if c]
class DetectorV5:
    band=0.55   # top-class probability below this -> 'uncertain'
    def fit(self,windows,F,y,C=3,Cl=0.3):
        self.vw=TfidfVectorizer(ngram_range=(1,2),min_df=3,sublinear_tf=True,max_features=200000)
        self.vc=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=5,sublinear_tf=True,lowercase=False,max_features=200000)
        self.lw=LogisticRegression(C=C,class_weight='balanced',max_iter=1500).fit(self.vw.fit_transform(windows),y)
        self.lc=LogisticRegression(C=C,class_weight='balanced',max_iter=1500).fit(self.vc.fit_transform(windows),y)
        self.sc=StandardScaler().fit(F); self.ll=LogisticRegression(C=Cl,class_weight='balanced',max_iter=3000).fit(np.clip(self.sc.transform(F),-5,5),y); return self
    def _logp(self,W,F):
        L=[np.log(self.lw.predict_proba(self.vw.transform(W))+1e-9),np.log(self.lc.predict_proba(self.vc.transform(W))+1e-9),
           np.log(self.ll.predict_proba(np.clip(self.sc.transform(F),-5,5))+1e-9)]
        return np.mean(L,0)
    def predict_proba(self,texts):
        T=[clean(t) for t in texts]; cs=[chunks(t) for t in T]; flat=[c for cc in cs for c in cc]
        L=self._logp(flat,np.nan_to_num(ling_feats(flat))); out=[];i=0
        for cc in cs:
            z=L[i:i+len(cc)].mean(0); p=np.exp(z-z.max()); out.append(p/p.sum()); i+=len(cc)
        return np.array(out)
    def predict(self,texts):
        P=self.predict_proba(texts); lab=np.array(CLASSES)[P.argmax(1)]; return np.where(P.max(1)<self.band,'uncertain',lab)
if __name__=='__main__':
    import sys, joblib
    m=joblib.load('detector_v5.joblib'); txt=sys.argv[1:] or [sys.stdin.read()]
    for t,p,l in zip(txt,m.predict_proba(txt),m.predict(txt)): print('  '.join(f'{c}={v:.2f}' for c,v in zip(CLASSES,p)),'->',l,'|',t[:60].replace('\n',' '))
