"""Q7 (bonus): UCI MAGIC Gamma Telescope logistic regression, standard vs ZSGD.

No fabricated benchmark: results are computed only after downloading the UCI data.
The held-out test set is never used for optimization or model selection.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

UCI_URL = "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip"
FEATURES = ("fLength","fWidth","fSize","fConc","fConc1","fAsym",
            "fM3Long","fM3Trans","fAlpha","fDist")


def load_magic(path=None, download=False):
    """Read official magic04.data, or download its official UCI ZIP on request."""
    if path is None:
        if not download:
            raise ValueError("Supply --data PATH or explicitly enable --download")
        request = urllib.request.Request(UCI_URL, headers={"User-Agent":"ENSAI-Q7-research/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            matches = [n for n in archive.namelist() if Path(n).name=="magic04.data"]
            if len(matches)!=1:
                raise ValueError("Official archive must contain one magic04.data")
            content=archive.read(matches[0])
    else:
        content=Path(path).read_bytes()
    digest=hashlib.sha256(content).hexdigest()
    lines=content.decode("utf-8").splitlines()
    if not lines:
        raise ValueError("Empty dataset")
    data=np.empty((len(lines),10),dtype=np.float64)
    labels=np.empty(len(lines),dtype=np.int8)
    for i,line in enumerate(lines):
        parts=line.strip().split(",")
        if len(parts)!=11 or parts[-1] not in ("g","h"):
            raise ValueError(f"Invalid MAGIC record on line {i+1}")
        data[i]=[float(v) for v in parts[:10]]
        labels[i]=int(parts[-1]=="g")  # gamma is the positive signal class
    if not np.isfinite(data).all() or len(np.unique(labels))!=2:
        raise ValueError("Non-finite features or missing class")
    return data,labels,digest


def split_and_scale(x,y,seed=42):
    if x.ndim!=2 or x.shape[1]!=10 or y.shape!=(len(x),):
        raise ValueError("Expected 10 features and matching labels")
    # 60% train, 20% validation, 20% test. Stratified and reproducible.
    xa,xt,ya,yt=train_test_split(x,y,test_size=.2,random_state=seed,stratify=y)
    xr,xv,yr,yv=train_test_split(xa,ya,test_size=.25,random_state=seed,stratify=ya)
    scaler=StandardScaler().fit(xr)
    return ((scaler.transform(xr),yr),(scaler.transform(xv),yv),
            (scaler.transform(xt),yt))


def sigmoid(logits):
    logits=np.asarray(logits)
    return np.exp(-np.logaddexp(0.,-logits))


def component_loss(theta,xi,yi,penalty=0.):
    """Logistic cross-entropy; theta[0] is intercept, no intercept penalty."""
    score=theta[0]+xi@theta[1:]
    return float(np.logaddexp(0.,score)-yi*score
                 +.5*penalty*np.dot(theta[1:],theta[1:]))


def two_point(theta,xi,yi,u,h=1e-2,penalty=0.):
    if not np.isfinite(h) or h<=0:
        raise ValueError("h must be finite and positive")
    return ((component_loss(theta+h*u,xi,yi,penalty)
            -component_loss(theta-h*u,xi,yi,penalty))/(2*h))*u


def zsgd(x,y,*,budget=200000,seed=42,h=.01,learning_rate=.05,
         penalty=1e-3,checkpoints=40):
    """ZSGD with Rademacher U, E[UU^T]=I and eta_k=lr/sqrt(k).

    One iteration costs exactly TWO component loss evaluations. Full-dataset
    diagnostic loss is excluded from the optimization-oracle budget.
    """
    if not isinstance(budget,(int,np.integer)) or budget<0 or checkpoints<1:
        raise ValueError("Invalid budget/checkpoints")
    if not np.isfinite(learning_rate) or learning_rate<=0 or penalty<0:
        raise ValueError("Invalid step or penalty")
    if x.ndim!=2 or len(x)!=len(y) or len(x)==0:
        raise ValueError("Invalid data")
    rng=np.random.default_rng(seed)
    theta=np.zeros(x.shape[1]+1)
    history=[]
    steps=budget//2
    for k in range(1,steps+1):
        i=int(rng.integers(len(y)))
        u=rng.choice(np.array([-1.,1.]),size=len(theta))
        gradient=two_point(theta,x[i],y[i],u,h,penalty)
        theta-=learning_rate/np.sqrt(k)*gradient
        if k==steps or k%max(1,steps//checkpoints)==0:
            history.append({"component_loss_calls":2*k,
                            "train_logloss":float(np.mean(np.logaddexp(0.,theta[0]+x@theta[1:])
                                             -y*(theta[0]+x@theta[1:])) )})
    return theta,history,{"component_loss_calls":2*steps,
                          "unused_budget":budget-2*steps,"updates":steps}


def scores(theta,x):
    return sigmoid(theta[0]+x@theta[1:])


def metrics(y,p,threshold=.5):
    pred=(p>=threshold).astype(int)
    fpr,tpr,_=roc_curve(y,p)
    # Gamma positive: false-positive rate = hadrons mislabeled as gamma.
    out={"roc_auc":float(roc_auc_score(y,p)),
         "log_loss":float(log_loss(y,p,labels=[0,1])),
         "accuracy":float(accuracy_score(y,pred)),
         "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
         "precision_gamma":float(precision_score(y,pred,zero_division=0)),
         "recall_gamma":float(recall_score(y,pred,zero_division=0)),
         "f1_gamma":float(f1_score(y,pred,zero_division=0)),
         "fpr_hadron_to_gamma":float(np.mean(pred[np.asarray(y)==0])),
         "confusion_matrix_true_rows_hadron_gamma":confusion_matrix(y,pred,labels=[0,1]).tolist()}
    for cap in (.01,.02,.05,.10,.20):
        out[f"max_tpr_at_fpr_le_{cap:.2f}"]=float(np.max(tpr[fpr<=cap]))
    return out


def experiment(path=None,download=False,seed=42,budget=200000,repeats=3,
               learning_rate=.05,penalty=1e-3):
    if repeats<1:
        raise ValueError("repeats must be positive")
    x,y,sha=load_magic(path,download)
    train,val,test=split_and_scale(x,y,seed)
    xr,yr=train; xv,yv=val; xt,yt=test
    # Match regularization convention: sklearn C=1/(n*penalty)
    # sklearn objective = mean(logloss) + ||w||²/(2*n*C).
    baseline=LogisticRegression(C=1/(len(yr)*penalty) if penalty>0 else 1e12,
                                solver="lbfgs",max_iter=3000,random_state=seed)
    baseline.fit(xr,yr)
    base_val=baseline.predict_proba(xv)[:,1]
    base_test=baseline.predict_proba(xt)[:,1]
    result={"dataset":"UCI MAGIC Gamma Telescope (id=159)",
            "source_url":UCI_URL,"source_sha256":sha,"rows":len(y),
            "features":list(FEATURES),"class_mapping":{"h":0,"g":1},
            "class_counts":{"hadron":int(np.sum(y==0)),"gamma":int(np.sum(y==1))},
            "split":{"train":len(yr),"validation":len(yv),"test":len(yt),
                     "seed":seed,"stratified":True,"scaler_fit":"train_only"},
            "training":{"zsgd_budget_component_loss_calls":budget,
                        "repeats":repeats,"learning_rate":learning_rate,
                        "eta":"learning_rate/sqrt(k)","h":.01,
                        "penalty":penalty,"direction":"Rademacher, E[UU^T]=I",
                        "baseline":"sklearn LogisticRegression(lbfgs), matched L2 penalty",
                        "selection":"no test-based tuning; parameters fixed in advance"},
            "baseline":{"validation":metrics(yv,base_val),
                        "test":metrics(yt,base_test)},
            "zsgd_runs":[]}
    curves=[]
    for run in range(repeats):
        theta,hist,cost=zsgd(xr,yr,budget=budget,seed=seed+100+run,
                             learning_rate=learning_rate,penalty=penalty)
        val_metrics=metrics(yv,scores(theta,xv))
        test_metrics=metrics(yt,scores(theta,xt))
        result["zsgd_runs"].append({"seed":seed+100+run,"oracle":cost,
                                    "validation":val_metrics,"test":test_metrics})
        curves.extend([{"run":run,**r} for r in hist])
    return result,curves


def save(result,curves,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/"q7.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    with (out/"q7_optimization.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=["run","component_loss_calls","train_logloss"])
        w.writeheader();w.writerows(curves)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for run in range(len(result["zsgd_runs"])):
        r=[row for row in curves if row["run"]==run]
        axes[0].plot([v["component_loss_calls"] for v in r],
                     [v["train_logloss"] for v in r],label=f"ZSGD seed {run+1}")
    axes[0].set(xlabel="Component-loss evaluations",ylabel="Training log-loss",
                title="ZSGD optimization")
    axes[0].legend();axes[0].grid(alpha=.25)
    names=["Standard LR"]+[f"ZSGD {i+1}" for i in range(len(result["zsgd_runs"]))]
    aucs=[result["baseline"]["test"]["roc_auc"]]+[r["test"]["roc_auc"] for r in result["zsgd_runs"]]
    axes[1].bar(names,aucs)
    axes[1].set(ylabel="Held-out ROC-AUC",title="Test discrimination",ylim=(0,1))
    axes[1].tick_params(axis="x",labelrotation=30)
    fig.tight_layout();fig.savefig(out/"q7_comparison.png",dpi=180);plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data",type=Path,help="Path to official magic04.data")
    source.add_argument("--download",action="store_true",help="Download official UCI dataset")
    parser.add_argument("--budget",type=int,default=200000)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--output",type=Path,default=Path("results"))
    args=parser.parse_args()
    result,curves=experiment(path=args.data,download=args.download,
                             budget=args.budget,repeats=args.repeats,seed=args.seed)
    save(result,curves,args.output)
    print(json.dumps({"baseline_test":result["baseline"]["test"],
                      "zsgd_test":[r["test"] for r in result["zsgd_runs"]]},indent=2))
