#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FEATURES=["heart_rate","mbp","resp_rate","temperature","spo2","glucose","lab_50813",
"lab_50983","lab_50971","lab_50912","lab_51006","lab_50878","lab_50861","lab_50885",
"lab_50862","lab_51222","lab_51265","lab_51274","lab_51275","vasopressor_ativo"]
LOG={"glucose","lab_50813","lab_50912","lab_51006","lab_50878","lab_50861",
     "lab_50885","lab_51274","lab_51275"}
BINARY={"vasopressor_ativo"}

def run(cmd): print("\n$"," ".join(map(str,cmd)),flush=True); subprocess.run(cmd,check=True)

def interp_cont(t,v,x):
    ok=np.isfinite(t)&np.isfinite(v)
    if ok.sum()==0:return np.full(len(x),np.nan,np.float32)
    q=pd.DataFrame({"t":t[ok],"v":v[ok]}).groupby("t",as_index=False)["v"].mean()
    tt=q.t.to_numpy(float); vv=q.v.to_numpy(float)
    if len(tt)==1:return np.full(len(x),vv[0],np.float32)
    return np.interp(x,tt,vv).astype(np.float32)

def interp_binary(t,v,x):
    ok=np.isfinite(t)&np.isfinite(v)
    if ok.sum()==0:return np.full(len(x),np.nan,np.float32)
    order=np.argsort(t[ok]); tt=t[ok][order]; vv=v[ok][order]; out=np.empty(len(x),np.float32)
    for i,z in enumerate(x):
        j=np.searchsorted(tt,z,side="right")-1; out[i]=vv[0] if j<0 else vv[j]
    return out

def eta2(values,labels):
    v=np.asarray(values,float); l=np.asarray(labels); ok=np.isfinite(v); v=v[ok]; l=l[ok]
    gm=v.mean(); sst=((v-gm)**2).sum()
    if sst<=0:return 0.0
    return float(sum(len(v[l==c])*(v[l==c].mean()-gm)**2 for c in np.unique(l))/sst)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--feature-dir",required=True,type=Path)
    ap.add_argument("--labels-dir",required=True,type=Path)
    ap.add_argument("--positive-stays",required=True,type=Path)
    ap.add_argument("--out-dir",required=True,type=Path)
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    pos=pd.read_parquet(args.positive_stays)[["stay_id","first_positive_time"]].drop_duplicates("stay_id").copy()
    pos["stay_id"]=pos["stay_id"].astype("int64"); pos["first_positive_time"]=pd.to_numeric(pos["first_positive_time"],errors="coerce")
    pos=pos.sort_values("stay_id").reset_index(drop=True); wanted=set(pos.stay_id); idx={s:i for i,s in enumerate(pos.stay_id)}
    X=np.full((len(pos),96,len(FEATURES)),np.nan,np.float32); source=np.zeros(len(pos),np.int32); filled=np.zeros(len(pos),bool)
    fps=sorted(p for p in args.feature_dir.glob("batch_*.parquet") if not p.name.endswith("_labels.parquet"))
    schema=pq.read_schema(fps[0]).names; missing=[c for c in FEATURES if c not in schema]
    if missing: raise SystemExit(f"Features ausentes: {missing}")
    for fi,fp in enumerate(fps,1):
        lp=args.labels_dir/f"{fp.stem}_labels.parquet"
        f=pd.read_parquet(fp,columns=["stay_id",*FEATURES]); l=pd.read_parquet(lp,columns=["stay_id","rel_charttime"])
        if len(f)!=len(l) or not np.array_equal(f.stay_id.to_numpy(np.int64),l.stay_id.to_numpy(np.int64)):
            raise RuntimeError(f"Desalinhamento {fp.name}")
        d=f.copy(); d["rel_charttime"]=pd.to_numeric(l.rel_charttime,errors="coerce"); d=d[d.stay_id.isin(wanted)]
        for sid,g in d.groupby("stay_id",sort=False):
            i=idx[int(sid)]; onset=float(pos.loc[i,"first_positive_time"])
            g=g[np.isfinite(g.rel_charttime)&(g.rel_charttime>=0)&(g.rel_charttime<=onset)].sort_values("rel_charttime")
            if g.empty: continue
            source[i]=len(g); filled[i]=True; targets=np.zeros(96) if onset<=0 else np.linspace(0,onset,96)
            t=g.rel_charttime.to_numpy(float)
            for j,c in enumerate(FEATURES):
                v=pd.to_numeric(g[c],errors="coerce").to_numpy(float)
                X[i,:,j]=interp_binary(t,v,targets) if c in BINARY else interp_cont(t,v,targets)
        if fi%10==0 or fi==len(fps): print(f"[INFO] {fi}/{len(fps)} | stays={filled.sum():,}")
    if filled.sum()!=len(pos): raise RuntimeError(f"Faltaram {len(pos)-filled.sum()} stays")
    med=[]
    for j,c in enumerate(FEATURES):
        vals=X[:,:,j]; finite=vals[np.isfinite(vals)]
        if len(finite)==0: raise RuntimeError(f"{c} totalmente ausente")
        m=float(np.median(finite)); med.append(m); X[:,:,j]=np.where(np.isfinite(vals),vals,m)
    for j,c in enumerate(FEATURES):
        if c in LOG:
            if X[:,:,j].min()<0: raise RuntimeError(f"{c} negativo")
            X[:,:,j]=np.log1p(X[:,:,j])
    lows=[]; highs=[]; means=[]; stds=[]
    for j,c in enumerate(FEATURES):
        vals=X[:,:,j].astype(float)
        if c in BINARY: lo,hi=float(vals.min()),float(vals.max())
        else:
            lo,hi=np.quantile(vals,[.005,.995]); vals=np.clip(vals,lo,hi)
        mu=float(vals.mean()); sd=float(vals.std()) or 1.0
        X[:,:,j]=((vals-mu)/sd).astype(np.float32)
        lows.append(lo); highs.append(hi); means.append(mu); stds.append(sd)
    ds=args.out_dir/"normalized_preonset96_ts.npy"; meta=args.out_dir/"normalized_preonset96_meta.parquet"; np.save(ds,X)
    m=pos.copy(); m["onset_hours"]=m.first_positive_time/3600.; m["source_preonset_rows"]=source; m.to_parquet(meta,index=False)
    np.savez(args.out_dir/"normalized_preonset96_preprocessing.npz",features=np.array(FEATURES,dtype=object),
             raw_medians=np.array(med),log_features=np.array(sorted(LOG),dtype=object),
             winsor_lower=np.array(lows),winsor_upper=np.array(highs),mean=np.array(means),std=np.array(stds))
    b=Path(__file__).resolve().parents[1]/"core"; embed=b/"embed_minirocket.py"; cluster=b/"cluster_minirocket.py"
    emb=args.out_dir/"normalized_preonset96_minirocket.npy"; emeta=args.out_dir/"normalized_preonset96_minirocket_meta.parquet"; cdir=args.out_dir/"kmeans"
    run([sys.executable,str(embed),"--dataset",str(ds),"--meta_in",str(meta),"--emb_out",str(emb),"--meta_out",str(emeta),
         "--n_kernels","1024","--fit_stays","50000","--batch_size","128","--seed","43","--log_path",str(args.out_dir/"01_minirocket.log")])
    run([sys.executable,str(cluster),"--embeddings",str(emb),"--meta",str(emeta),"--output_dir",str(cdir),"--clusterer","kmeans",
         "--k_min","2","--k_max","6","--kmeans_epochs","10","--kmeans_batch","4096","--kmeans_n_init","5",
         "--pca_dim","50","--pca_sample","100000","--silhouette_sample","20000","--save_all_summaries",
         "--log_path",str(args.out_dir/"02_kmeans.log")])
    rows=[]
    for k in range(2,7):
        a=pd.read_parquet(cdir/f"cluster_embeddings_kmeans_k{k}.parquet")[["stay_id","cluster"]]; d=m.merge(a,on="stay_id",validate="one_to_one")
        e1=eta2(d.onset_hours,d.cluster); e2=eta2(d.source_preonset_rows,d.cluster)
        for c,g in d.groupby("cluster"):
            rows.append({"k":k,"cluster":int(c),"n":len(g),"median_onset_hours":float(g.onset_hours.median()),
                         "median_source_preonset_rows":float(g.source_preonset_rows.median()),
                         "eta2_onset_hours_global_for_k":e1,"eta2_source_rows_global_for_k":e2})
    audit=pd.DataFrame(rows); audit.to_csv(args.out_dir/"temporal_artifact_audit.csv",index=False)
    print("\n=== NORMALIZED PRE-ONSET AUDIT ==="); print(audit.to_string(index=False)); print("\n[OK]",args.out_dir)
if __name__=="__main__": main()
