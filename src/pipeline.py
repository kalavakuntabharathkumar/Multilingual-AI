import argparse, hashlib
from pathlib import Path
import pandas as pd, yaml
from .metrics import weighted_kappa, bootstrap_mean_ci

def load_config(path):
    with open(path,encoding="utf-8") as f: return yaml.safe_load(f)

def validate_ratings(df):
    required={"prompt_id","rater_id","rating"}
    if not required.issubset(df.columns): raise ValueError(f"Ratings CSV must contain {sorted(required)}")
    df=df.copy(); df["rating"]=pd.to_numeric(df["rating"],errors="raise")
    if not df["rating"].between(1,5).all() or not (df["rating"]%1==0).all():
        raise ValueError("Ratings must be integer values from 1 to 5")
    return df

def paired_agreement(ratings):
    pivot=ratings.pivot_table(index="prompt_id",columns="rater_id",values="rating",aggfunc="first")
    paired=pivot.dropna(axis=0,how="any")
    if paired.shape[1]!=2: raise ValueError("Expected exactly two raters in paired rows")
    return weighted_kappa(paired.iloc[:,0].astype(int),paired.iloc[:,1].astype(int)),len(paired)

def run(config_path,mode="demo",provider=None):
    cfg=load_config(config_path)
    prompts=pd.read_csv(cfg["paths"]["prompts"])
    ratings=validate_ratings(pd.read_csv(cfg["paths"]["human_ratings"]))
    judges=pd.read_csv(cfg["paths"]["judge_scores"])
    out=Path(cfg["paths"]["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    kappa,n=paired_agreement(ratings)
    merged=ratings.merge(prompts[["prompt_id","language"]],on="prompt_id",how="left")
    rows=[]
    for language,g in merged.groupby("language"):
        mean,lo,hi=bootstrap_mean_ci(g.rating.values,cfg.get("bootstrap_iterations",2000),
            cfg.get("confidence_level",.95),cfg.get("seed",42))
        rows.append({"language":language,"mean_rating":mean,"ci_low":lo,"ci_high":hi,"n_ratings":len(g)})
    summary=pd.DataFrame(rows).sort_values("language")
    summary.to_csv(out/"language_summary.csv",index=False)
    judges.groupby("rubric_version").judge_rating.mean().to_csv(out/"judge_rubric_summary.csv")
    digest=hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    report=["# Multilingual LLM Evaluation Report","",f"- Mode: **{mode}**",
        f"- Prompt rows: {len(prompts)}",f"- Human rating rows: {len(ratings)}",
        f"- Paired prompts: {n}",f"- Quadratic weighted Cohen's kappa: **{kappa:.3f}**",
        f"- Config SHA-256: `{digest}`","","## Language summary","",summary.to_markdown(index=False),"",
        "## Interpretation","",
        "Percentile bootstrap intervals summarize rating-row sampling uncertainty; they do not remove sampling bias.",
        "Demo mode uses synthetic records. Do not present demo metrics as empirical findings."]
    (out/"evaluation_report.md").write_text("\\n".join(report),encoding="utf-8")
    if mode=="live":
        if provider=="openai":
            from .providers import generate_openai
            generated=generate_openai(prompts.to_dict("records"),cfg["models"]["openai"],
                cfg.get("generation",{}).get("max_tokens",256),cfg.get("generation",{}).get("temperature",0))
        elif provider=="transformers":
            from .providers import generate_transformers
            generated=generate_transformers(prompts.to_dict("records"),cfg["models"]["transformers"],
                cfg.get("generation",{}).get("max_tokens",256))
        else: raise ValueError("Live mode requires provider=openai or transformers")
        pd.DataFrame(generated).to_csv(out/f"{provider}_responses.csv",index=False)
    print(f"Report written to {out/'evaluation_report.md'}")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/demo.yaml")
    p.add_argument("--mode",choices=["demo","live"],default="demo")
    p.add_argument("--provider",choices=["openai","transformers"])
    a=p.parse_args(); run(a.config,a.mode,a.provider)
