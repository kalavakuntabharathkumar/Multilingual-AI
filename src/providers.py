def generate_openai(prompts, model, max_tokens=256, temperature=0.0):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install requirements-openai.txt") from exc
    client=OpenAI(); rows=[]
    for item in prompts:
        r=client.chat.completions.create(model=model,temperature=temperature,max_tokens=max_tokens,
            messages=[{"role":"user","content":item["prompt"]}])
        rows.append({**item,"model":model,"response":r.choices[0].message.content})
    return rows

def generate_transformers(prompts, model_name, max_tokens=256):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements-transformers.txt") from exc
    tok=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto",device_map="auto")
    rows=[]
    for item in prompts:
        rendered=tok.apply_chat_template([{"role":"user","content":item["prompt"]}],
                                         tokenize=False,add_generation_prompt=True)
        inputs=tok(rendered,return_tensors="pt").to(model.device)
        out=model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False)
        answer=tok.decode(out[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True)
        rows.append({**item,"model":model_name,"response":answer})
    return rows
