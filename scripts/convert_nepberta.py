"""Convert the published NepBERTa TensorFlow checkpoint to PyTorch safetensors.

The NepBERTa Hub repository contains only tf_model.h5, which Transformers 5 cannot
load. Run this in a separate environment with transformers==4.44.2,
tensorflow==2.16.2, tf-keras==2.16.0 and torch:

    python scripts/convert_nepberta.py models_local/nepberta-pt

The script checks masked-LM logits for parity between the TF and PyTorch models
and records the result in <out>/conversion.json.
"""
import sys, numpy as np, json
from transformers import AutoTokenizer, TFAutoModelForMaskedLM, AutoModelForMaskedLM
REV = "1b8293fd5ed883081619e9895540c58eb740e7d0"
out = sys.argv[1]
tok = AutoTokenizer.from_pretrained("NepBERTa/NepBERTa", revision=REV)
tf_m = TFAutoModelForMaskedLM.from_pretrained("NepBERTa/NepBERTa", revision=REV)
pt_m = AutoModelForMaskedLM.from_pretrained("NepBERTa/NepBERTa", revision=REV, from_tf=True)
pt_m.eval()
sents = ["नेपालको राजधानी काठमाडौं हो", "कोभिड संक्रमितको संख्या बढ्दै गएको छ", "सरकारले नयाँ नीति ल्याएको छ"]
res = []
import torch
for s in sents:
    a = tf_m(**tok(s, return_tensors="tf")).logits.numpy()
    with torch.no_grad():
        b = pt_m(**tok(s, return_tensors="pt")).logits.numpy()
    res.append({"max_abs_diff": float(np.abs(a - b).max()), "argmax_agree": float((a.argmax(-1) == b.argmax(-1)).mean())})
pt_m.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
json.dump({"source": "NepBERTa/NepBERTa", "revision": REV, "converter": "transformers 4.44.2 from_tf=True",
           "parity_checks": res}, open(f"{out}/conversion.json", "w"), indent=2)
print(json.dumps(res, indent=2))
print([n for n, _ in pt_m.named_parameters()][:3])
