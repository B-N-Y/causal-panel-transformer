"""Prepare the light-tunnel data of Section 8.7 from the public benchmark of Gamella, Bing and Runge (2025), built on the Causal
Chambers (Gamella, Peters and Buehlmann, 2025; CC BY 4.0).

Input: lt_crl_benchmark_v1.zip of the benchmark (2,763,884,276 bytes, MD5 8ec492afffd0142abd70b1fa7082b104), downloaded by the
user from the Causal Chambers repository. Output, in the order of citris_1.csv:
  <out_dir>/citris_1_x64.npy   the 100,000 images of citris_1 (uint8, 100000 x 64 x 64 x 3)
  <out_dir>/citris_1_truth.csv the columns flag (observed regime), red, green, blue, pol_1, pol_2 (recorded factors)
Usage: python scripts/prepare_light_tunnel.py data/lt_crl_benchmark_v1.zip data/light_tunnel [--skip_md5]"""
import hashlib, io, os, sys, zipfile
import numpy as np, pandas as pd
from PIL import Image

MD5 = "8ec492afffd0142abd70b1fa7082b104"
ZIP, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
if "--skip_md5" not in sys.argv:
    h = hashlib.md5()
    with open(ZIP, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    if h.hexdigest() != MD5:
        sys.exit(f"MD5 mismatch: {h.hexdigest()} (expected {MD5})")
K = "lt_crl_benchmark_v1/citris_1/"
zf = zipfile.ZipFile(ZIP)
d = pd.read_csv(zf.open(K + "citris_1.csv"))
X = np.stack([np.asarray(Image.open(io.BytesIO(zf.read(K + "images_64/" + f))).convert("RGB")) for f in d.image_file])
np.save(os.path.join(OUT, "citris_1_x64.npy"), X)
d[["flag", "red", "green", "blue", "pol_1", "pol_2"]].to_csv(os.path.join(OUT, "citris_1_truth.csv"), index=False)
print("images", X.shape, "->", OUT)
