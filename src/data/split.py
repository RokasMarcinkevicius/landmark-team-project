import shutil, random, os, glob
random.seed(42)
CLASSES = ["gediminas_tower","vilnius_cathedral","gate_of_dawn","st_anne","three_crosses"]
SRC = "data/raw"
for c in CLASSES:
    os.makedirs(f"data/train/{c}", exist_ok=True)
    os.makedirs(f"data/val/{c}", exist_ok=True)
    os.makedirs(f"data/test/{c}", exist_ok=True)
    imgs = glob.glob(os.path.join(SRC, c, "*"))
    imgs = [p for p in imgs if os.path.isfile(p)]
    random.shuffle(imgs)
    n = len(imgs); tr=int(0.7*n); va=int(0.15*n)
    for i,p in enumerate(imgs):
        dst = ("data/train/" if i<tr else "data/val/" if i<tr+va else "data/test/") + c
        shutil.copy2(p, dst)
