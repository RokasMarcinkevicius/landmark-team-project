import os, re, time, random, argparse, csv
from urllib.parse import quote
from urllib.request import urlretrieve
import mwclient
from tqdm import tqdm

CATEGORIES = {
    "gediminas_tower": "Category:Gediminas Tower",
    "vilnius_cathedral": "Category:Vilnius Cathedral",
    "gate_of_dawn": "Category:Gate of Dawn",
    "st_anne": "Category:Church of St. Anne in Vilnius",
    "three_crosses": "Category:Three Crosses monument, Vilnius",
}

EXCLUDE_SUBSTR = ["historical images", "in art"]  # fotoğraf dışını azalt
VALID_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

def clean(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')

def iter_files_for_category(site, catname):
    cat = site.Categories[catname]
    for m in cat.members(namespace=6):  # File:
        yield m

def is_valid_title(title: str) -> bool:
    t = title.lower()
    return all(s not in t for s in EXCLUDE_SUBSTR)

def main(args):
    random.seed(42)
    site = mwclient.Site('commons.wikimedia.org')
    os.makedirs("data/raw", exist_ok=True)
    meta_rows = []
    for cls, cat in CATEGORIES.items():
        outdir = os.path.join("data/raw", cls)
        os.makedirs(outdir, exist_ok=True)
        files = []
        for fpage in iter_files_for_category(site, cat):
            if not is_valid_title(fpage.name):
                continue
            ii = fpage.imageinfo
            url = ii.get('url')
            mime = ii.get('mime', '')
            width = ii.get('width', 0)
            height = ii.get('height', 0)
            if not url:
                continue
            ext = os.path.splitext(url)[1]
            if ext not in VALID_EXT:
                continue
            if width < args.min_size or height < args.min_size:
                continue
            files.append((fpage.name, url, width, height, fpage.descriptionurl,
                          ii.get('commonsmetadata', {}).get('Artist', ''),
                          ii.get('commonmetadata', {}).get('LicenseShortName', '')))
        random.shuffle(files)
        picked = files[:args.max_per_class]
        for name, url, w, h, pageurl, artist, license_short in tqdm(picked, desc=f"Downloading {cls}"):
            fname = clean(os.path.basename(name)) + os.path.splitext(url)[1].lower()
            dest = os.path.join(outdir, fname)
            try:
                urlretrieve(url, dest)
                meta_rows.append([cls, fname, url, w, h, pageurl, artist, license_short])
                time.sleep(0.05)  # nazik olalım
            except Exception:
                continue
    # meta
    with open("data/SOURCES.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class","file","url","width","height","pageurl","artist","license"])
        w.writerows(meta_rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=300)
    ap.add_argument("--min-size", type=int, default=400)
    args = ap.parse_args()
    main(args)
