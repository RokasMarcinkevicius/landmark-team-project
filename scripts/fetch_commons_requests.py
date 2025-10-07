import os, re, time, random, argparse, csv, requests, sys, pathlib
from tqdm import tqdm
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

CATEGORIES = {
    "gediminas_tower": "Category:Gediminas Tower",
    "vilnius_cathedral": "Category:Vilnius Cathedral",
    "gate_of_dawn": "Category:Gate of Dawn",
    "st_anne": "Category:Church of St. Anne in Vilnius",
    "three_crosses": "Category:Three Crosses monument, Vilnius",
}
EXCLUDE_SUBSTR = ["historical images", "in art"]
VALID_EXT = {".jpg",".jpeg",".png",".JPG",".JPEG",".PNG"}
API = "https://commons.wikimedia.org/w/api.php"
UA = "landmark-id/0.1 (academic; Vilnius MVP; contact: arda.student)"
LOG_FILE = "logs/last_response.html"

def clean(s): return re.sub(r"[^a-z0-9]+","_",s.lower()).strip("_")

def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    retry = Retry(
        total=5, connect=5, read=5, backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s

def mw_get_json(session, params):
    # MediaWiki bazen HTML dönebilir; kontrol edip debug’a yaz.
    params = dict(params)
    params.setdefault("format", "json")
    params.setdefault("origin", "*")  # CORS safe, sorun çıkarmaz
    r = session.get(API, params=params, timeout=30)
    try:
        j = r.json()
        return j
    except Exception:
        pathlib.Path("logs").mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "wb") as f:
            f.write(r.content)
        raise RuntimeError(f"Non-JSON response ({r.status_code}). Saved to {LOG_FILE}")

def list_category_files(session, cat):
    params = {
        "action":"query","list":"categorymembers","cmtitle":cat,
        "cmtype":"file","cmlimit":"500"
    }
    files = []
    while True:
        j = mw_get_json(session, params)
        files.extend(j.get("query",{}).get("categorymembers",[]))
        cont = j.get("continue")
        if not cont: break
        params.update(cont)
        time.sleep(0.05)
    return files

def get_imageinfo(session, title):
    params = {
        "action":"query","prop":"imageinfo","titles":title,
        "iiprop":"url|size|mime|extmetadata|commonmetadata"
    }
    j = mw_get_json(session, params)
    pages = j.get("query",{}).get("pages",{})
    for _, p in pages.items():
        return (p.get("imageinfo") or [{}])[0]
    return {}

def main(args):
    random.seed(42)
    os.makedirs("data/raw", exist_ok=True)
    meta_rows = []
    s = make_session()

    for cls, cat in CATEGORIES.items():
        outdir = os.path.join("data/raw", cls); os.makedirs(outdir, exist_ok=True)
        cms = list_category_files(s, cat)
        if not cms:
            print(f"[WARN] No files listed for {cat}. See {LOG_FILE} if error occurred.", file=sys.stderr)
        random.shuffle(cms)
        kept = 0
        for cm in tqdm(cms, desc=f"Fetching {cls}"):
            title = cm["title"]  # "File:..."
            t_low = title.lower()
            if any(s in t_low for s in EXCLUDE_SUBSTR):
                continue
            ii = get_imageinfo(s, title)
            url = ii.get("url"); w = ii.get("width",0); h = ii.get("height",0)
            if not url: continue
            ext = os.path.splitext(url)[1]
            if ext not in VALID_EXT: continue
            if w < args.min_size or h < args.min_size: continue

            fname = clean(os.path.basename(title.replace("File:",""))) + ext.lower()
            dest = os.path.join(outdir, fname)
            if os.path.exists(dest): 
                continue

            try:
                with s.get(url, stream=True, timeout=60) as resp:
                    resp.raise_for_status()
                    with open(dest,"wb") as f:
                        for chunk in resp.iter_content(1<<14):
                            f.write(chunk)
                meta_rows.append([
                    cls, fname, url, w, h,
                    ii.get("descriptionurl",""),
                    ii.get("extmetadata",{}).get("Artist",{}).get("value",""),
                    ii.get("extmetadata",{}).get("LicenseShortName",{}).get("value","")
                ])
                kept += 1
                if kept >= args.max_per_class: break
                time.sleep(0.03)
            except Exception as e:
                # Sessiz geç; çok sıkı olma
                continue

    with open("data/SOURCES.csv","w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class","file","url","width","height","pageurl","artist","license"])
        w.writerows(meta_rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=150)
    ap.add_argument("--min-size", type=int, default=400)
    args = ap.parse_args()
    main(args)
