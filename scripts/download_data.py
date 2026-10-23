"""Download the raw files of the twelve datasets into data/raw and check their SHA-256 hashes.

  python scripts/download_data.py
"""
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
UCI = "https://archive.ics.uci.edu/static/public/"
MOTO = ("https://raw.githubusercontent.com/Malware-Hunter/Dataset-Matters/main/ReABTIC2021/"
        "0_Datasets_Sources/MotoDroid_22K%20%282021%29.csv")

# target file: (url, member of the zip archive or None for a plain file, sha256)
FILES = {
    "car/car.data": (UCI + "19/car+evaluation.zip", "car.data",
                     "b703a9ac69f11e64ce8c223c0a40de4d2e9d769f7fb20be5f8f2e8a619893d83"),
    "nursery/nursery.data": (UCI + "76/nursery.zip", "nursery.data",
                             "8e0389c3dd37590248a921c2726d869ee96b817761a35eb8416afa24f31f931d"),
    "tictactoe/tic-tac-toe.data": (UCI + "101/tic+tac+toe+endgame.zip", "tic-tac-toe.data",
                                   "a996a756484d0cf8cbad1680d823aadbfcff8ae2d78c450e6999362d580301c5"),
    "krk/krkopt.data": (UCI + "23/chess+king+rook+vs+king.zip", "krkopt.data",
                        "497b61d20d3d9e2ce310f9f4d2ca92dbdb0691d9e449d983ec6f7baf387dc7a9"),
    "monks/monks-1.test": (UCI + "70/monk+s+problems.zip", "monks-1.test",
                           "e501ada3b21d876ab86f3e1614747bd2edc443a685f229a25e480aa1dea629f6"),
    "krkp/kr-vs-kp.data": (UCI + "22/chess+king+rook+vs+king+pawn.zip", "kr-vs-kp.data",
                           "70a5540902b4063d3977d3e820ac5b9fe5624eb5d93709eee5d50d7a94fca205"),
    "mushroom/agaricus-lepiota.data": (UCI + "73/mushroom.zip", "agaricus-lepiota.data",
                                       "e65d082030501a3ebcbcd7c9f7c71aa9d28fdfff463bf4cf4716a3fe13ac360e"),
    "phish/Training Dataset.arff": (UCI + "327/phishing+websites.zip", "Training Dataset.arff",
                                    "3bcf93b1612306537ee2737751f1bac185b2a77886ab376644c3bf0c7f7cb3e6"),
    "naticus/data.csv": (UCI + "722/naticusdroid+android+permissions+dataset.zip", "data.csv",
                         "5764687009af15605f46941902d6a1c4c1a07618dfefb94a0275edb4869780ba"),
    "splice/splice.data": (UCI + "69/molecular+biology+splice+junction+gene+sequences.zip", "splice.data",
                           "ebbae10c85d3c285e2a2489a37e42bc2ee8d91d15005e391a9962fa3389a114f"),
    "vote/house-votes-84.data": (UCI + "105/congressional+voting+records.zip", "house-votes-84.data",
                                 "c87c14110a5ba91d4a1e313ec7392824458152bf071fa5f5452340488337936e"),
    "moto.csv": (MOTO, None, "ec2042679414fdda1368e347c14c12730e30bee9179b29735a25e4b01b4fef06"),
}


def fetch(url, member):
    with urllib.request.urlopen(url) as r:
        body = r.read()
    if member is None:
        return body
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        name = next(n for n in z.namelist() if n.split("/")[-1] == member)
        return z.read(name)


def main():
    for target, (url, member, sha) in FILES.items():
        path = RAW / target
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(fetch(url, member))
        ok = hashlib.sha256(path.read_bytes()).hexdigest() == sha
        print(f"{target:35s} {'ok' if ok else 'HASH MISMATCH, results may differ'}")


if __name__ == "__main__":
    main()
