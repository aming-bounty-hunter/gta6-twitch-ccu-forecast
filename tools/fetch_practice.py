"""Fetch only the four practice months documented in the kit's NOTICE.

The cache and extracted labels stay outside the repository. Verify every
overlapping historical observation against the kit before accepting a month.
No downloaded code is executed; main.py never uses this network tool.
"""

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_IDS = {"p00": "117671", "p01": "28264", "p02": "65645", "p03": "97296"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, default=ROOT.parent / "gta6-twitch-ccu-forecast-01-kit")
    parser.add_argument("--cache", type=Path, default=Path("/tmp/gta-practice-cache"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/gta-practice-answers.json"))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    answers, sources = {}, []
    for path in sorted((args.kit / "input").glob("p*.in")):
        case = json.loads(path.read_text())
        name = case["game"]["name"].replace(" ", "_")
        year, month = case["day_start"][:7].split("-")
        url = ("https://sullygnome.com/api/charts/linecharts/getconfig/GameViewers/30/0/"
               f"{SOURCE_IDS[case['id']]}/{urllib.parse.quote(name)}/%20/%20/{year}/{int(month)}/%20/0/")
        cache = args.cache / (case["id"] + ".json")
        if not cache.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                "Referer": "https://sullygnome.com/game/" + urllib.parse.quote(name),
                "Accept": "application/json", "X-Requested-With": "XMLHttpRequest"})
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
            json.loads(raw)
            cache.write_bytes(raw)
            time.sleep(1)
        raw = cache.read_bytes()
        chart = json.loads(raw)["data"]
        values = next(d["data"] for d in chart["datasets"] if d["label"] == "Average viewers")
        points = {t.replace(" ", "T") + ":00Z": v for t, v in zip(chart["labels"], values) if v is not None}
        matched = 0
        for row in case["history"]:
            if row["at"] in points:
                if row["viewers"] != points[row["at"]]:
                    raise ValueError(f"{case['id']}: source changed at {row['at']}")
                matched += 1
        if matched < 12 or points.get(case["day_start"]) != case["known"][0]["viewers"]:
            raise ValueError(f"{case['id']}: source does not match kit history/hour 1")
        truth = {int(t[11:13]) + 1: v for t, v in points.items()
                 if t[:10] == case["day_start"][:10] and int(t[11:13]) >= 1}
        if len(truth) < 12:
            raise ValueError(f"{case['id']}: fewer than 12 answers")
        answers[case["id"]] = {"hours": [{"hour": h, "viewers": y} for h, y in sorted(truth.items())]}
        sources.append({"id": case["id"], "url": url, "sha256": hashlib.sha256(raw).hexdigest(),
                        "history_matches": matched})
        print(f"{case['id']}: {len(truth)} labels, {matched} exact kit history matches")
    args.output.write_text(json.dumps({"cases": answers, "sources": sources}, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
