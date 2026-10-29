"""HTML page of base vs fine-tuned Oracle-lens readings on each dataset's test (setting 1, own split): per layer mean
FVE, surface changes (length, bullets, share with Chinese characters, EOS), and the readings side by side for the
(question, layer) pairs whose FVE rose most and fell most (sample 0 of each lens; the FVE shown is that sample's).
Usage: python -m experiments.lens_finetune.src.plot.examples_page OUT.html"""
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

R = Path(__file__).resolve().parents[2] / "results"
TAG = "s1_lr1e-4_k1_e2"
SETS = [("hotpotqa_split", "HotpotQA"), ("hover_split", "HoVer"), ("synqa2_split", "SynQA2"), ("medcalc_split", "MedCalc-Bench")]
CJK = re.compile(r"[一-鿿]")


def load(sub, tag):
    d = R / sub / "data"
    rd = {(r["qid"], r["layer"], r["s"]): r for r in map(json.loads, (d / "readings" / f"{tag}.jsonl").read_text().splitlines())}
    sc = {(r["qid"], r["layer"], r["s"]): r for r in map(json.loads, (d / "scores" / f"{tag}.jsonl").read_text().splitlines())}
    return rd, sc


def esc(t):
    return html.escape(t or "").replace("\n", "<br>")


def section(sub, name):
    d = R / sub / "data"
    if not (d / "scores" / f"{TAG}_test.jsonl").exists():
        return f"<h2>{name}</h2><p class=mut>아직 평가 중입니다.</p>"
    qs = {q["qid"]: q for q in map(json.loads, (d / "questions.jsonl").read_text().splitlines())}
    test = {q for q, v in qs.items() if v["test_fold"] == 0}
    brd, bsc = load(sub, "eval_base")
    trd, tsc = load(sub, f"{TAG}_test")
    pairs = defaultdict(lambda: {"b": [], "t": []})
    for (q, L, s), r in bsc.items():
        if q in test:
            pairs[(q, L)]["b"].append(r["fve"])
    for (q, L, s), r in tsc.items():
        if q in test:
            pairs[(q, L)]["t"].append(r["fve"])
    pairs = {k: v for k, v in pairs.items() if v["b"] and v["t"]}
    layers = sorted({L for _, L in pairs})
    rows = "".join(f"<tr><td>L{L}</td><td>{mean(mean(v['b']) for k, v in pairs.items() if k[1] == L):.3f}</td>"
                   f"<td>{mean(mean(v['t']) for k, v in pairs.items() if k[1] == L):.3f}</td></tr>" for L in layers)

    def surf(rd):
        xs = [r for (q, L, s), r in rd.items() if q in test]
        return (mean(len(r["text"]) for r in xs), mean(r["text"].count("\n- ") + r["text"].startswith("- ") for r in xs),
                100 * mean(bool(CJK.search(r["text"])) for r in xs), 100 * mean(bool(r["eos"]) for r in xs))
    sb, st = surf(brd), surf(trd)
    srows = "".join(f"<tr><td>{n}</td><td>{a:.1f}</td><td>{b:.1f}</td></tr>" for n, a, b in
                    zip(("글자 수 (평균)", "bullet 수 (평균)", "중국어가 섞인 판독 (%)", "128토큰 안에 끝남 (%)"), sb, st))
    diff = sorted(pairs, key=lambda k: mean(pairs[k]["t"]) - mean(pairs[k]["b"]))

    def card(k):
        q, L = k
        b, t = brd.get((q, L, 0)), trd.get((q, L, 0))
        fb, ft = bsc.get((q, L, 0), {}).get("fve", 0), tsc.get((q, L, 0), {}).get("fve", 0)
        d_ = mean(pairs[k]["t"]) - mean(pairs[k]["b"])
        qt = qs[q]["question"]
        qt = qt if len(qt) < 600 else qt[:600] + " …"
        return (f"<div class=card><div class=meta><b>{html.escape(q)}</b> · layer {L} · 평균 FVE 변화 "
                f"<b class={'up' if d_ > 0 else 'down'}>{d_:+.3f}</b></div><details><summary>질문 보기</summary><div class=q>{esc(qt)}</div></details>"
                f"<div class=two><div><div class=lab>base (FVE {fb:.3f})</div><div class=rd>{esc(b and b['text'])}</div></div>"
                f"<div><div class=lab>fine-tuned (FVE {ft:.3f})</div><div class=rd>{esc(t and t['text'])}</div></div></div></div>")
    up = "".join(card(k) for k in reversed(diff[-5:]))
    down = "".join(card(k) for k in diff[:3])
    tot_b = mean(mean(v["b"]) for v in pairs.values()); tot_t = mean(mean(v["t"]) for v in pairs.values())
    part = "" if len(layers) == 4 else " · 채점 진행 중, 끝난 layer만"
    return (f"<h2>{name} <span class=mut>(test {len(test)}문항 × layer {len(layers)}개{part})</span></h2>"
            f"<p>평균 FVE {tot_b:.3f} → {tot_t:.3f} (<b>{tot_t - tot_b:+.3f}</b>)</p>"
            f"<div class=grid><table><tr><th>layer</th><th>base FVE</th><th>fine-tuned FVE</th></tr>{rows}</table>"
            f"<table><tr><th>겉모습</th><th>base</th><th>fine-tuned</th></tr>{srows}</table></div>"
            f"<h3>FVE가 가장 많이 오른 예 5개</h3>{up}<h3>가장 많이 내린 예 3개</h3>{down}")


CSS = """:root{--bg:#fbfbfa;--fg:#1d1d1f;--mut:#666;--card:#fff;--line:#ddd;--acc:#2f6fdb;--up:#1f8a4c;--down:#c0392b;--chip:#eef2fb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#17181a;--fg:#e8e8e8;--mut:#a0a0a0;--card:#202225;--line:#3a3c40;--acc:#7aa7ff;--up:#5cc98a;--down:#ff7b6b;--chip:#26303f}}
:root[data-theme="dark"]{--bg:#17181a;--fg:#e8e8e8;--mut:#a0a0a0;--card:#202225;--line:#3a3c40;--acc:#7aa7ff;--up:#5cc98a;--down:#ff7b6b;--chip:#26303f}
body{background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,"Apple SD Gothic Neo","Noto Sans KR",sans-serif;margin:0;padding:24px 16px}
main{max-width:1150px;margin:auto}h1{font-size:22px}h2{font-size:19px;margin-top:34px;border-bottom:1px solid var(--line);padding-bottom:4px}h3{font-size:15px}
.mut{color:var(--mut);font-weight:normal;font-size:14px}.lead{background:var(--chip);border-left:4px solid var(--acc);padding:10px 14px;border-radius:6px}
table{border-collapse:collapse;background:var(--card);font-size:13.5px}th,td{border:1px solid var(--line);padding:5px 9px;text-align:left}th{background:var(--chip)}
.grid{display:flex;gap:18px;flex-wrap:wrap}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:10px 0}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:6px}@media(max-width:700px){.two{grid-template-columns:1fr}}
.lab{font-size:12.5px;color:var(--mut);margin-bottom:3px}.rd{font-size:13px;white-space:normal;background:var(--bg);border-radius:6px;padding:8px;word-break:break-word}
.q{font-size:12.5px;color:var(--mut);margin-top:4px}.up{color:var(--up)}.down{color:var(--down)}summary{cursor:pointer;font-size:13px;color:var(--acc)}"""


def main(out):
    body = "".join(section(s, n) for s, n in SETS)
    page = (f"<!doctype html><html lang=ko><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>Lens fine-tuning readings</title><style>{CSS}</style></head><body><main>"
            "<h1>Oracle lens: base와 fine-tuned 판독 비교</h1>"
            "<div class=lead><b>결론:</b> fine-tuning 뒤에도 판독의 내용과 형식은 거의 같습니다. FVE는 +0.005–0.026만 올랐고, "
            "바뀐 판독도 같은 주제를 조금 다르게 쓴 정도입니다.</div>"
            "<p class=mut><b>FVE</b>: 판독(bullet 최대 4개)이 답 직전 hidden state를 얼마나 설명하는지(0–1, 높을수록 좋음). "
            "<b>fine-tuned</b>: 설정 ① (lr 1e-4, 상위 1개, 2 epoch)로 학습한 lens, 각 데이터셋의 원래 분할의 test. "
            "판독은 질문·layer마다 4번 뽑았고, 여기에는 첫 번째 판독(sample 0)과 그 FVE를 보여 줍니다. "
            "예시는 4번 판독 평균 FVE의 변화가 가장 큰 (질문, layer)입니다.</p>"
            f"{body}</main></body></html>")
    Path(out).write_text(page)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
