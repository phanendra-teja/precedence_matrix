#!/usr/bin/env python3
"""
Stage 7 - THE PRECEDENCE MATRIX from annotations alone.   M[A,B] = 1  means  "A must be removed / disconnected before B".

    python s7_precedence.py --instances work\\instances\\instances.csv --parts work\\review\\part_groups.csv ^
        --verified work\\review\\pair_review.csv --out work\\precedence

Input is ONLY what you annotated (class + mask per part). Evidence tiers, strongest first (every cell records its tier):
  V  verified   - you looked at that exact pair and labelled it (pair_review.csv)                     -> CONFIRMED matrix
  C  pair rule  - an explicit exception in class_precedence.csv ("Cable before Motherboard when touching")
  S  stage rule - LAYERED DISASSEMBLY: every class has a removal stage in class_stage.csv (0 = screws first ... 5 = battery last).
                  Two parts that are neighbours in the image (gap <= limit) get  earlier-stage -> later-stage.
                  Equal/unknown stage => no edge (stays '?').
  M  model      - learned from your verified pairs (s8), used only above --threshold
  G  far apart  - optional (--far_zero PX): 'no direct dependency'
Cell values: 1 = A before B, 0 = no direct dependency, ? = unknown (never silently 0), ! = contradiction.
S edges are a domain assumption (typical phone layering), NOT proof - s9_evaluate.py measures how often they match your labels.
Conflicts are settled by tier; a cycle is broken only by dropping its weakest non-verified edge, and every drop is logged.
Outputs per image: precedence_confirmed_*.csv, precedence_inferred_direct/source/confidence/transitive.csv, precedence_edges.csv,
removal_levels.csv, report.txt.  Globally: all_edges.csv, summary.json, next_to_label.csv, eval_sample.csv, pair_suggestions.csv.
"""
import argparse, csv, itertools, json, random
from collections import defaultdict, Counter
from pathlib import Path
import networkx as nx
from common import short_label
from build_precedence import build_matrices

TIER_RANK = {"V": 4, "C": 3, "S": 2, "M": 1, "G": 0}
VALID = {"A_BEFORE_B", "B_BEFORE_A", "NO_DIRECT_DEPENDENCY"}
CONF_MAP = {"high": 1.0, "medium": 0.8, "med": 0.8, "low": 0.6, "": 0.8}
NOTE = ("Inferred cells (tiers C/S/M) are hypotheses; only 'confirmed' cells are verified by you. "
        "Precedence prediction has not yet been quantitatively validated unless s9_evaluate.py reports it on pairs you labelled.")
RULES_HEADER = "rule_id,before_class,after_class,max_gap_px,status,support,confidence,rationale\n"
STAGE_KEYWORDS = [(0, ["screw", "bolt", "fastener"]), (1, ["shield", "cover", "bracket", "plate", "cage", "coil"]),
                  (2, ["connector", "connection", "wire", "cable", "flex", "ribbon", "fpc"]),
                  (3, ["graphite", "sheet", "foil", "tape", "sticker", "film", "adhesive", "pad"]),
                  (4, ["camera", "flash", "speaker", "mic", "sensor", "module", "antenna", "button", "vibrat"]),
                  (5, ["battery", "cell", "motherboard", "mainboard", "pcb", "logic", "board"])]
STAGE_NAMES = {0: "fasteners", 1: "covers/shields", 2: "cables & connectors", 3: "stuck-on sheets", 4: "modules", 5: "battery/board"}


def read_csv(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


# ------------------------------------------------------------------ rules
def load_rules(path):
    out = []
    if not Path(path).exists():
        return out
    for r in read_csv(path):
        if (r.get("status") or "").strip().upper() in ("OFF", "DISABLED", "REJECTED"):
            continue
        r["max_gap_px"] = float(r.get("max_gap_px") or 6); r["confidence"] = float(r.get("confidence") or 0.5)
        out.append(r)
    return out


def rule_match(rules, ca, cb, gap):
    for ru in rules:
        before_ok = ca != cb if ru["before_class"] == "*" else ca == ru["before_class"]
        after_ok = cb != ca if ru["after_class"] == "*" else cb == ru["after_class"]
        if before_ok and after_ok and gap <= ru["max_gap_px"]:
            return ru
    return None


def default_stage(name):
    low = name.lower()
    for stage, kws in STAGE_KEYWORDS:
        if any(k in low for k in kws):
            return stage
    return None


def ensure_stage_table(path, classes):
    """Create / extend class_stage.csv so EVERY class in your data has a row (guessed from its name; edit it!)."""
    path = Path(path); rows = read_csv(path) if path.exists() else []
    have = {r["class_name"] for r in rows}
    added = []
    for c in sorted(set(classes) - have):
        st = default_stage(c)
        rows.append({"class_name": c, "stage": "" if st is None else st, "max_gap_px": 15 if st == 0 else "", "status": "DRAFT",
                     "rationale": (f"guessed from the name: {STAGE_NAMES[st]}" if st is not None else "name not recognised - set a stage or leave blank")})
        added.append(c)
    if added or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["class_name", "stage", "max_gap_px", "status", "rationale"]); w.writeheader(); w.writerows(rows)
    return added


def load_stages(path):
    out = {}
    if path and Path(path).exists():
        for r in read_csv(path):
            st = (r.get("stage") or "").strip()
            off = (r.get("status") or "").strip().upper() in ("OFF", "DISABLED")
            out[r["class_name"]] = {"stage": None if (not st or off) else float(st),
                                    "gap": float(r["max_gap_px"]) if (r.get("max_gap_px") or "").strip() else None}
    return out


def stage_edge(stages, ca, cb, gap, default_gap):
    """'AB' if A (class ca) should come off before B, 'BA' for the reverse, None if no stage evidence / not neighbours."""
    sa, sb = stages.get(ca), stages.get(cb)
    if not sa or not sb or sa["stage"] is None or sb["stage"] is None or sa["stage"] == sb["stage"]:
        return None
    before, direction = (sa, "AB") if sa["stage"] < sb["stage"] else (sb, "BA")
    gate = before["gap"] if before["gap"] is not None else default_gap
    return direction if gap <= gate else None


def predict_rules(rules, stages, ca, cb, gap, default_gap):
    """Rule-only prediction for an ordered pair of classes -> ('AB'|'BA'|'CONFLICT'|None, rule id or 'STAGE')."""
    h_ab, h_ba = rule_match(rules, ca, cb, gap), rule_match(rules, cb, ca, gap)
    if h_ab and h_ba:
        return "CONFLICT", h_ab["rule_id"] + "/" + h_ba["rule_id"]
    if h_ab:
        return "AB", h_ab["rule_id"]
    if h_ba:
        return "BA", h_ba["rule_id"]
    d = stage_edge(stages, ca, cb, gap, default_gap)
    return (d, "STAGE") if d else (None, "")


# ------------------------------------------------------------------ evidence collection
def _rk(e):
    return (TIER_RANK[e["tier"]], e["conf"])


def _put(d, key, e):
    if key not in d or _rk(e) > _rk(d[key]):
        d[key] = e


def collect(ids, cls, pairs, rules, stages, stage_gap, stage_conf, ver_rows, bundle, thr, model_gap, far_zero):
    """Instance-level evidence for ONE image -> (edges, zeros, rule_conflicts, decided_by_V)."""
    edges, zeros, rule_conflicts, decided = {}, {}, [], set()
    for r in ver_rows:
        lab = (r.get("label") or "").strip().upper()
        if lab not in VALID:
            continue
        a, b = r["instance_a_id"], r["instance_b_id"]
        e = {"tier": "V", "conf": CONF_MAP.get((r.get("confidence") or "").strip().lower(), 0.8),
             "src": ((r.get("evidence_type") or "verified") + (": " + r["evidence_reference"] if r.get("evidence_reference") else "")).strip()}
        if lab == "A_BEFORE_B":
            edges[(a, b)] = e
        elif lab == "B_BEFORE_A":
            edges[(b, a)] = e
        else:
            zeros[frozenset((a, b))] = e
        decided.add(frozenset((a, b)))
    for i, u in enumerate(ids):
        for v in ids[i + 1:]:
            k = frozenset((u, v))
            if k in decided or (u, v) not in pairs:
                continue
            gap = float(pairs[(u, v)]["contour_distance"]); added = False
            hit_uv, hit_vu = rule_match(rules, cls[u], cls[v], gap), rule_match(rules, cls[v], cls[u], gap)
            if hit_uv and hit_vu:
                rule_conflicts.append((u, v, hit_uv["rule_id"], hit_vu["rule_id"]))
            else:
                for (x, y), ru in (((u, v), hit_uv), ((v, u), hit_vu)):
                    if ru:
                        _put(edges, (x, y), {"tier": "C", "conf": ru["confidence"], "src": ru["rule_id"]}); added = True
            d = stage_edge(stages, cls[u], cls[v], gap, stage_gap)
            if d and not (hit_uv or hit_vu):                           # explicit pair rules beat the generic stage order
                x, y = (u, v) if d == "AB" else (v, u)
                _put(edges, (x, y), {"tier": "S", "conf": stage_conf,
                                     "src": f"stage {cls[x]}({stages[cls[x]]['stage']:g})<{cls[y]}({stages[cls[y]]['stage']:g})"}); added = True
            if bundle is not None and gap <= model_gap:
                from pfeat import predict_pair
                p = predict_pair(bundle, pairs[(u, v)], pairs[(v, u)])
                top = max(p["AB"], p["BA"])
                if top >= thr and top > p["NONE"]:
                    x, y = (u, v) if p["AB"] >= p["BA"] else (v, u)
                    _put(edges, (x, y), {"tier": "M", "conf": round(top, 3), "src": "model"}); added = True
                elif p["NONE"] >= thr:
                    _put(zeros, k, {"tier": "M", "conf": round(p["NONE"], 3), "src": "model"}); added = True
            if not added and far_zero > 0 and gap >= far_zero:
                zeros[k] = {"tier": "G", "conf": 0.5, "src": f"gap>={far_zero:g}px"}
    return edges, zeros, rule_conflicts, decided


def resolve(edges, zeros, part_of):
    """Merge to PART level, settle conflicts by tier, break cycles by dropping the weakest non-verified edge (logged)."""
    log = {"overridden": [], "conflicts": [], "dropped_for_cycle": [], "unresolved_cycles": []}
    pe, pz = {}, {}
    for (u, v), e in edges.items():
        pu, pv = part_of.get(u, u), part_of.get(v, v)
        if pu != pv and ((pu, pv) not in pe or _rk(e) > _rk(pe[(pu, pv)])):
            pe[(pu, pv)] = dict(e)
    for k, z in zeros.items():
        a, b = tuple(k); pa, pb = part_of.get(a, a), part_of.get(b, b)
        kk = frozenset((pa, pb))
        if pa != pb and (kk not in pz or _rk(z) > _rk(pz[kk])):
            pz[kk] = dict(z)
    conflict_pairs = set()
    for (u, v) in list(pe):
        if (u, v) not in pe or (v, u) not in pe:
            continue
        e1, e2 = pe[(u, v)], pe[(v, u)]
        if _rk(e1) == _rk(e2):
            log["conflicts"].append((u, v, "opposing edges, same tier")); conflict_pairs.add(frozenset((u, v)))
            del pe[(u, v)], pe[(v, u)]
        elif _rk(e1) > _rk(e2):
            log["overridden"].append((v, u, e2["tier"], "overridden by " + e1["tier"])); del pe[(v, u)]
        else:
            log["overridden"].append((u, v, e1["tier"], "overridden by " + e2["tier"])); del pe[(u, v)]
    for (u, v) in list(pe):
        k = frozenset((u, v))
        if (u, v) not in pe or k not in pz:
            continue
        e, z = pe[(u, v)], pz[k]
        if TIER_RANK[z["tier"]] > TIER_RANK[e["tier"]]:
            log["overridden"].append((u, v, e["tier"], "edge overridden by no-dependency " + z["tier"])); del pe[(u, v)]
        elif TIER_RANK[e["tier"]] > TIER_RANK[z["tier"]]:
            del pz[k]
        else:
            log["conflicts"].append((u, v, "edge vs no-dependency, same tier")); conflict_pairs.add(k)
            del pe[(u, v)]; del pz[k]
    G = nx.DiGraph(); G.add_edges_from(pe)
    for _ in range(1000):
        cycles = list(itertools.islice(nx.simple_cycles(G), 200))
        if not cycles:
            break
        fixable = [c for c in cycles if any(pe[x]["tier"] != "V" for x in zip(c, c[1:] + c[:1]))]
        if not fixable:
            log["unresolved_cycles"] = cycles; break
        c = fixable[0]
        weakest = min((x for x in zip(c, c[1:] + c[:1]) if pe[x]["tier"] != "V"), key=lambda x: _rk(pe[x]))
        log["dropped_for_cycle"].append((*weakest, pe[weakest]["tier"], pe[weakest]["conf"], pe[weakest]["src"]))
        G.remove_edge(*weakest); del pe[weakest]
    return pe, pz, conflict_pairs, log


def write_matrix(path, labels, cell):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow([""] + labels)
        for u in labels:
            w.writerow([u] + [cell(u, v) for v in labels])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True, type=Path)
    ap.add_argument("--pairs", type=Path, help="default: pair_features.csv next to instances.csv")
    ap.add_argument("--verified", type=Path, help="pair_review.csv (your labels)")
    ap.add_argument("--parts", type=Path, help="part_groups.csv (merge fragments of one physical part)")
    ap.add_argument("--rules", default=Path("work/class_precedence.csv"), type=Path, help="optional pair-level exceptions")
    ap.add_argument("--stages_csv", default=Path("work/class_stage.csv"), type=Path, help="class removal stages (the main assumption - edit it)")
    ap.add_argument("--stage_gap", type=float, default=8.0, help="px: default max gap for two parts to count as neighbours")
    ap.add_argument("--stage_conf", type=float, default=0.5, help="nominal confidence of a stage edge (see s9_evaluate for the measured precision)")
    ap.add_argument("--no_stages", action="store_true", help="ignore class_stage.csv")
    ap.add_argument("--model", type=Path, help="precedence_model.joblib from s8_train_model.py")
    ap.add_argument("--prefer_model", action="store_true", help="rank model (M) above stage rules (S) - use once s8 shows it beats them")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--model_gap", type=float, default=40.0)
    ap.add_argument("--far_zero", type=float, default=0.0)
    ap.add_argument("--out", default=Path("work/precedence"), type=Path)
    ap.add_argument("--next_n", type=int, default=60)
    ap.add_argument("--eval_sample", type=int, default=100, help="size of the random blind evaluation sample to label")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.prefer_model:
        TIER_RANK["M"] = 2.5
    pairs_path = a.pairs or a.instances.parent / "pair_features.csv"
    if not a.rules.exists():
        a.rules.parent.mkdir(parents=True, exist_ok=True); a.rules.write_text(RULES_HEADER, encoding="utf-8")
    rules = load_rules(a.rules)

    inst = defaultdict(list); cls = {}
    for r in read_csv(a.instances):
        inst[r["image_id"]].append(r); cls[r["instance_id"]] = r["class_name"]
    stages = {}
    if not a.no_stages:
        added = ensure_stage_table(a.stages_csv, set(cls.values()))
        if added:
            print(f"class_stage.csv: added rows for {added} - CHECK the stages in {a.stages_csv} (0 = removed first ... 5 = last)")
        stages = load_stages(a.stages_csv)
    bundle = None
    if a.model:
        import joblib
        bundle = joblib.load(a.model)
    pairs = {(r["instance_a_id"], r["instance_b_id"]): r for r in read_csv(pairs_path)}
    part_of = {}
    if a.parts and a.parts.exists():
        part_of = {r["instance_id"]: r["part_id"].strip() for r in read_csv(a.parts) if (r.get("part_id") or "").strip()}
    ver_all = read_csv(a.verified) if a.verified and a.verified.exists() else []
    ver_by_img = defaultdict(list); review_info = {}
    for r in ver_all:
        ver_by_img[r["image_id"]].append(r)
        for k in ((r["instance_a_id"], r["instance_b_id"]), (r["instance_b_id"], r["instance_a_id"])):
            review_info[k] = (r.get("pair_no", ""), r.get("review_image", ""))
    a.out.mkdir(parents=True, exist_ok=True)

    all_edges, summary, todo, sugg_rows, pool = [], {}, [], [], []
    class_pair_freq = Counter()
    for img, rows in inst.items():
        rows = sorted(rows, key=lambda r: r["instance_id"])
        ids = [r["instance_id"] for r in rows]
        device = rows[0]["device_id"]
        members = defaultdict(list)
        for i in ids:
            members[part_of.get(i, i)].append(i)
        nodes = list(members)
        label = {p: "+".join(short_label(m, cls[m]) for m in ms) for p, ms in members.items()}
        edges, zeros, rule_conf, decided = collect(ids, cls, pairs, rules, stages, a.stage_gap, a.stage_conf,
                                                   ver_by_img.get(img, []), bundle, a.threshold, a.model_gap, a.far_zero)
        pe, pz, conflict_pairs, log = resolve(edges, zeros, part_of)
        G = nx.DiGraph(); G.add_nodes_from(nodes); G.add_edges_from(pe)
        acyclic = nx.is_directed_acyclic_graph(G)

        vpairs = []
        for r in ver_by_img.get(img, []):
            pa, pb = part_of.get(r["instance_a_id"], r["instance_a_id"]), part_of.get(r["instance_b_id"], r["instance_b_id"])
            if pa != pb:
                vpairs.append({"a": pa, "b": pb, "label": r.get("label", "")})
        cres = build_matrices(nodes, vpairs)
        d = a.out / img; d.mkdir(parents=True, exist_ok=True)
        L = [label[p] for p in nodes]; inv = {label[p]: p for p in nodes}
        write_matrix(d / "precedence_confirmed_direct.csv", L, lambda u, v: cres["direct"][inv[u]][inv[v]])
        write_matrix(d / "precedence_confirmed_transitive.csv", L, lambda u, v: cres["transitive"][inv[u]][inv[v]])
        paths = {}
        if acyclic:
            for u in nodes:
                for v in nx.descendants(G, u):
                    if (u, v) not in pe:
                        paths[(u, v)] = nx.shortest_path(G, u, v)

        def cell(u, v, kind):
            pu, pv = inv[u], inv[v]
            if pu == pv:
                return "0" if kind == "val" else "-"
            if frozenset((pu, pv)) in conflict_pairs:
                return "!" if kind != "conf" else ""
            if (pu, pv) in pe:
                e = pe[(pu, pv)]
                return {"val": "1", "src": e["tier"], "conf": f"{e['conf']:.2f}"}[kind]
            if frozenset((pu, pv)) in pz:
                z = pz[frozenset((pu, pv))]
                return {"val": "0", "src": z["tier"], "conf": f"{z['conf']:.2f}"}[kind]
            return "?" if kind != "conf" else ""
        write_matrix(d / "precedence_inferred_direct.csv", L, lambda u, v: cell(u, v, "val"))
        write_matrix(d / "precedence_inferred_source.csv", L, lambda u, v: cell(u, v, "src"))
        write_matrix(d / "precedence_inferred_confidence.csv", L, lambda u, v: cell(u, v, "conf"))
        write_matrix(d / "precedence_inferred_transitive.csv", L,
                     lambda u, v: "1" if (inv[u], inv[v]) in paths or (inv[u], inv[v]) in pe else cell(u, v, "val"))
        with open(d / "precedence_edges.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "device_id", "from", "to", "kind", "tier", "confidence", "source"])
            for (u, v), e in pe.items():
                row = [img, device, label[u], label[v], "direct", e["tier"], e["conf"], e["src"]]
                w.writerow(row); all_edges.append(row)
            for (u, v), pth in paths.items():
                weakest = min([pe[x] for x in zip(pth, pth[1:])], key=_rk)
                row = [img, device, label[u], label[v], "implied", weakest["tier"], weakest["conf"], "path: " + " > ".join(label[p] for p in pth)]
                w.writerow(row); all_edges.append(row)
        levels = [sorted(label[p] for p in g) for g in nx.topological_generations(G)] if acyclic else []
        with open(d / "removal_levels.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["level", "parts_with_no_remaining_prerequisite"])
            for k, g in enumerate(levels):
                w.writerow([k, " | ".join(g)])

        adj_parts = set()
        for u, v in itertools.combinations(ids, 2):
            if (u, v) in pairs and pairs[(u, v)]["visual_relation"] != "separate":
                pu, pv = part_of.get(u, u), part_of.get(v, v)
                if pu != pv:
                    adj_parts.add(frozenset((pu, pv))); class_pair_freq[tuple(sorted((cls[u], cls[v])))] += 1
        decided_parts = {frozenset(k) for k in pe} | set(pz) | conflict_pairs
        tiers = Counter(e["tier"] for e in pe.values())
        summary[img] = {"parts": len(nodes), "edges_by_tier": dict(tiers), "no_dependency_pairs": len(pz),
                        "confirmed_edges": len(cres["edges"]), "adjacent_part_pairs": len(adj_parts),
                        "adjacent_pairs_with_a_decision": len(adj_parts & decided_parts), "acyclic": acyclic,
                        "conflicts": len(log["conflicts"]) + len(rule_conf), "dropped_for_cycle": len(log["dropped_for_cycle"]),
                        "unresolved_cycles": len(log["unresolved_cycles"]), "levels": len(levels)}
        rep = [f"image: {img}   device: {device}   parts: {len(nodes)} (from {len(ids)} instances)",
               f"edges by tier: {dict(tiers)}   (V verified, C pair rule, S stage rule, M model)   no-dependency pairs: {len(pz)}",
               f"adjacent part pairs: {len(adj_parts)}, with a direct decision: {len(adj_parts & decided_parts)}",
               f"rule conflicts: {rule_conf}", f"conflicts resolved to '!': {log['conflicts']}",
               f"overridden by a higher tier: {len(log['overridden'])}",
               f"edges dropped to break cycles (weakest non-verified first): {log['dropped_for_cycle']}",
               f"UNRESOLVED cycles among verified edges (fix your labels): {log['unresolved_cycles']}",
               "graph: " + ("ACYCLIC" if acyclic else "HAS CYCLES")]
        for k, g in enumerate(levels):
            rep.append(f"  level {k}: {', '.join(g)}")
        (d / "report.txt").write_text("\n".join(rep + ["", NOTE]), encoding="utf-8")

        for u, v in itertools.combinations(ids, 2):
            if (u, v) not in pairs or pairs[(u, v)]["visual_relation"] == "separate":
                continue
            pu, pv = part_of.get(u, u), part_of.get(v, v)
            if pu == pv:
                continue
            e = pe.get((pu, pv)) or pe.get((pv, pu))
            sug = "" if e is None else ("A_BEFORE_B" if (pu, pv) in pe else "B_BEFORE_A")
            pn, ri = review_info.get((u, v), ("", ""))
            la, lb = short_label(u, cls[u]), short_label(v, cls[v])
            if frozenset((u, v)) not in decided:
                reason = "no rule or model covers it" if e is None else f"only {e['tier']} evidence (conf {e['conf']:.2f}, {e['src']})"
                todo.append({"image_id": img, "a": u, "b": v, "class_pair": tuple(sorted((cls[u], cls[v]))),
                             "score": 1.0 if e is None else 1 - e["conf"], "reason": reason, "relation": pairs[(u, v)]["visual_relation"]})
                pool.append({"pair_no": pn, "review_image": ri, "image_id": img, "label_a": la, "label_b": lb,
                             "class_pair": tuple(sorted((cls[u], cls[v])))})
            sugg_rows.append([pn, img, la, lb, pairs[(u, v)]["visual_relation"], sug or "NO_EDGE", e["tier"] if e else "", e["conf"] if e else "", e["src"] if e else ""])

    mx = max(class_pair_freq.values()) if class_pair_freq else 1
    for t in todo:
        t["rank_score"] = t["score"] * (0.5 + 0.5 * class_pair_freq[t["class_pair"]] / mx)
    todo.sort(key=lambda t: -t["rank_score"])
    per_cp, chosen = Counter(), []
    for t in todo:
        if per_cp[t["class_pair"]] < max(3, a.next_n // 6) and len(chosen) < a.next_n:
            per_cp[t["class_pair"]] += 1; chosen.append(t)
    with open(a.out / "next_to_label.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "pair_no", "review_image", "image_id", "label_a", "label_b", "relation", "reason"])
        for k, t in enumerate(chosen, 1):
            pn, ri = review_info.get((t["a"], t["b"]), ("", ""))
            w.writerow([k, pn, ri, t["image_id"], short_label(t["a"], cls[t["a"]]), short_label(t["b"], cls[t["b"]]), t["relation"], t["reason"]])
    # blind random sample (round-robin over class pairs) -> label THESE first, without looking at suggestions: unbiased precision estimate
    rng = random.Random(a.seed); buckets = defaultdict(list)
    pool_n = Counter(p["class_pair"] for p in pool)
    for p in pool:
        buckets[p["class_pair"]].append(p)
    for b in buckets.values():
        rng.shuffle(b)
    sample, keys = [], sorted(buckets)
    while len(sample) < a.eval_sample and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(sample) < a.eval_sample:
                sample.append(buckets[k].pop())
    drawn = Counter(p["class_pair"] for p in sample)
    with open(a.out / "eval_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["pair_no", "review_image", "image_id", "label_a", "label_b", "class_pair", "weight"])
        for p in sample:      # weight = how many real neighbouring pairs of this class pair each sampled pair stands for
            w.writerow([p["pair_no"], p["review_image"], p["image_id"], p["label_a"], p["label_b"], " + ".join(p["class_pair"]),
                        round(pool_n[p["class_pair"]] / drawn[p["class_pair"]], 3)])
    with open(a.out / "pair_suggestions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["pair_no", "image_id", "label_a", "label_b", "relation", "engine_says", "tier", "confidence", "source"])
        w.writerows(sugg_rows)
    with open(a.out / "all_edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["image_id", "device_id", "from", "to", "kind", "tier", "confidence", "source"]); w.writerows(all_edges)
    tot = Counter()
    for s in summary.values():
        for k in ("confirmed_edges", "adjacent_part_pairs", "adjacent_pairs_with_a_decision", "conflicts", "dropped_for_cycle", "unresolved_cycles"):
            tot[k] += s[k]
        for t, n in s["edges_by_tier"].items():
            tot["edges_" + t] += n
    json.dump({"totals": dict(tot), "per_image": summary}, open(a.out / "summary.json", "w"), indent=2)
    print(json.dumps({"totals": dict(tot)}, indent=2))
    print(f"matrices -> {a.out}\\<image>\\   worklists -> next_to_label.csv, eval_sample.csv ({len(sample)} pairs), pair_suggestions.csv\n{NOTE}")


if __name__ == "__main__":
    main()
