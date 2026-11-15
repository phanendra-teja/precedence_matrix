#!/usr/bin/env python3
"""
Step 5: confirmed precedence graph + direct/transitive tri-state matrices, per image.

    python build_precedence.py --instances outputs\\instances\\instances.csv ^
        --verified annotations\\precedence\\verified_pairs.csv --out outputs\\matrices

Matrix convention: M[A,B] = 1  -> A must be removed/disconnected before B (confirmed)
                   0           -> reviewed, explicitly NO_DIRECT_DEPENDENCY (and diagonal)
                   ?           -> unknown / not assessed  (NEVER turned into 0 automatically)
                   !           -> contradictory labels, needs review
Only labels A_BEFORE_B / B_BEFORE_A / NO_DIRECT_DEPENDENCY in verified_pairs.csv count.
UNKNOWN or blank labels are ignored. Node = "remove this instance" (prototype definition).
If verified_pairs.csv is missing/empty, every off-diagonal entry is '?', which is a valid result.
"""
import argparse, csv, json, itertools
from collections import defaultdict
from pathlib import Path
import networkx as nx

DISCLAIMER = "Precedence prediction has not yet been quantitatively validated."


def build_matrices(nodes, pairs):
    """nodes: list of ids. pairs: list of dicts with a, b, label, + optional evidence fields."""
    log = {"invalid": [], "duplicates": [], "contradictions": []}
    edges, nodep = {}, {}
    for p in pairs:
        a, b, lab = p["a"], p["b"], (p.get("label") or "").strip().upper()
        if lab in ("", "UNKNOWN"):
            continue
        if a not in nodes or b not in nodes:
            log["invalid"].append((a, b, "node not in instance list")); continue
        if a == b:
            log["invalid"].append((a, b, "self-loop")); continue
        if lab == "A_BEFORE_B":
            key = (a, b)
        elif lab == "B_BEFORE_A":
            key = (b, a)
        elif lab == "NO_DIRECT_DEPENDENCY":
            k = frozenset((a, b))
            if k in nodep:
                log["duplicates"].append((a, b, lab))
            else:
                nodep[k] = p
            continue
        else:
            log["invalid"].append((a, b, f"unknown label {lab}")); continue
        if key in edges:
            log["duplicates"].append((*key, lab)); continue
        edges[key] = p
    # contradictions: edge vs explicit no-dependency -> exclude edge, flag pair
    conflict_pairs = set()
    for (u, v) in list(edges):
        k = frozenset((u, v))
        if k in nodep:
            log["contradictions"].append((u, v, "edge vs NO_DIRECT_DEPENDENCY"))
            conflict_pairs.add(k); del edges[(u, v)]
    G = nx.DiGraph(); G.add_nodes_from(nodes); G.add_edges_from(edges)
    cycles = list(itertools.islice(nx.simple_cycles(G), 50))
    acyclic = not cycles
    res = {"edges": edges, "nodep": nodep, "conflict_pairs": conflict_pairs, "log": log,
           "cycles": cycles, "acyclic": acyclic,
           "components": [sorted(c) for c in nx.weakly_connected_components(G) if len(c) > 1]}
    paths = {}
    if acyclic:
        res["topological_order"] = list(nx.lexicographical_topological_sort(G))
        res["parallel_levels"] = [sorted(g) for g in nx.topological_generations(G)]
        res["no_confirmed_prerequisite"] = sorted(n for n in nodes if G.in_degree(n) == 0)
        for u in nodes:
            for v in nx.descendants(G, u):
                if (u, v) not in edges:
                    paths[(u, v)] = nx.shortest_path(G, u, v)
    res["transitive_paths"] = paths
    D = {a: {b: ("0" if a == b else "?") for b in nodes} for a in nodes}
    for k in nodep:
        a, b = tuple(k); D[a][b] = D[b][a] = "0"
    for k in conflict_pairs:
        a, b = tuple(k); D[a][b] = D[b][a] = "!"
    for (u, v) in edges:
        D[u][v] = "1"
    T = {a: dict(D[a]) for a in nodes}
    for (u, v) in paths:
        T[u][v] = "1"                      # explicit 'no DIRECT dependency' is consistent with implied precedence
    res["direct"], res["transitive"] = D, T
    return res


def write_csv_matrix(path, nodes, M):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow([""] + nodes)
        for a in nodes:
            w.writerow([a] + [M[a][b] for b in nodes])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True, type=Path)
    ap.add_argument("--verified", type=Path, default=Path("annotations/precedence/verified_pairs.csv"))
    ap.add_argument("--out", default=Path("outputs/matrices"), type=Path)
    ap.add_argument("--parts", type=Path, help="part_groups.csv (instance_id, part_id): merge fragments of ONE physical part, "
                    "e.g. a battery split in two by a sheet lying over it. Nodes then become parts.")
    a = ap.parse_args()

    part_of = {}
    if a.parts and a.parts.exists():
        for r in csv.DictReader(open(a.parts, encoding="utf-8")):
            if (r.get("part_id") or "").strip():
                part_of[r["instance_id"]] = r["part_id"].strip()
    nodes_by_img, dev_by_img, within_part = defaultdict(list), {}, defaultdict(int)
    for r in csv.DictReader(open(a.instances, encoding="utf-8")):
        nid = part_of.get(r["instance_id"], r["instance_id"])
        if nid not in nodes_by_img[r["image_id"]]:
            nodes_by_img[r["image_id"]].append(nid)
        dev_by_img[r["image_id"]] = r["device_id"]
    pairs_by_img = defaultdict(list)
    if a.verified.exists():
        for r in csv.DictReader(open(a.verified, encoding="utf-8")):
            pa = part_of.get(r["instance_a_id"], r["instance_a_id"]); pb = part_of.get(r["instance_b_id"], r["instance_b_id"])
            if pa == pb:                      # two fragments of the same physical part: no ordering between them
                within_part[r["image_id"]] += 1
                continue
            pairs_by_img[r["image_id"]].append({**r, "a": pa, "b": pb})
    else:
        print(f"(no {a.verified}: all pairs stay '?')")

    summary = {}
    for img, nodes in nodes_by_img.items():
        res = build_matrices(nodes, pairs_by_img.get(img, []))
        d = a.out / img; d.mkdir(parents=True, exist_ok=True)
        write_csv_matrix(d / "direct_matrix.csv", nodes, res["direct"])
        write_csv_matrix(d / "transitive_matrix.csv", nodes, res["transitive"])
        with open(d / "edge_ledger.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["device_id", "source_operation", "target_operation", "edge_status", "evidence_reference",
                        "confidence", "origin", "direct_or_transitive", "notes"])
            for (u, v), p in res["edges"].items():
                fwd = p["label"].strip().upper() == "A_BEFORE_B"
                op_u = (p.get("operation_a") if fwd else p.get("operation_b")) or u
                op_v = (p.get("operation_b") if fwd else p.get("operation_a")) or v
                w.writerow([dev_by_img[img], op_u, op_v, "CONFIRMED", p.get("evidence_reference", ""),
                            p.get("confidence", ""), p.get("evidence_type", ""), "direct", p.get("notes", "")])
            for (u, v), path in res["transitive_paths"].items():
                w.writerow([dev_by_img[img], u, v, "CONFIRMED_IMPLIED", "", "", "transitive_closure",
                            "transitive", "path: " + " > ".join(path)])
        n = len(nodes); total = n * (n - 1) // 2
        assessed = sum(1 for a_ in nodes for b_ in nodes if a_ < b_ and res["direct"][a_][b_] != "?")
        lines = [f"image: {img}   device: {dev_by_img[img]}   instances: {n}",
                 f"unordered pairs assessed: {assessed}/{total}  (unknown rate {1 - assessed / total:.1%})" if total else "",
                 f"confirmed direct edges: {len(res['edges'])}   explicit no-dependency pairs: {len(res['nodep'])}",
                 f"contradictions: {res['log']['contradictions']}", f"invalid rows: {res['log']['invalid']}",
                 f"duplicates: {res['log']['duplicates']}",
                 f"pairs inside one merged part (ignored): {within_part[img]}"]
        if res["acyclic"]:
            lines += ["graph: ACYCLIC",
                      f"one valid order (not the only one): {res['topological_order']}",
                      f"parallel levels: {res['parallel_levels']}",
                      f"operations with no confirmed prerequisite: {len(res['no_confirmed_prerequisite'])}"]
        else:
            lines += ["graph: CYCLES FOUND - nothing was removed; review these edges:"] + [f"  {c}" for c in res["cycles"]]
        lines += ["", DISCLAIMER]
        (d / "report.txt").write_text("\n".join(lines), encoding="utf-8")
        json.dump({"nodes": nodes, "direct": res["direct"], "transitive": res["transitive"]},
                  open(d / "matrices.json", "w"), indent=1)
        summary[img] = {"edges": len(res["edges"]), "acyclic": res["acyclic"], "assessed_pairs": assessed, "total_pairs": total}
    json.dump(summary, open(a.out / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2)); print(DISCLAIMER)


if __name__ == "__main__":
    main()
