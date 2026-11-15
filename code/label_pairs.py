#!/usr/bin/env python3
"""
Keyboard labelling tool for pair_review.csv - one picture at a time, one key per decision.

    python label_pairs.py --review work\\review\\pair_review.csv --queue sample

queue:  sample = the 100-pair BLIND random sample (do this first; the engine's guess is NOT shown)
        next   = the most informative pairs for training the model (next_to_label.csv)
        all    = every neighbouring pair (priority 1 = no screw first)
Window keys (the picture shows A in red, B in blue):
    a  A must come off before B  (A is on top of B / a screw holding B / a cable plugged into B)
    b  B must come off before A
    n  no direct dependency (just side by side, either order works)
    u  unsure - cannot tell from the picture (kept UNKNOWN, remembered as reviewed)
    z  or Backspace: undo the last decision          f: show the whole numbered image          q or Esc: save and quit
    CAPITAL A / B / N = same label but LOW confidence (default is high). Every label is saved at once.
Close Excel first (it locks the file). Rows that already have a label are skipped unless --relabel.
`--keys abnuz` applies a key sequence without a window (used by the tests).
"""
import argparse, csv, os, sys
from pathlib import Path

LABEL_KEYS = {"a": "A_BEFORE_B", "b": "B_BEFORE_A", "n": "NO_DIRECT_DEPENDENCY"}
HELP = "a: A first | b: B first | n: side by side | u: unsure | z: undo | f: full image | q: save+quit | CAPITAL = low confidence"


def is_open(row):
    return (row.get("label") or "UNKNOWN").strip().upper() in ("", "UNKNOWN") and not (row.get("notes") or "").startswith("reviewed-unsure")


def build_queue(rows, mode, prec_dir, relabel=False):
    by_no = {r["pair_no"]: i for i, r in enumerate(rows)}
    if mode == "sample":
        f = Path(prec_dir) / "eval_sample.csv"
        nos = [r["pair_no"] for r in csv.DictReader(open(f, encoding="utf-8"))] if f.exists() else []
    elif mode == "next":
        f = Path(prec_dir) / "next_to_label.csv"
        nos = [r["pair_no"] for r in csv.DictReader(open(f, encoding="utf-8")) if r["pair_no"]] if f.exists() else []
    else:
        nos = [r["pair_no"] for r in sorted(rows, key=lambda r: (int(r.get("priority") or 1), int(r["pair_no"])))]
    idx = [by_no[n] for n in nos if n in by_no]
    return idx if relabel else [i for i in idx if is_open(rows[i])]


class Session:
    """The decision logic, separate from the window so it can be tested."""
    def __init__(self, rows, queue):
        self.rows, self.queue, self.pos, self.history = rows, queue, 0, []

    def current(self):
        return self.rows[self.queue[self.pos]] if self.pos < len(self.queue) else None

    def handle(self, ch):
        """Apply one key. Returns 'labelled' | 'unsure' | 'undo' | None."""
        if ch in ("z", "\x08"):
            if not self.history:
                return None
            self.pos, snap = self.history.pop()
            self.rows[self.queue[self.pos]].clear(); self.rows[self.queue[self.pos]].update(snap)
            return "undo"
        row = self.current()
        if row is None:
            return None
        k = ch.lower()
        if k in LABEL_KEYS:
            self.history.append((self.pos, dict(row)))
            row.update(label=LABEL_KEYS[k], confidence="high" if ch.islower() else "low", evidence_type="image_inspection",
                       annotator=row.get("annotator") or "manual")
            if (row.get("notes") or "").startswith("reviewed-unsure"):
                row["notes"] = row["notes"].split("; ", 1)[-1]
            self.pos += 1
            return "labelled"
        if ch == "u":
            self.history.append((self.pos, dict(row)))
            row["notes"] = "reviewed-unsure; " + (row.get("notes") or "")
            self.pos += 1
            return "unsure"
        return None


def save(path, fields, rows):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def wait_for_key(cv, win, state, poll_ms=100):
    """Block until a key is pressed -> key code, or None if the user closed the window.
    Some OpenCV builds report WND_PROP_VISIBLE as 0/-1 before the window is drawn (or always), so 'closed' is only
    believed AFTER the window has been seen visible at least once."""
    while True:
        k = cv.waitKey(poll_ms)
        if k != -1:
            return k
        try:
            vis = cv.getWindowProperty(win, cv.WND_PROP_VISIBLE)
        except Exception:
            vis = -1
        if vis >= 1:
            state["seen"] = True
        elif state.get("seen"):
            return None


def render(pair_img, header, max_w=1300, max_h=780, min_w=1000):
    import cv2, numpy as np
    h, w = pair_img.shape[:2]
    s = min(max_w / w, max_h / h, 3.0)
    img = cv2.resize(pair_img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    if img.shape[1] < min_w:
        pad = np.full((img.shape[0], min_w - img.shape[1], 3), 35, np.uint8); img = np.hstack([img, pad])
    for text, bar_h, scale in ((header, 30, 0.6), (HELP, 30, 0.55)):
        bar = np.full((bar_h, img.shape[1], 3), 35, np.uint8)
        cv2.putText(bar, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        img = np.vstack([bar, img]) if text is header else np.vstack([img, bar])
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True, type=Path, help="pair_review.csv")
    ap.add_argument("--queue", choices=["sample", "next", "all"], default="sample")
    ap.add_argument("--prec", type=Path, help="precedence folder (default: ..\\precedence next to the review folder)")
    ap.add_argument("--relabel", action="store_true")
    ap.add_argument("--keys", help="apply this key sequence without a window")
    ap.add_argument("--selftest", action="store_true", help="open a test window, print what OpenCV reports, and exit")
    a = ap.parse_args()
    if a.selftest:
        import cv2, numpy as np
        cv2.namedWindow("selftest", cv2.WINDOW_NORMAL)
        cv2.imshow("selftest", render(np.full((300, 600, 3), 90, np.uint8), "selftest - press any key (or wait 10 s)"))
        for i in range(100):
            k = cv2.waitKey(100); vis = cv2.getWindowProperty("selftest", cv2.WND_PROP_VISIBLE)
            if i % 10 == 0 or k != -1:
                print(f"t={i / 10:.1f}s  waitKey={k}  WND_PROP_VISIBLE={vis}")
            if k != -1:
                break
        cv2.destroyAllWindows(); print("OpenCV", cv2.__version__); return
    prec = a.prec or a.review.parent.parent / "precedence"
    with open(a.review, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f); fields = rd.fieldnames; rows = list(rd)
    queue = build_queue(rows, a.queue, prec, a.relabel)
    if not queue:
        print(f"Nothing to label in queue '{a.queue}' (all done, or {prec / 'eval_sample.csv'} missing - run the precedence stage first)."); return
    ses = Session(rows, queue)
    if a.keys is not None:
        for ch in a.keys:
            ses.handle(ch)
        save(a.review, fields, rows); print(f"applied {len(a.keys)} keys, position {ses.pos}/{len(queue)}"); return

    import cv2
    win, review_dir = "label pairs  (a / b / n / u / z / f / q)", a.review.parent
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    except cv2.error as e:
        sys.exit(f"OpenCV cannot open windows here ({e}).\nFix: pip uninstall -y opencv-python-headless && pip install opencv-python  -  or label in Excel instead.")
    print(f"OpenCV {cv2.__version__}: {len(queue)} pairs in queue '{a.queue}'. A window should appear - click it, then press a / b / n / u / z / f / q.")
    state, full = {"seen": False}, False
    while ses.current() is not None:
        row = ses.current()
        path = review_dir / ("overlays_ids/" + row["image_id"] + ".jpg" if full else row["review_image"])
        img = cv2.imread(str(path))
        if img is None:
            print("missing image", path); ses.pos += 1; continue
        done = sum(1 for r in rows if not is_open(r))
        frame = render(img, f"[{ses.pos + 1}/{len(queue)}] pair {row['pair_no']}   {row['image_id']}   (labelled so far: {done})")
        cv2.imshow(win, frame)
        k = wait_for_key(cv2, win, state)
        if k is None:
            save(a.review, fields, rows); print("window closed - saved"); return
        if k in (27, ord("q")):
            break
        if k == ord("f"):
            full = not full; continue
        res = ses.handle(chr(k) if 0 <= k < 256 else "")
        if res:
            full = False; save(a.review, fields, rows)
    save(a.review, fields, rows)
    cv2.destroyAllWindows()
    lab = sum(1 for r in rows if (r.get("label") or "UNKNOWN").upper() not in ("", "UNKNOWN"))
    print(f"saved. labelled pairs in the sheet: {lab}. Next: python run_all.py --root ... --stages evaluate")


if __name__ == "__main__":
    main()