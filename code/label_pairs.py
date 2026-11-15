#!/usr/bin/env python3
"""
Keyboard labelling tool for pair_review.csv - one picture at a time, one key per decision.

    python label_pairs.py --review work\\review\\pair_review.csv --queue sample
    python label_pairs.py --selftest                      # checks that a window can open on this machine

queue:  sample = the 100-pair BLIND random sample (do this first; the engine's guess is NOT shown)
        next   = the most informative pairs for training the model (next_to_label.csv)
        all    = every neighbouring pair (priority 1 = no screw first)
The window has buttons for every action (and a 'low confidence' tick box), or use the keys below after clicking the window:
    a  A must come off before B  (A is on top of B / a screw holding B / a cable plugged into B)
    b  B must come off before A
    n  no direct dependency (just side by side, either order works)
    u  unsure - cannot tell from the picture (kept UNKNOWN, remembered as reviewed)
    z  or Backspace: undo the last decision        f: show the whole numbered image        q or Esc: save and quit
    CAPITAL A / B / N = same label but LOW confidence (default is high; make sure Caps Lock is OFF).
Every label is saved at once. Close Excel first (it locks the file). Labelled rows are skipped unless --relabel.
Uses Tk (ships with Python) and Pillow; --backend cv uses an OpenCV window instead (q to quit).
`--keys abnuz` applies a key sequence without any window (used by the tests).
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


def render(pair_img, header, max_w=1200, max_h=700, min_w=900):
    import cv2, numpy as np
    h, w = pair_img.shape[:2]
    s = min(max_w / w, max_h / h, 3.0)
    img = cv2.resize(pair_img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    if img.shape[1] < min_w:
        pad = np.full((img.shape[0], min_w - img.shape[1], 3), 35, np.uint8); img = np.hstack([img, pad])
    for text, bar_h, scale in ((header, 30, 0.6), (HELP, 30, 0.55)):
        bar = np.full((bar_h, img.shape[1], 3), 35, np.uint8)
        cv2.putText(bar, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        img = np.vstack([bar, img]) if text is header else np.vstack([img, bar])
    return img


def frame_for(ses, rows, queue, review_dir, full, max_w=1200, max_h=700):
    """Next displayable frame (BGR) for the current pair, skipping missing pictures; None when the queue is finished."""
    import cv2
    while ses.current() is not None:
        row = ses.current()
        path = review_dir / ("overlays_ids/" + row["image_id"] + ".jpg" if full else row["review_image"])
        img = cv2.imread(str(path))
        if img is None:
            print("missing image", path); ses.pos += 1; continue
        done = sum(1 for r in rows if not is_open(r))
        return render(img, f"[{ses.pos + 1}/{len(queue)}] pair {row['pair_no']}   {row['image_id']}   (done so far: {done})", max_w=max_w, max_h=max_h)
    return None


def run_tk(ses, rows, fields, path, queue, review_dir, scripted=None, clicks=None, debug=False):
    import tkinter as tk
    from PIL import Image, ImageTk
    root = tk.Tk(); root.title("label pairs - buttons or keys: a / b / n / u / z / f / q")
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    max_w, max_h = min(1200, int(sw * 0.9)), min(700, int(sh * 0.6))          # leave room for title bar, buttons, taskbar
    panel = tk.Label(root, bd=0); panel.pack()
    bar = tk.Frame(root); bar.pack(fill="x")
    msg = tk.StringVar(value="Click a button or press a key. A = red part, B = blue part.")
    status = tk.Frame(root); status.pack(fill="x", padx=6, pady=2)
    tk.Label(status, textvariable=msg, anchor="w", font=("Segoe UI", 10)).pack(side="left")
    low = tk.BooleanVar(value=False)
    tk.Checkbutton(status, text="low confidence", variable=low).pack(side="right")
    st = {"full": False, "photo": None, "n": 0, "closing": False}
    btns = {}

    def finish(reason):
        if st["closing"]:
            return
        st["closing"] = True
        save(path, fields, rows)
        got = sum(1 for r in rows if (r.get("label") or "UNKNOWN").upper() not in ("", "UNKNOWN"))
        print(f"closing ({reason}). labelled this session: {st['n']}; labelled pairs in the sheet: {got}.")
        root.destroy()

    def show():
        fr = frame_for(ses, rows, queue, review_dir, st["full"], max_w, max_h)
        if fr is None:
            finish("queue finished"); return
        st["photo"] = ImageTk.PhotoImage(Image.fromarray(fr[:, :, ::-1]))
        panel.configure(image=st["photo"])

    def act(ch):
        if debug:
            print("event:", repr(ch))
        if ch == "q":
            finish("q / Save&quit pressed"); return
        if ch == "f":
            st["full"] = not st["full"]; show(); return
        if ch in ("a", "b", "n") and low.get():
            ch = ch.upper()
        row = ses.current(); pair_no = row["pair_no"] if row else "-"
        res = ses.handle(ch)
        if not res:
            return
        st["full"] = False; save(path, fields, rows)
        if res == "labelled":
            st["n"] += 1; text = f"pair {pair_no}: {row['label']} ({row['confidence']} confidence)"
        elif res == "unsure":
            text = f"pair {pair_no}: marked unsure"
        else:
            text = "undid the last decision"
        msg.set(text); print(text); show()

    def on_key(e):
        if debug:
            print("key:", e.keysym, repr(e.char))
        if e.keysym == "Escape":
            finish("Esc pressed"); return
        act("z" if e.keysym == "BackSpace" else (e.char or ""))

    for ch, text in (("a", "A first  (a)"), ("b", "B first  (b)"), ("n", "Side by side  (n)"), ("u", "Unsure  (u)"),
                     ("z", "Undo  (z)"), ("f", "Whole image  (f)"), ("q", "Save && quit  (q)".replace("&&", "&"))):
        btns[ch] = tk.Button(bar, text=text, width=14, command=lambda c=ch: act(c)); btns[ch].pack(side="left", padx=2, pady=4)
    root.bind("<Key>", on_key); root.protocol("WM_DELETE_WINDOW", lambda: finish("window closed with the X"))
    show(); root.update()
    root.lift(); root.attributes("-topmost", True); root.after(400, lambda: root.attributes("-topmost", False)); root.focus_force()
    print("window is open (look for 'label pairs' in the taskbar if you cannot see it).")
    if scripted is not None:                                   # test hook: feed keys as real Tk key events
        def feed(i=0):
            if i >= len(scripted):
                root.after(150, lambda: finish("scripted keys done")); return
            root.focus_force(); root.event_generate("<Key>", keysym=scripted[i])
            root.after(120, lambda: feed(i + 1))
        root.after(300, feed)
    if clicks is not None:                                     # test hook: press the real buttons
        def click(i=0):
            if i >= len(clicks):
                root.after(150, lambda: finish("scripted clicks done")); return
            btns[clicks[i]].invoke(); root.after(120, lambda: click(i + 1))
        root.after(300, click)
    root.mainloop()


def run_cv(ses, rows, fields, path, queue, review_dir):
    import cv2
    win, full = "label pairs (a/b/n/u/z/f/q)", False
    while ses.current() is not None:
        fr = frame_for(ses, rows, queue, review_dir, full)
        if fr is None:
            break
        cv2.imshow(win, fr)
        k = cv2.waitKey(0)                                     # blocking; no close-detection (unreliable on some builds): press q
        if k in (27, ord("q")):
            break
        if k == ord("f"):
            full = not full; continue
        if ses.handle(chr(k) if 0 <= k < 256 else ""):
            full = False; save(path, fields, rows)
    cv2.destroyAllWindows()


def selftest():
    import tkinter as tk
    from PIL import Image, ImageTk
    import numpy as np
    root = tk.Tk(); root.title("selftest")
    fr = render(np.full((300, 600, 3), 90, np.uint8), "selftest - press any key, or wait 6 seconds")
    photo = ImageTk.PhotoImage(Image.fromarray(fr[:, :, ::-1])); tk.Label(root, image=photo).pack()
    def key(e):
        print(f"key received: char={e.char!r} keysym={e.keysym}"); root.destroy()
    root.bind("<Key>", key); root.after(6000, root.destroy); root.focus_force()
    print("A grey window should be open. Press a key in it (or wait 6 s).")
    root.mainloop(); print("Tk window test finished: if you saw the window, labelling will work.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", type=Path, help="pair_review.csv")
    ap.add_argument("--queue", choices=["sample", "next", "all"], default="sample")
    ap.add_argument("--prec", type=Path, help="precedence folder (default: ..\\precedence next to the review folder)")
    ap.add_argument("--relabel", action="store_true")
    ap.add_argument("--backend", choices=["tk", "cv"], default="tk")
    ap.add_argument("--keys", help="apply this key sequence without a window")
    ap.add_argument("--gui_keys", help="(tests) open the Tk window and feed these keys as events")
    ap.add_argument("--gui_clicks", help="(tests) open the Tk window and press these buttons")
    ap.add_argument("--debug_keys", action="store_true", help="print every key / button event to the console")
    ap.add_argument("--selftest", action="store_true", help="open a test window and exit")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not a.review:
        ap.error("--review is required")
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
    print(f"{len(queue)} pairs in queue '{a.queue}'. A window opens: use the buttons, or click it and press a / b / n / u / z / f / q.")
    if a.backend == "tk":
        try:
            run_tk(ses, rows, fields, a.review, queue, a.review.parent, a.gui_keys, a.gui_clicks, a.debug_keys)
        except ImportError as e:
            sys.exit(f"{e}\nTk/Pillow missing: pip install pillow  (Tk ships with python.org/conda Python), or use --backend cv")
    else:
        run_cv(ses, rows, fields, a.review, queue, a.review.parent)
    save(a.review, fields, rows)
    lab = sum(1 for r in rows if (r.get("label") or "UNKNOWN").upper() not in ("", "UNKNOWN"))
    print(f"saved. labelled pairs in the sheet: {lab}. " + ("Next: python run_all.py --root ... --stages evaluate" if lab else "Nothing labelled yet - run it again and use the buttons."))


if __name__ == "__main__":
    main()