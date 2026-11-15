# Dismantling project v2 - precedence matrix from annotations ONLY

Input = your Roboflow COCO annotations (class + mask per part). No videos, no manuals.
Put this folder at `D:\precedence_matrix\code`; everything is written to `D:\precedence_matrix\work`.
Run from `D:\precedence_matrix\code` in the `gpu` env (extra packages: `pip install scikit-learn joblib`).

    python smoke_test.py       # must print "pipeline smoke test: OK"

## Run order
| step | command | result |
|---|---|---|
| 1 | `python run_all.py --root D:\precedence_matrix --stages audit` | `work\audit\audit_report.txt`; fill `work\devices.csv` (same device_id = same phone; optional but better splits) |
| 2 | `... --stages prepare` then `... --stages train` | YOLO dataset + `work\checkpoints\seg\weights\best.pt` (only needed for NEW images) |
| 3 | `... --stages extract,geometry,review --regroup` | instances, geometry matrices, review sheets, auto-merged Battery pieces |
| 4 | `... --stages precedence` | creates `work\class_stage.csv`, the precedence matrices, `eval_sample.csv`, `next_to_label.csv` |
| 5 | edit `work\class_stage.csv`, rerun step 4 | the main assumption of the method (see below) |
| 6 | `... --stages label` (blind sample, keyboard tool), then `... --stages evaluate` | measured precision of the stage rule |
| 7 | label more pairs, `... --stages model,evaluate,precedence` | learned model + group-CV accuracy |
| 8 | `python run_all.py --images D:\new_phones --weights work\checkpoints\seg\weights\best.pt --stages infer` | matrix for brand-new images |

## Matrices from the masks alone (work\geometry\<image>\)
distance_px, adjacency (T/N/.), containment, notch_filled (stacking cue), direction, area_ratio_log10, nodes.csv;
class-level: class_cooccurrence_matrix, class_contact_matrix, class_near_matrix, class_adjacency_counts. These are geometry facts, not precedence.

## The precedence matrix (work\precedence\<image>\)   1 = row before column, 0 = no direct dependency, ? = unknown, ! = contradiction
Where the cells come from (tier, in order of strength):
- V verified: pairs YOU labelled by looking at the picture -> `precedence_confirmed_*.csv`
- C pair rule: optional exceptions in `class_precedence.csv`
- S stage rule: layered disassembly. `class_stage.csv` gives each class a removal stage (0 screws ... 5 battery/board; guessed from the class name,
  edit it). Two parts that are neighbours in the image (gap <= 8 px, screws 15 px) get earlier-stage -> later-stage. Equal/unknown stage -> `?`.
- M model: learned from your verified pairs (step 7), used above 0.85 confidence
`precedence_inferred_*.csv` = all tiers together; `precedence_inferred_source.csv` shows the tier of every cell.
Also: precedence_edges.csv, removal_levels.csv (a valid removal order in parallel levels), report.txt, and globally all_edges.csv, summary.json.
A cycle is broken only by dropping its weakest non-verified edge, and the drop is logged.

## Fast labelling (recommended): `python run_all.py --root D:\precedence_matrix --stages label`
A window shows one pair at a time (A red, B blue). Keys: `a` A first, `b` B first, `n` side by side, `u` unsure, `z` undo, `f` whole image,
`q` save and quit; a CAPITAL letter = low confidence. Every label is saved immediately and rows you already did are skipped, so you can stop and resume.
`--queue sample` (default, the blind 100) -> then `--queue next` (most informative for the model) -> `--queue all`. Close Excel first.

## Labelling in Excel instead (open `work\review\pair_review.csv`)
Each row is a pair of neighbouring parts; `review_image` is the picture (A red, B blue). Set `label`:
- A_BEFORE_B - A lies on top of B / is a screw fastening B / is a cable plugged into B: A must come off first
- B_BEFORE_A - the reverse
- NO_DIRECT_DEPENDENCY - they just sit side by side, either can come off first
- UNKNOWN - you cannot tell from the picture (leave it; do not guess)
Fill `evidence_type` = image_inspection, and `confidence` (high = clearly stacked/fastened, low = judgement).
Label `work\precedence\eval_sample.csv` FIRST and do not look at pair_suggestions.csv while doing it: that blind random sample gives the
unbiased precision number for your report. `next_to_label.csv` lists the most informative pairs for training the model.
Re-running `--stages review` never overwrites labelled work (it writes pair_review.new.csv).

## Class names
Spelling variants are merged automatically (`Flash Light` = `FlashLight`, extra spaces, plural `s`). For merges only you can decide
(e.g. `Connection wire` is really `Connector`) create `work\class_aliases.csv`:

    from,to
    Connection wire,Connector

then rerun `extract,geometry,review --regroup` and `precedence` (delete `work\class_stage.csv` if you have not edited it, so it is rebuilt).
Retrain (`prepare`, `train`) only if you want the segmentation model to use the merged classes; the matrices use your annotations directly.

## The number to report (step 6)
`rule_evaluation.json` -> `blind_sample.frequency_weighted.precision_when_fired`: how often a stage edge matches your label on pairs
nobody chose, re-weighted by how common each class pair is. Report it together with `labelled`, `left_without_a_label` and the sample size.

## Honest limits
- Masks cannot prove physical precedence. S/C/M cells are hypotheses; the only quantitative claims come from `s9_evaluate.py` / `s8_train_model.py` on pairs you labelled.
- The stage table is a domain assumption about typical phone layering. Check it for YOUR classes; s9 shows which class pairs it gets wrong.
- Screws mostly fasten brackets/covers that are not in your class list, so screw edges only say "a screw next to this part".
- Rare classes (e.g. Connection wire, Flash Light) have few test instances; report segmentation numbers with the image counts.
- GPU training and the infer stage were not run in the test environment; everything else is covered by smoke_test.py.
