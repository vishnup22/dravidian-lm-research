import argparse
from pathlib import Path

BASE     = Path("/nfs/storage1/home/pulipakv/Dravidian")
PROC_DIR = BASE / "data" / "processed"

LANGUAGES = ["te", "ta", "kn", "ml"]

TRAIN_RATIO = 0.96
VAL_RATIO   = 0.02
TEST_RATIO  = 0.02


def split_language(lang: str) -> None:
    in_path = PROC_DIR / f"{lang}_clean.txt"
    if not in_path.exists():
        print(f"  [{lang}] {in_path.name} not found — skipping")
        return

    print(f"  [{lang}] Counting lines...")
    with open(in_path, encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    train_end = int(total_lines * TRAIN_RATIO)
    val_end   = train_end + int(total_lines * VAL_RATIO)

    train_path = PROC_DIR / f"{lang}_train.txt"
    val_path   = PROC_DIR / f"{lang}_val.txt"
    test_path  = PROC_DIR / f"{lang}_test.txt"

    print(f"  [{lang}] Total lines : {total_lines:,}")
    print(f"  [{lang}] Train       : {train_end:,}  ({TRAIN_RATIO*100:.0f}%)")
    print(f"  [{lang}] Val         : {val_end - train_end:,}  ({VAL_RATIO*100:.0f}%)")
    print(f"  [{lang}] Test        : {total_lines - val_end:,}  ({TEST_RATIO*100:.0f}%)")

    with open(in_path, encoding="utf-8") as fin, \
         open(train_path, "w", encoding="utf-8") as ftrain, \
         open(val_path,   "w", encoding="utf-8") as fval, \
         open(test_path,  "w", encoding="utf-8") as ftest:

        for i, line in enumerate(fin):
            if i < train_end:
                ftrain.write(line)
            elif i < val_end:
                fval.write(line)
            else:
                ftest.write(line)

    print(f"  [{lang}] Done → {lang}_train.txt / {lang}_val.txt / {lang}_test.txt\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=LANGUAGES, default=None)
    args = parser.parse_args()
    langs = [args.lang] if args.lang else LANGUAGES

    print("=" * 62)
    print("  TRAIN / VAL / TEST SPLIT  (96 / 2 / 2)")

    for lang in langs:
        split_language(lang)

    print("All splits complete.")


if __name__ == "__main__":
    main()
