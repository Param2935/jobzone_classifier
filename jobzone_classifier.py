import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

DATA_DIR = "data"

FILES = {
    "essential_skills.csv":    f"https://www.onetcenter.org/dl_files/database/db_31_0_csv/essential_skills.csv",
    "transferable_skills.csv": f"https://www.onetcenter.org/dl_files/database/db_31_0_csv/transferable_skills.csv",
    "job_zones.csv":           f"https://www.onetcenter.org/dl_files/database/db_31_0_csv/job_zones.csv",
}

HEADERS = {"User-Agent": "INST414-student-project/1.0 (educational use)"}

PROBE_CODE = "15-1252.00" 

TEST_SIZE = 0.25
RANDOM_STATE = 42

FIG_DIR = "figures"
DEPTH_RANGE = range(1, 21)


# Data collection

def download_file(url, dest):
    """Download url to dest unless it is already cached."""
    if os.path.exists(dest):
        return
    import requests
    print(f"downloading {url}")
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)
    time.sleep(1)


def collect_data():
    """Download (if needed) and load the source CSVs into DataFrames."""
    os.makedirs(DATA_DIR, exist_ok=True)
    frames = {}
    for filename, url in FILES.items():
        dest = os.path.join(DATA_DIR, filename)
        download_file(url, dest)
        df = pd.read_csv(dest)
        df.columns = df.columns.str.strip()
        frames[filename.replace(".csv", "")] = df
    return frames


# Creating labeled dataset 
def build_dataset(frames):
    """
    Returns:
        X      : occupations x 35 skills, raw importance ratings (1-5)
        y      : Job Zone per occupation (ints; 2, 3, 4 or 5)
        titles : O*NET-SOC Code : occupation title
    """
    skills = pd.concat(
        [frames["essential_skills"], frames["transferable_skills"]],
        ignore_index=True,
    )
    im = skills[skills["Scale ID"] == "IM"]

    X = im.pivot_table(
        index="O*NET-SOC Code",
        columns="Element Name",
        values="Data Value",
        aggfunc="mean",
    )
    n_pivoted = len(X)
    X = X.dropna(axis=0, how="any")
    print(f"skill matrix: {n_pivoted} occupations pivoted, "
          f"{n_pivoted - len(X)} dropped for missing skills")

    zones = (frames["job_zones"]
             .drop_duplicates("O*NET-SOC Code")
             .set_index("O*NET-SOC Code")["Job Zone"])
    y = zones.reindex(X.index)

    unlabeled = int(y.isna().sum())
    if unlabeled:
        print(f"labels: {unlabeled} occupations had no Job Zone and were dropped")
        y = y.dropna()
        X = X.loc[y.index]
    y = y.astype(int)
    y.name = "job_zone"

    titles = (skills.drop_duplicates("O*NET-SOC Code")
                    .set_index("O*NET-SOC Code")["Title"]
                    .reindex(X.index))

    return X, y, titles


# Check input data

def describe_dataset(X, y, titles):
    """Print the checks that confirm the dataset is built correctly."""
    print(f"\ndataset: {X.shape[0]} occupations x {X.shape[1]} skill features")
    print(f"importance range (expect 1.0-5.0): "
          f"{X.values.min():.2f} to {X.values.max():.2f}")

    print(f"distinct Job Zones (expect [2, 3, 4, 5]): "
          f"{[int(z) for z in sorted(y.unique())]}")

    counts = y.value_counts().sort_index()
    print("\nclass distribution:")
    for zone, n in counts.items():
        print(f"  Job Zone {zone}: {n:4d}  ({n / len(y):6.1%})")
    majority_zone = counts.idxmax()
    baseline = counts.max() / len(y)
    print(f"\nmajority-class baseline: always predict Job Zone {majority_zone} "
          f"-> {baseline:.1%} accuracy")

    if PROBE_CODE in X.index:
        top = X.loc[PROBE_CODE].nlargest(3)
        print(f"\nspot check {PROBE_CODE} ({titles.get(PROBE_CODE)}):")
        print(f"  Job Zone: {y.loc[PROBE_CODE]}")
        print(f"  top skills by importance: "
              f"{', '.join(f'{k} {v:.2f}' for k, v in top.items())}")


# Training and evaluation

def accuracy(y_true, y_pred):
    """Share of predictions that match the true label."""
    return float((np.asarray(y_pred) == np.asarray(y_true)).mean())


def split_data(X, y):
    """
    Split into a training set and a held-out test set.

    Stratified so each Job Zone keeps its share in both halves; without this
    the smallest zone could be under-represented in the test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\ntrain/test split: {len(X_train)} train, {len(X_test)} test")
    print("  test-set zones: "
          + ", ".join(f"zone {z}: {n}"
                      for z, n in y_test.value_counts().sort_index().items()))
    return X_train, X_test, y_train, y_test


def majority_baseline(y_train, y_test):
    """
    Baseline: always predict the most common Job Zone in the training data.
    """
    majority = y_train.value_counts().idxmax()
    preds = np.full(len(y_test), majority)
    acc = accuracy(y_test, preds)
    print(f"\nbaseline (always predict Job Zone {majority}): "
          f"test accuracy {acc:.1%}")
    return acc


def fit_tree(X_train, y_train, X_test, y_test, max_depth=None):
    """
    Fit a decision tree and report train and test accuracy.
    """
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)

    train_acc = accuracy(y_train, tree.predict(X_train))
    test_acc = accuracy(y_test, tree.predict(X_test))

    depth_label = "unpruned" if max_depth is None else f"max_depth={max_depth}"
    print(f"\ndecision tree ({depth_label}):")
    print(f"  actual depth:   {tree.get_depth()}")
    print(f"  leaf nodes:     {tree.get_n_leaves()}")
    print(f"  train accuracy: {train_acc:.1%}")
    print(f"  test accuracy:  {test_acc:.1%}")
    print(f"  train - test gap: {train_acc - test_acc:.1%}")
    return tree, train_acc, test_acc



# Chosing tree complexity
def sweep_depth(X_train, y_train, X_test, y_test, depths=DEPTH_RANGE):
    """Fit a tree at each max_depth; return accuracies and size per depth."""
    rows = []
    for d in depths:
        tree = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
        tree.fit(X_train, y_train)
        rows.append({
            "max_depth": d,
            "leaves": tree.get_n_leaves(),
            "train_accuracy": accuracy(y_train, tree.predict(X_train)),
            "test_accuracy": accuracy(y_test, tree.predict(X_test)),
        })
    metrics = pd.DataFrame(rows).set_index("max_depth")
    print("\ndepth sweep:")
    print(metrics.round({"train_accuracy": 3, "test_accuracy": 3}).to_string())
    return metrics


def plot_depth_sweep(metrics, baseline, path):
    """Save the train-vs-test accuracy curve used to choose max_depth."""
    os.makedirs(FIG_DIR, exist_ok=True)
    best = int(metrics["test_accuracy"].idxmax())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics.index, metrics["train_accuracy"], "o-", label="Training accuracy")
    ax.plot(metrics.index, metrics["test_accuracy"], "o-", label="Test accuracy")
    ax.axhline(baseline, ls=":", color="gray",
               label=f"Majority-class baseline ({baseline:.1%})")
    ax.axvline(best, ls="--", color="darkred", alpha=0.6,
               label=f"Best test accuracy (max_depth={best})")

    ax.set_xlabel("max_depth")
    ax.set_ylabel("accuracy")
    ax.set_title("tree accuracy vs. depth")
    ax.set_xticks(list(metrics.index))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved figure -> {path}")


def pick_depth(metrics):
    """
    Choose max_depth as the value with the highest test accuracy.
    """
    best = int(metrics["test_accuracy"].idxmax())
    print(f"\nchosen max_depth = {best} "
          f"(test accuracy {metrics.loc[best, 'test_accuracy']:.1%}, "
          f"{int(metrics.loc[best, 'leaves'])} leaves)")
    return best



# Fitted tree
def plot_fitted_tree(tree, feature_names, path, max_depth=None,
                     figsize=(30, 12), fontsize=7):
    """Draw the fitted tree. max_depth renders only the top levels."""
    os.makedirs(FIG_DIR, exist_ok=True)
    class_names = [f"Zone {c}" for c in tree.classes_]

    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(
        tree,
        feature_names=list(feature_names),
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,     
        max_depth=max_depth,
        fontsize=fontsize,
        ax=ax,
    )
    title = "Predicting Job Zone from skill importance"
    if max_depth is not None:
        title += f"  (top {max_depth} levels)"
    ax.set_title(title, fontsize=14)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure -> {path}")


def print_top_splits(tree, feature_names, levels=2):
    """Print the root split and the splits immediately below it."""
    t = tree.tree_
    print(f"\ntop splits (the skills the tree keys on):")

    def walk(node, depth, label):
        if depth > levels or t.feature[node] == -2:   # -2 marks a leaf
            return
        skill = feature_names[t.feature[node]]
        print(f"{'    ' * depth}{label}{skill} <= {t.threshold[node]:.2f}")
        walk(t.children_left[node], depth + 1, "yes -> ")
        walk(t.children_right[node], depth + 1, "no  -> ")

    walk(0, 0, "root: ")


# Error analysis

def analyze_errors(tree, X_test, y_test, titles, n_examples=5):
    """
    Look at what the model got wrong on the test set.
    """
    pred = tree.predict(X_test)
    confidence = tree.predict_proba(X_test).max(axis=1)

    results = pd.DataFrame({
        "title": titles.reindex(X_test.index).values,
        "actual_zone": y_test.values,
        "predicted_zone": pred,
        "confidence": confidence.round(3),
    }, index=X_test.index)
    results["correct"] = results["actual_zone"] == results["predicted_zone"]
    results["zones_off"] = (results["actual_zone"]
                            - results["predicted_zone"]).abs()

    errors = results[~results["correct"]]
    n_err = len(errors)
    off_by_one = int((errors["zones_off"] == 1).sum())

    print(f"\nerrors: {n_err} of {len(results)} test occupations misclassified")
    print(f"  off by one zone : {off_by_one} ({off_by_one / n_err:.0%} of errors)")
    print(f"  off by two or more: {n_err - off_by_one}")

    worst = errors.sort_values("confidence").head(n_examples)
    print(f"\n{n_examples} lowest-confidence errors:")
    print(worst[["title", "actual_zone", "predicted_zone",
                 "confidence"]].to_string())

    results.to_csv("test_predictions.csv")
    worst.to_csv("error_examples.csv")
    print("\nsaved test_predictions.csv and error_examples.csv")
    return results, worst


def main():
    frames = collect_data()
    X, y, titles = build_dataset(frames)
    describe_dataset(X, y, titles)

    X.to_csv("features.csv")
    y.to_csv("labels.csv")
    print("\nsaved features.csv and labels.csv")

    X_train, X_test, y_train, y_test = split_data(X, y)
    base_acc = majority_baseline(y_train, y_test)
    _, _, unpruned_acc = fit_tree(X_train, y_train, X_test, y_test)

    metrics = sweep_depth(X_train, y_train, X_test, y_test)
    metrics.to_csv("depth_sweep.csv")
    plot_depth_sweep(metrics, base_acc,
                     os.path.join(FIG_DIR, "depth_sweep.png"))

    best_depth = pick_depth(metrics)
    final_tree, _, final_acc = fit_tree(
        X_train, y_train, X_test, y_test, max_depth=best_depth)

    print("\nsummary (test accuracy):")
    print(f"  majority-class baseline : {base_acc:.1%}")
    print(f"  unpruned tree           : {unpruned_acc:.1%}")
    print(f"  tree at max_depth={best_depth:<2}    : {final_acc:.1%}")
    print(f"  final tree vs baseline  : {final_acc - base_acc:+.1%}")

    plot_fitted_tree(final_tree, X.columns,
                     os.path.join(FIG_DIR, "tree_full.png"),
                     figsize=(38, 14), fontsize=7)
    plot_fitted_tree(final_tree, X.columns,
                     os.path.join(FIG_DIR, "tree_top3.png"),
                     max_depth=3, figsize=(28, 10), fontsize=9)
    print_top_splits(final_tree, list(X.columns))

    analyze_errors(final_tree, X_test, y_test, titles)


if __name__ == "__main__":
    main()