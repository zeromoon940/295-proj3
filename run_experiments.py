import argparse
import json
import random
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mapf_core import baseline_order, pair_features, plan_prioritized, prepare_instance, read_map, read_scenario


MAPS = ["random-32-32-20", "maze-32-32-4", "room-64-64-16"]
COUNTS = [20, 40, 60, 80]
TRAIN_IDS = list(range(1, 16))
VAL_IDS = list(range(16, 21))
TEST_IDS = list(range(21, 26))
MAP_URL = "https://movingai.com/benchmarks/mapf/mapf-map.zip"
SCEN_URL = "https://movingai.com/benchmarks/mapf/mapf-scen-random.zip"


class RankNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--random-eval", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as src:
        path.write_bytes(src.read())


def ensure_data(data_dir):
    raw = data_dir / "raw"
    maps_dir = data_dir / "maps"
    scen_dir = data_dir / "scen-random"
    maps_dir.mkdir(parents=True, exist_ok=True)
    scen_dir.mkdir(parents=True, exist_ok=True)
    map_zip = raw / "mapf-map.zip"
    scen_zip = raw / "mapf-scen-random.zip"
    download(MAP_URL, map_zip)
    download(SCEN_URL, scen_zip)
    with zipfile.ZipFile(map_zip) as zf:
        for name in MAPS:
            target = maps_dir / f"{name}.map"
            if not target.exists():
                target.write_bytes(zf.read(f"{name}.map"))
    with zipfile.ZipFile(scen_zip) as zf:
        for name in MAPS:
            for sid in TRAIN_IDS + VAL_IDS + TEST_IDS:
                member = f"scen-random/{name}-random-{sid}.scen"
                target = scen_dir / f"{name}-random-{sid}.scen"
                if not target.exists():
                    target.write_bytes(zf.read(member))


def active_config(args):
    if args.mode == "smoke":
        return {
            "maps": ["random-32-32-20"],
            "counts": [20],
            "train_ids": [1, 2],
            "val_ids": [16],
            "test_ids": [21],
            "samples": min(args.samples, 6),
            "random_eval": min(args.random_eval, 3),
            "epochs": min(args.epochs, 3),
        }
    return {
        "maps": MAPS,
        "counts": COUNTS,
        "train_ids": TRAIN_IDS,
        "val_ids": VAL_IDS,
        "test_ids": TEST_IDS,
        "samples": args.samples,
        "random_eval": args.random_eval,
        "epochs": args.epochs,
    }


def load_instances(data_dir, cfg):
    instances = {"train": [], "val": [], "test": []}
    for map_name in cfg["maps"]:
        grid = read_map(data_dir / "maps" / f"{map_name}.map")
        for split, ids in (("train", cfg["train_ids"]), ("val", cfg["val_ids"]), ("test", cfg["test_ids"])):
            for sid in ids:
                scen = read_scenario(data_dir / "scen-random" / f"{map_name}-random-{sid}.scen", max(cfg["counts"]))
                for count in cfg["counts"]:
                    instances[split].append(prepare_instance(map_name, sid, count, grid, scen))
    return instances


def label_job(payload):
    instance, samples, seed = payload
    rng = random.Random(seed)
    candidates = []
    orders = [
        ("natural", baseline_order(instance, "natural", rng)),
        ("shortest", baseline_order(instance, "shortest", rng)),
        ("manhattan", baseline_order(instance, "manhattan", rng)),
    ]
    for sample in range(samples):
        orders.append((f"sample_{sample}", baseline_order(instance, "random", rng)))
    for source, order in orders:
        result = plan_prioritized(instance, order)
        if result["success"]:
            candidates.append((result["cost"], source, order, result["runtime"]))
    row = {
        "map": instance.map_name,
        "scenario_id": instance.scenario_id,
        "agent_count": instance.agent_count,
        "attempts": len(orders),
        "successes": len(candidates),
        "solved": bool(candidates),
    }
    if candidates:
        cost, source, order, runtime = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
        row.update({"best_cost": cost, "best_source": source, "best_runtime": runtime, "best_order": order})
    return row


def make_labels(instances, samples, seed, workers):
    payloads = [(inst, samples, seed + n * 9973) for n, inst in enumerate(instances)]
    rows = []
    if workers <= 1:
        for payload in payloads:
            rows.append(label_job(payload))
            print(f"label {len(rows)}/{len(payloads)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(label_job, payload) for payload in payloads]
            for future in as_completed(futures):
                rows.append(future.result())
                print(f"label {len(rows)}/{len(payloads)}", flush=True)
    rows.sort(key=lambda r: (r["map"], r["scenario_id"], r["agent_count"]))
    return rows


def label_lookup(rows):
    out = {}
    for row in rows:
        out[(row["map"], row["scenario_id"], row["agent_count"])] = row
    return out


def build_pairs(instances, labels):
    xs = []
    ys = []
    meta = []
    for instance in instances:
        key = (instance.map_name, instance.scenario_id, instance.agent_count)
        row = labels.get(key)
        if row is None or not row["solved"]:
            continue
        order = row["best_order"]
        ranks = {agent: rank for rank, agent in enumerate(order)}
        for i in range(instance.agent_count):
            for j in range(i + 1, instance.agent_count):
                xs.append(pair_features(instance, i, j))
                ys.append(1.0 if ranks[i] < ranks[j] else 0.0)
                xs.append(pair_features(instance, j, i))
                ys.append(1.0 if ranks[j] < ranks[i] else 0.0)
                meta.append((instance.map_name, instance.scenario_id, instance.agent_count))
    if not xs:
        raise RuntimeError("no solved labeled training instances")
    return np.stack(xs), np.array(ys, dtype=np.float32), meta


def train_model(train_x, train_y, val_x, val_y, epochs, seed, output_dir):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    train_xn = (train_x - mean) / std
    val_xn = (val_x - mean) / std
    model = RankNet(train_x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    ds = TensorDataset(torch.tensor(train_xn, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=4096, shuffle=True)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(yb)
            total += len(yb)
        train_acc = pair_accuracy(model, train_xn, train_y, device)
        val_acc = pair_accuracy(model, val_xn, val_y, device)
        row = {"epoch": epoch, "loss": total_loss / total, "train_acc": train_acc, "val_acc": val_acc}
        history.append(row)
        print(f"epoch {epoch} loss {row['loss']:.4f} train {train_acc:.3f} val {val_acc:.3f}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "mean": mean, "std": std, "dim": train_x.shape[1]}, output_dir / "model.pt")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    return model, mean, std, device, pd.DataFrame(history)


def pair_accuracy(model, x, y, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, len(x), 65536):
            xb = torch.tensor(x[start:start + 65536], dtype=torch.float32, device=device)
            pred = (torch.sigmoid(model(xb)).cpu().numpy() >= 0.5).astype(np.float32)
            target = y[start:start + 65536]
            correct += int((pred == target).sum())
            total += len(target)
    return correct / max(1, total)


def learned_order(instance, model, mean, std, device):
    model.eval()
    scores = np.zeros(instance.agent_count, dtype=np.float64)
    pairs = []
    indexes = []
    for i in range(instance.agent_count):
        for j in range(i + 1, instance.agent_count):
            pairs.append(pair_features(instance, i, j))
            indexes.append((i, j))
    x = (np.stack(pairs) - mean) / std
    probs = []
    with torch.no_grad():
        for start in range(0, len(x), 65536):
            xb = torch.tensor(x[start:start + 65536], dtype=torch.float32, device=device)
            probs.extend(torch.sigmoid(model(xb)).cpu().numpy().tolist())
    for (i, j), p in zip(indexes, probs):
        scores[i] += p
        scores[j] += 1.0 - p
    return sorted(range(instance.agent_count), key=lambda a: (-scores[a], a))


def eval_job(payload):
    instance, orders = payload
    rows = []
    for method, trial, order in orders:
        result = plan_prioritized(instance, order)
        rows.append({
            "map": instance.map_name,
            "scenario_id": instance.scenario_id,
            "agent_count": instance.agent_count,
            "method": method,
            "trial": trial,
            "success": result["success"],
            "cost": result["cost"],
            "runtime": result["runtime"],
        })
    return rows


def evaluate(instances, model, mean, std, device, random_eval, seed, workers):
    payloads = []
    for n, instance in enumerate(instances):
        rng = random.Random(seed + n * 7919)
        orders = [
            ("learned", 0, learned_order(instance, model, mean, std, device)),
            ("shortest", 0, baseline_order(instance, "shortest", rng)),
            ("manhattan", 0, baseline_order(instance, "manhattan", rng)),
            ("natural", 0, baseline_order(instance, "natural", rng)),
        ]
        for trial in range(random_eval):
            orders.append(("random", trial, baseline_order(instance, "random", rng)))
        payloads.append((instance, orders))
    rows = []
    if workers <= 1:
        for payload in payloads:
            rows.extend(eval_job(payload))
            print(f"eval {len(rows)} rows", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(eval_job, payload) for payload in payloads]
            done = 0
            for future in as_completed(futures):
                rows.extend(future.result())
                done += 1
                print(f"eval {done}/{len(payloads)}", flush=True)
    return pd.DataFrame(rows)


def summarize(eval_df):
    rows = []
    for keys, group in eval_df.groupby(["map", "agent_count", "method"]):
        solved = group[group["success"]]
        rows.append({
            "map": keys[0],
            "agent_count": keys[1],
            "method": keys[2],
            "success_rate": group["success"].mean(),
            "mean_cost_solved": solved["cost"].mean(),
            "mean_runtime": group["runtime"].mean(),
            "runs": len(group),
            "solved_runs": int(group["success"].sum()),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["map", "agent_count", "method"])


def plot_results(summary, history, labels_df, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)
    methods = ["learned", "shortest", "manhattan", "natural", "random"]
    colors = {
        "learned": "#1f77b4",
        "shortest": "#2ca02c",
        "manhattan": "#ff7f0e",
        "natural": "#9467bd",
        "random": "#7f7f7f",
    }
    for metric, ylabel, filename in [
        ("success_rate", "Success rate", "success_rate.png"),
        ("mean_cost_solved", "Mean sum of costs on solved runs", "cost.png"),
        ("mean_runtime", "Mean runtime in seconds", "runtime.png"),
    ]:
        maps = list(summary["map"].unique())
        fig, axes = plt.subplots(1, len(maps), figsize=(5 * len(maps), 3.6), sharey=False)
        if len(maps) == 1:
            axes = [axes]
        for ax, map_name in zip(axes, maps):
            sub = summary[summary["map"] == map_name]
            x = np.arange(len(sorted(sub["agent_count"].unique())))
            counts = sorted(sub["agent_count"].unique())
            width = 0.16
            for offset, method in enumerate(methods):
                values = []
                for count in counts:
                    row = sub[(sub["agent_count"] == count) & (sub["method"] == method)]
                    values.append(float(row[metric].iloc[0]) if len(row) else np.nan)
                ax.bar(x + (offset - 2) * width, values, width=width, label=method, color=colors[method])
            ax.set_title(map_name)
            ax.set_xticks(x)
            ax.set_xticklabels(counts)
            ax.set_xlabel("Agents")
            ax.set_ylabel(ylabel)
            if metric == "success_rate":
                ax.set_ylim(0, 1.05)
        axes[-1].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=200)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(history["epoch"].to_numpy(), history["train_acc"].to_numpy(), label="train")
    ax.plot(history["epoch"].to_numpy(), history["val_acc"].to_numpy(), label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pairwise accuracy")
    ax.set_ylim(0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "pairwise_accuracy.png", dpi=200)
    plt.close(fig)
    coverage = labels_df.groupby(["map", "agent_count"])["solved"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for map_name in coverage["map"].unique():
        sub = coverage[coverage["map"] == map_name]
        ax.plot(sub["agent_count"].to_numpy(), sub["solved"].to_numpy(), marker="o", label=map_name)
    ax.set_xlabel("Agents")
    ax.set_ylabel("Label coverage")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "label_coverage.png", dpi=200)
    plt.close(fig)


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2))


def main():
    args = parse_args()
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    base = Path(__file__).resolve().parent
    data_dir = base / "data"
    output_dir = base / "outputs" / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = active_config(args)
    save_json(output_dir / "config.json", cfg | {"seed": args.seed, "workers": args.workers})
    ensure_data(data_dir)
    instances = load_instances(data_dir, cfg)
    labels_path = output_dir / "labels.csv"
    if labels_path.exists() and not args.force:
        labels_df = pd.read_csv(labels_path)
        labels_df["best_order"] = labels_df["best_order"].apply(lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else x)
        label_rows = labels_df.to_dict("records")
    else:
        label_rows = make_labels(instances["train"] + instances["val"], cfg["samples"], args.seed, args.workers)
        labels_df = pd.DataFrame(label_rows)
        labels_save = labels_df.copy()
        labels_save["best_order"] = labels_save["best_order"].apply(lambda x: json.dumps(x) if isinstance(x, list) else "")
        labels_save.to_csv(labels_path, index=False)
    labels = label_lookup(label_rows)
    train_x, train_y, _ = build_pairs(instances["train"], labels)
    val_x, val_y, _ = build_pairs(instances["val"], labels)
    model, mean, std, device, history = train_model(train_x, train_y, val_x, val_y, cfg["epochs"], args.seed, output_dir)
    eval_df = evaluate(instances["test"], model, mean, std, device, cfg["random_eval"], args.seed + 100000, args.workers)
    eval_df.to_csv(output_dir / "evaluation.csv", index=False)
    summary = summarize(eval_df)
    summary.to_csv(output_dir / "summary.csv", index=False)
    plot_results(summary, history, pd.DataFrame(label_rows), output_dir / "figures")
    labels_all = pd.DataFrame(label_rows)
    train_mask = labels_all["scenario_id"].isin(cfg["train_ids"])
    val_mask = labels_all["scenario_id"].isin(cfg["val_ids"])
    run_info = {
        "seconds": time.perf_counter() - started,
        "device": str(device),
        "train_pairs": int(len(train_y)),
        "val_pairs": int(len(val_y)),
        "train_label_coverage": float(labels_all.loc[train_mask, "solved"].mean()),
        "val_label_coverage": float(labels_all.loc[val_mask, "solved"].mean()),
    }
    save_json(output_dir / "run_info.json", run_info)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(run_info, indent=2), flush=True)


if __name__ == "__main__":
    main()
