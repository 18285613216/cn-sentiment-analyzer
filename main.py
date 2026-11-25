#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""中文短评情感分析器（纯 Python 标准库实现）。

- 特征：字符级 1-gram + 2-gram（中文不依赖分词）
- 模型：多项式朴素贝叶斯（Multinomial NB），log 空间计算，拉普拉斯平滑
- 评估：分层 5 折交叉验证（固定随机种子 42），每折在训练集内部划 20% dev
        从 alpha ∈ {0.1, 0.3, 1.0} 中选择平滑系数
- CLI：--eval / --predict TEXT / --train

用法:
    python main.py --eval
    python main.py --predict "东西不错但物流太慢"
    python main.py --train
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import csv
import json
import math
import os
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "reviews.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.json")

SEED = 42
N_FOLDS = 5
ALPHAS = (0.1, 0.3, 1.0)
DEV_RATIO = 0.2
LABEL_NAMES = {1: "正面", 0: "负面"}
# 置信度温度缩放的候选值（只用 dev 数据拟合，不参与特征/参数选择）
TEMPERATURE_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0)


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load_data(path=DATA_PATH):
    """读取 data/reviews.csv，返回 [{'label','domain','text'}, ...]。"""
    if not os.path.exists(path):
        raise SystemExit("数据文件不存在: %s" % path)
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for col in ("label", "domain", "text"):
            if col not in fieldnames:
                raise SystemExit("数据缺少列: %s（实际列: %s）" % (col, fieldnames))
        for line_no, row in enumerate(reader, start=2):
            label = (row.get("label") or "").strip()
            text = (row.get("text") or "").strip()
            domain = (row.get("domain") or "").strip()
            if label not in ("0", "1"):
                raise SystemExit("第 %d 行 label 非法: %r" % (line_no, label))
            if not text:
                raise SystemExit("第 %d 行 text 为空" % line_no)
            rows.append({"label": int(label), "domain": domain, "text": text})
    if not rows:
        raise SystemExit("数据为空: %s" % path)
    return rows


def extract_features(text):
    """字符 1-gram 与 2-gram 计数，返回 {feature: count}。

    'u:' 前缀表示单字，'b:' 前缀表示相邻双字，避免两类特征重名。
    """
    chars = [ch for ch in text.strip().lower() if not ch.isspace()]
    feats = {}
    for ch in chars:
        key = "u:" + ch
        feats[key] = feats.get(key, 0) + 1
    for first, second in zip(chars, chars[1:]):
        key = "b:" + first + second
        feats[key] = feats.get(key, 0) + 1
    return feats


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class MultinomialNB:
    """多项式朴素贝叶斯（二分类，类别 0/1），全部在 log 空间计算。"""

    def __init__(self, alpha=1.0, temperature=1.0):
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.vocab_size = 0
        self.log_prior = {0: 0.0, 1: 0.0}
        self.log_prob = {0: {}, 1: {}}
        self.n_docs = {0: 0, 1: 0}

    def fit(self, features, labels):
        """features: [dict], labels: [0/1]。"""
        if len(features) != len(labels):
            raise ValueError("特征与标签数量不一致")
        vocab = set()
        for feats in features:
            vocab.update(feats.keys())
        self.vocab_size = len(vocab)

        counts = {0: {}, 1: {}}
        totals = {0: 0, 1: 0}
        for feats, y in zip(features, labels):
            self.n_docs[y] += 1
            for key, cnt in feats.items():
                counts[y][key] = counts[y].get(key, 0) + cnt
                totals[y] += cnt

        n_samples = len(labels)
        # 先验用 log 概率；V 用全局词表大小
        denom = {c: totals[c] + self.alpha * self.vocab_size for c in (0, 1)}
        for c in (0, 1):
            self.log_prior[c] = math.log(self.n_docs[c] / float(n_samples))
            # 未在训练集中出现的特征在预测时忽略，故这里只需存词表内特征
            self.log_prob[c] = {
                key: math.log((counts[c].get(key, 0) + self.alpha) / denom[c])
                for key in vocab
            }
        return self

    def scores(self, feats):
        """返回 (log P(负面|x), log P(正面|x))。"""
        s0, s1 = self.log_prior[0], self.log_prior[1]
        lp0, lp1 = self.log_prob[0], self.log_prob[1]
        for key, cnt in feats.items():
            v0 = lp0.get(key)
            if v0 is None:  # 训练词表外的特征直接忽略
                continue
            s0 += cnt * v0
            s1 += cnt * lp1[key]
        return s0, s1

    def predict_one(self, feats):
        """返回 (label, p_positive)。p_positive 经温度缩放校准。"""
        s0, s1 = self.scores(feats)
        d = (s1 - s0) / self.temperature
        if d >= 0:  # 数值稳定写法，避免 exp 溢出
            p1 = 1.0 / (1.0 + math.exp(-d))
        else:
            e = math.exp(d)
            p1 = e / (1.0 + e)
        return (1 if p1 >= 0.5 else 0), p1

    # ---------------- 持久化 ---------------- #
    def to_dict(self):
        return {
            "format_version": 2,
            "alpha": self.alpha,
            "temperature": round(self.temperature, 4),
            "vocab_size": self.vocab_size,
            "n_docs": {"0": self.n_docs[0], "1": self.n_docs[1]},
            "log_prior": {"0": round(self.log_prior[0], 8), "1": round(self.log_prior[1], 8)},
            "log_prob": {
                "0": {k: round(v, 8) for k, v in self.log_prob[0].items()},
                "1": {k: round(v, 8) for k, v in self.log_prob[1].items()},
            },
        }

    @classmethod
    def from_dict(cls, obj):
        model = cls(alpha=obj["alpha"], temperature=obj.get("temperature", 1.0))
        model.vocab_size = int(obj["vocab_size"])
        model.n_docs = {0: int(obj["n_docs"]["0"]), 1: int(obj["n_docs"]["1"])}
        model.log_prior = {0: float(obj["log_prior"]["0"]), 1: float(obj["log_prior"]["1"])}
        model.log_prob = {
            0: {k: float(v) for k, v in obj["log_prob"]["0"].items()},
            1: {k: float(v) for k, v in obj["log_prob"]["1"].items()},
        }
        return model


# --------------------------------------------------------------------------- #
# 切分与训练辅助
# --------------------------------------------------------------------------- #
def stratified_kfold_indices(labels, k=N_FOLDS, seed=SEED):
    """自己实现的分层 k 折：类内打乱后轮转分配，保证每折正负比例一致。"""
    rng = random.Random(seed)
    by_class = {0: [], 1: []}
    for idx, y in enumerate(labels):
        by_class[y].append(idx)
    folds = [[] for _ in range(k)]
    for cls in (0, 1):
        indices = by_class[cls][:]
        rng.shuffle(indices)
        for pos, idx in enumerate(indices):
            folds[pos % k].append(idx)
    for fold in folds:
        fold.sort()
    return folds


def stratified_split(indices, labels, ratio=DEV_RATIO, seed=SEED):
    """把 indices 按类别分层划成 (训练子集, dev 子集)。"""
    rng = random.Random(seed)
    by_class = {0: [], 1: []}
    for idx in indices:
        by_class[labels[idx]].append(idx)
    train_part, dev_part = [], []
    for cls in (0, 1):
        items = by_class[cls][:]
        rng.shuffle(items)
        n_dev = int(round(len(items) * ratio))
        if len(items) > 1:
            n_dev = max(1, min(n_dev, len(items) - 1))
        dev_part.extend(items[:n_dev])
        train_part.extend(items[n_dev:])
    return sorted(train_part), sorted(dev_part)


def fit_model(features, labels, alpha):
    model = MultinomialNB(alpha=alpha)
    model.fit(features, labels)
    return model


def accuracy_of(model, features, labels, indices):
    if not indices:
        return 0.0
    correct = 0
    for idx in indices:
        pred, _ = model.predict_one(features[idx])
        if pred == labels[idx]:
            correct += 1
    return correct / float(len(indices))


def select_alpha(features, labels, train_idx, seed=SEED):
    """在训练集内部划 20% dev，按 dev 准确率选 alpha（平局取更小的 alpha）。

    返回 (best_alpha, dev_accs, sub_train_idx, dev_idx)。
    """
    sub_train, dev_idx = stratified_split(train_idx, labels, DEV_RATIO, seed)
    dev_accs = {}
    for alpha in ALPHAS:
        model = fit_model([features[i] for i in sub_train], [labels[i] for i in sub_train], alpha)
        dev_accs[alpha] = accuracy_of(model, features, labels, dev_idx)
    best = max(ALPHAS, key=lambda a: (round(dev_accs[a], 10), -a))
    return best, dev_accs, sub_train, dev_idx


def softplus(x):
    """log(1 + exp(x))，数值稳定。"""
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def fit_temperature(model, features, labels, indices):
    """在 dev 集上网格搜索温度 T，最小化 log loss，用于校准置信度。

    p_positive = sigmoid((s1 - s0) / T)。T 越大，置信度越保守。
    """
    diffs = []
    for idx in indices:
        s0, s1 = model.scores(features[idx])
        diffs.append((s1 - s0, labels[idx]))
    best_t, best_loss = 1.0, float("inf")
    for t in TEMPERATURE_GRID:
        loss = 0.0
        for d, y in diffs:
            x = d / t
            loss += softplus(-x) if y == 1 else softplus(x)
        loss /= float(len(diffs))
        if loss < best_loss - 1e-12:
            best_t, best_loss = t, loss
    return best_t, best_loss


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def classification_metrics(y_true, y_pred):
    """返回 (per-class metrics dict, 混淆矩阵 2x2, accuracy)。

    混淆矩阵行是真实类别、列是预测类别，顺序为 [负面(0), 正面(1)]。
    """
    per_class = {}
    for c in (0, 1):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for t in y_true if t == c),
        }
    cm = [[0, 0], [0, 0]]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / float(len(y_true))
    return per_class, cm, acc


def feature_name(key):
    kind = "单字" if key.startswith("u:") else "双字"
    return key[2:], kind


def top_features(model, feats, top_n=5):
    """按对分类判定的贡献绝对值排序，返回 [(feature, contribution), ...]。

    contribution = count * (log P(f|正面) - log P(f|负面))，正数支持正面。
    """
    items = []
    for key, cnt in feats.items():
        lp0 = model.log_prob[0].get(key)
        lp1 = model.log_prob[1].get(key)
        if lp0 is None or lp1 is None:
            continue
        contribution = cnt * (lp1 - lp0)
        items.append((key, contribution))
    items.sort(key=lambda item: (-abs(item[1]), item[0]))
    return items[:top_n]


def print_top_features(model, feats, top_n=5):
    picks = top_features(model, feats, top_n)
    if not picks:
        print("  （没有命中训练词表中的特征）")
        return
    for rank, (key, contribution) in enumerate(picks, start=1):
        shown, kind = feature_name(key)
        print("  %d. %-6s  %s  贡献 %+.4f" % (rank, shown, kind, contribution))


def save_model(model, path=MODEL_PATH):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(model.to_dict(), f, ensure_ascii=False, indent=1)
        f.write("\n")


def load_model(path=MODEL_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return MultinomialNB.from_dict(json.load(f))


# --------------------------------------------------------------------------- #
# 命令：--eval
# --------------------------------------------------------------------------- #
def cmd_eval():
    rows = load_data()
    features = [extract_features(r["text"]) for r in rows]
    labels = [r["label"] for r in rows]
    n = len(rows)

    print("=" * 62)
    print("中文短评情感分析器 · 分层 5 折交叉验证")
    print("=" * 62)
    print("数据文件: %s" % os.path.relpath(DATA_PATH, BASE_DIR))
    print("样本数: %d（正面 %d / 负面 %d）" % (n, labels.count(1), labels.count(0)))
    domains = {}
    for r in rows:
        domains[r["domain"]] = domains.get(r["domain"], 0) + 1
    print("领域分布: " + "  ".join("%s %d" % (d, c) for d, c in sorted(domains.items())))
    vocab = set()
    for feats in features:
        vocab.update(feats.keys())
    print("特征: 字符 1-gram + 2-gram，全量词表 %d 维" % len(vocab))
    print("平滑系数候选: %s（每折在训练集内 20%% dev 上选择）" % ", ".join(str(a) for a in ALPHAS))
    print("随机种子: %d" % SEED)

    folds = stratified_kfold_indices(labels, N_FOLDS, SEED)
    oof_pred = [None] * n
    per_alpha_test_correct = {alpha: 0 for alpha in ALPHAS}

    print("\n" + "-" * 62)
    print("各折结果")
    print("-" * 62)

    for fold_no, test_idx in enumerate(folds, start=1):
        test_set = set(test_idx)
        train_idx = [i for i in range(n) if i not in test_set]
        best_alpha, dev_accs, sub_train, dev_idx = select_alpha(features, labels, train_idx, seed=SEED + fold_no)

        # 用选中的 alpha 在完整训练集上训练，预测该折
        model = fit_model([features[i] for i in train_idx], [labels[i] for i in train_idx], best_alpha)
        correct = 0
        for idx in test_idx:
            pred, _ = model.predict_one(features[idx])
            oof_pred[idx] = pred
            if pred == labels[idx]:
                correct += 1
        test_acc = correct / float(len(test_idx))

        # 仅作透明参考：每个 alpha 在该折测试集上的准确率（不参与选择）
        for alpha in ALPHAS:
            m = fit_model([features[i] for i in train_idx], [labels[i] for i in train_idx], alpha)
            per_alpha_test_correct[alpha] += sum(
                1 for idx in test_idx if m.predict_one(features[idx])[0] == labels[idx]
            )

        dev_text = "  ".join("α=%s dev=%.4f" % (a, dev_accs[a]) for a in ALPHAS)
        print("折 %d: 训练 %d / dev %d / 测试 %d | 选择 α=%s | 测试 acc=%.4f (%d/%d)"
              % (fold_no, len(sub_train), len(dev_idx), len(test_idx), best_alpha,
                 test_acc, correct, len(test_idx)))
        print("      %s" % dev_text)

    print("\n" + "-" * 62)
    print("alpha 对比（仅供参考；实际选择依据是每折内部 dev）")
    print("-" * 62)
    for alpha in ALPHAS:
        acc = per_alpha_test_correct[alpha] / float(n)
        print("  α=%-3s  5 折总体准确率 %.4f (%d/%d)" % (alpha, acc, per_alpha_test_correct[alpha], n))

    per_class, cm, acc = classification_metrics(labels, oof_pred)

    print("\n" + "-" * 62)
    print("总体指标（240 条 out-of-fold 预测，每条恰好被预测一次）")
    print("-" * 62)
    print("accuracy: %.4f (%d/%d)" % (acc, sum(1 for t, p in zip(labels, oof_pred) if t == p), n))
    print()
    print("类别      precision  recall   f1       support")
    for c in (1, 0):
        m = per_class[c]
        print("%-6s    %.4f     %.4f   %.4f   %d"
              % (LABEL_NAMES[c], m["precision"], m["recall"], m["f1"], m["support"]))
    macro_p = (per_class[0]["precision"] + per_class[1]["precision"]) / 2.0
    macro_r = (per_class[0]["recall"] + per_class[1]["recall"]) / 2.0
    macro_f1 = (per_class[0]["f1"] + per_class[1]["f1"]) / 2.0
    print("macro     %.4f     %.4f   %.4f   %d" % (macro_p, macro_r, macro_f1, n))

    print()
    print("混淆矩阵（行=真实，列=预测）:")
    print("                预测负面   预测正面")
    print("  真实负面        %4d       %4d" % (cm[0][0], cm[0][1]))
    print("  真实正面        %4d       %4d" % (cm[1][0], cm[1][1]))

    # 错例（最多 8 条），方便判断模型到底错在哪类难例上
    errors = [(i, rows[i]["text"], labels[i], oof_pred[i]) for i in range(n) if labels[i] != oof_pred[i]]
    print("\n" + "-" * 62)
    print("错例（共 %d 条，最多展示 8 条）" % len(errors))
    print("-" * 62)
    for idx, text, true_label, pred_label in errors[:8]:
        print("  [%s→%s] %s" % (LABEL_NAMES[true_label], LABEL_NAMES[pred_label], text))
    print()


# --------------------------------------------------------------------------- #
# 命令：--train
# --------------------------------------------------------------------------- #
def build_full_model(rows, verbose=True):
    """用全量数据训练；alpha 用固定种子的 80/20 dev 划分选择，温度在 dev 上校准。"""
    features = [extract_features(r["text"]) for r in rows]
    labels = [r["label"] for r in rows]
    indices = list(range(len(rows)))
    best_alpha, dev_accs, sub_train, dev_idx = select_alpha(features, labels, indices, seed=SEED)

    # 用训练子集拟合一个模型，在 dev 上定温度（避免用测试信息）
    sub_model = fit_model([features[i] for i in sub_train], [labels[i] for i in sub_train], best_alpha)
    temperature, dev_logloss = fit_temperature(sub_model, features, labels, dev_idx)

    model = fit_model(features, labels, best_alpha)
    model.temperature = temperature
    if verbose:
        print("全量训练: %d 条（其中 %d 条训练 / %d 条 dev 用于选 alpha 和温度）"
              % (len(rows), len(sub_train), len(dev_idx)))
        print("alpha 候选的 dev 准确率: " + "  ".join("α=%s %.4f" % (a, dev_accs[a]) for a in ALPHAS))
        print("选定 alpha: %s" % best_alpha)
        print("置信度温度 T: %s（dev log loss %.4f）" % (temperature, dev_logloss))
        print("词表大小: %d（字符 1-gram + 2-gram）" % model.vocab_size)
        train_pred = [model.predict_one(f)[0] for f in features]
        train_acc = sum(1 for t, p in zip(labels, train_pred) if t == p) / float(len(labels))
        print("训练集拟合准确率: %.4f（仅表示记忆能力，不是泛化指标）" % train_acc)
    return model


def cmd_train():
    rows = load_data()
    print("=" * 62)
    print("在全量数据上训练并保存模型")
    print("=" * 62)
    model = build_full_model(rows)
    save_model(model)
    print("模型已保存: %s" % MODEL_PATH)
    print("提示: 之后 --predict 会直接加载该模型")


# --------------------------------------------------------------------------- #
# 命令：--predict
# --------------------------------------------------------------------------- #
def cmd_predict(text):
    text = (text or "").strip()
    if not text:
        raise SystemExit("--predict 需要一段非空文本")

    if os.path.exists(MODEL_PATH):
        model = load_model()
        source = "model.json（已训练，未重新拟合）"
    else:
        rows = load_data()
        model = build_full_model(rows, verbose=False)
        source = "未找到 model.json，已用全量数据即时训练"

    feats = extract_features(text)
    label, p1 = model.predict_one(feats)
    confidence = p1 if label == 1 else 1.0 - p1

    print("=" * 62)
    print("预测结果")
    print("=" * 62)
    print("输入: %s" % text)
    print("模型: %s" % source)
    print("模型参数: alpha=%s，置信度温度 T=%s" % (model.alpha, model.temperature))
    print("情感: %s" % LABEL_NAMES[label])
    print("置信度: %.4f" % confidence)
    print("证据特征（top 5，正数支持正面、负数支持负面）:")
    print_top_features(model, feats, top_n=5)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def build_parser():
    parser = argparse.ArgumentParser(
        description="中文短评情感分析器：字符 n-gram + 多项式朴素贝叶斯（纯标准库）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python main.py --eval\n"
            "  python main.py --predict \"东西不错但物流太慢\"\n"
            "  python main.py --train"
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--eval", action="store_true", help="跑分层 5 折交叉验证并打印全部指标")
    group.add_argument("--predict", metavar="TEXT", help="预测一条中文短评的情感与置信度")
    group.add_argument("--train", action="store_true", help="在全量数据上训练并保存 model.json")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.eval:
        cmd_eval()
    elif args.predict is not None:
        cmd_predict(args.predict)
    elif args.train:
        cmd_train()
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
