#!/usr/bin/env python3
"""
evaluation.py - UCR 评估指标（修复版）

关键修复：
1. 去掉 Top-k 的过度宽容逻辑（旧代码 5 重检查 + 自动降阈值，会把 0 分判成正确）
2. 去掉 F1 的"直接标记异常区间"兜底（旧代码用真实标签直接生成预测，等于作弊）
3. VUS-ROC/PR 保留标准实现
4. 新增 AUROC 指标（旧代码完全没有，导致答辩无法回答排序能力问题）
"""

import numpy as np
from scipy import integrate
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


class UCREvaluator:
    """UCR 标准评估指标"""

    def compute_auroc(self, scores, y_true):
        """
        AUROC — 衡量排序能力，是最该看的指标
        0.5 = 随机水平，<0.5 = 比随机还差
        """
        if len(np.unique(y_true)) < 2:
            return 0.5
        try:
            return float(roc_auc_score(y_true, scores))
        except Exception:
            return 0.5

    def compute_vus_roc(self, scores, y_true):
        """VUS-ROC"""
        if len(np.unique(y_true)) < 2:
            return 0.5
        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            if len(fpr) < 2:
                return 0.5
            try:
                return float(integrate.trapezoid(tpr, fpr))
            except AttributeError:
                return float(integrate.trapz(tpr, fpr))
        except Exception:
            return 0.5

    def compute_vus_pr(self, scores, y_true):
        """VUS-PR / AP"""
        if len(np.unique(y_true)) < 2:
            return float(np.mean(y_true))
        try:
            return float(average_precision_score(y_true, scores))
        except Exception:
            return float(np.mean(y_true))

    def compute_top_k(self, scores, y_true, k_values=(1, 3, 5), tolerance=None):
        """
        Top-k Accuracy（修复版）

        修复：去掉旧代码的 5 重宽容检查和自动降阈值兜底，
        只做最基本的检查：Top-k 预测点是否落在异常区间（含合理容差）内。
        """
        anomaly_indices = np.where(y_true == 1)[0]
        if len(anomaly_indices) == 0:
            return {f"top_{k}": 1.0 for k in k_values}

        anomaly_start = anomaly_indices[0]
        anomaly_end = anomaly_indices[-1]

        # 容差：基于异常长度的合理范围，不再过度放大
        anomaly_length = anomaly_end - anomaly_start + 1
        if tolerance is None:
            if anomaly_length <= 5:
                tolerance = min(anomaly_length * 3, 50)
            elif anomaly_length <= 50:
                tolerance = min(anomaly_length * 2, 100)
            else:
                tolerance = min(anomaly_length, 200)

        top_indices = np.argsort(scores)[::-1][: max(k_values)]
        results = {}
        for k in k_values:
            top_k = top_indices[:k]
            hit = any(
                (anomaly_start - tolerance <= idx <= anomaly_end + tolerance)
                for idx in top_k
            )
            results[f"top_{k}"] = 1.0 if hit else 0.0
        return results

    def compute_point_metrics(self, scores, y_true):
        """
        点级 F1 / Precision / Recall（修复版）

        修复：去掉旧代码的"直接用真实标签区间生成预测"兜底——那等于作弊。
        只做阈值搜索，选最佳 F1（这是标准做法，但必须声明是 oracle threshold）。
        """
        num_anomalies = np.sum(y_true == 1)
        if num_anomalies == 0:
            return {"f1": 1.0, "precision": 1.0, "recall": 1.0, "threshold": 0.0, "method": "no_anomaly"}

        score_min, score_max = np.min(scores), np.max(scores)
        score_range = score_max - score_min
        if score_range < 1e-12:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.0, "method": "constant_score"}

        # 候选阈值：百分位 + 均值标准差
        thresholds = list(np.percentile(scores, [70, 75, 80, 85, 90, 95]))
        score_mean, score_std = np.mean(scores), np.std(scores)
        thresholds += [score_mean + f * score_std for f in (0.5, 1.0, 1.5, 2.0)]
        thresholds += [score_min + f * score_range for f in (0.1, 0.2, 0.3, 0.5, 0.7)]
        thresholds = sorted(set(thresholds))

        best_f1, best_p, best_r, best_t = 0.0, 0.0, 0.0, thresholds[0]
        for t in thresholds:
            y_pred = (scores > t).astype(int)
            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_p, best_r, best_t = f1, p, r, t

        return {
            "f1": best_f1,
            "precision": best_p,
            "recall": best_r,
            "threshold": float(best_t),
            "method": "oracle_threshold_search",
        }

    def evaluate(self, scores, y_true):
        """计算全部指标"""
        scores = np.asarray(scores, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.int32)
        metrics = {}
        metrics.update(self.compute_top_k(scores, y_true))
        metrics["auroc"] = self.compute_auroc(scores, y_true)
        metrics["vus_roc"] = self.compute_vus_roc(scores, y_true)
        metrics["vus_pr"] = self.compute_vus_pr(scores, y_true)
        metrics.update(self.compute_point_metrics(scores, y_true))
        return metrics

    def evaluate_batch(self, all_scores, all_labels):
        """批量评估并求平均"""
        all_metrics = [self.evaluate(s, l) for s, l in zip(all_scores, all_labels)]
        avg = {}
        if all_metrics:
            for key in all_metrics[0]:
                vals = [m[key] for m in all_metrics if isinstance(m[key], (int, float))]
                avg[key] = float(np.mean(vals)) if vals else 0.0
        return avg, all_metrics

    def print_metrics(self, metrics, prefix=""):
        """格式化打印指标"""
        print(f"{prefix}AUROC={metrics['auroc']:.4f}  VUS-ROC={metrics['vus_roc']:.4f}  VUS-PR={metrics['vus_pr']:.4f}")
        print(f"{prefix}Top-1={metrics['top_1']:.1f}  Top-3={metrics['top_3']:.1f}  Top-5={metrics['top_5']:.1f}")
        print(f"{prefix}F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  (threshold={metrics.get('threshold', 0):.4f}, {metrics.get('method', '')})")
