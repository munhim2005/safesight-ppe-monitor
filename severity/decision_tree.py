"""
decision_tree.py
----------------
Decision Tree classifier implemented FROM SCRATCH using only numpy.
No sklearn, no scipy — just numpy and pure Python.

Replaces the MLP for PPE violation severity classification.

Why a Decision Tree over an MLP here:
    - Input space is small (9 binary features = 512 possible states)
    - Output is fully interpretable: every prediction traces a path
      through human-readable if/else splits
    - Learns feature importance and split logic from data (not hardcoded)
    - Academically correct choice for low-dimensional binary classification

Architecture:
    - Criterion : Gini impurity (standard for classification)
    - Max depth  : 8  (deep enough to capture all OSHA rule combos)
    - Min samples: 5  (prevents overfitting to noise)
    - Fits in under 1 second on 2000 synthetic samples

The 9-Feature Input (all binary 0/1):
    [0] hard_hat
    [1] vest
    [2] glass       (eye protection)
    [3] glove
    [4] boots
    [5] ear_prot    (ear protection)
    [6] mask
    [7] using_tool  (circular_saw or welding_equipment nearby)
    [8] fire_safety (fire_extinguisher or fire_prevention_net nearby)

Output: one of "NONE", "LOW", "MEDIUM", "HIGH", "STOP_WORK"
"""

import numpy as np

# ── Severity constants ────────────────────────────────────────────────────────
SEVERITY_MAP     = {0: "NONE", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "STOP_WORK"}
SEVERITY_REVERSE = {v: k for k, v in SEVERITY_MAP.items()}

FEATURE_NAMES = [
    "hard_hat", "vest", "glass", "glove", "boots",
    "ear_prot", "mask", "using_tool", "using_hot_work", "fire_safety", "legs_visible", "torso_visible",
]


# ── Tree node ─────────────────────────────────────────────────────────────────

class _Node:
    """A single node in the decision tree."""
    __slots__ = (
        "feature_idx", "threshold",   # split criteria (None if leaf)
        "left", "right",              # child nodes   (None if leaf)
        "label",                      # class label   (None if internal)
        "gini", "n_samples",          # diagnostics
    )

    def __init__(self):
        self.feature_idx = None
        self.threshold   = None
        self.left        = None
        self.right       = None
        self.label       = None
        self.gini        = None
        self.n_samples   = None


# ── Decision Tree ─────────────────────────────────────────────────────────────

class DecisionTree:
    """
    Decision Tree for PPE violation severity classification.

    Usage:
        dt = DecisionTree()
        dt.train(X_train, y_train)
        severity = dt.predict(feature_vector)   # "HIGH", "NONE", etc.

    The tree is automatically saved/loaded from disk — you never need
    to call train() manually in production; the app handles it.
    """

    def __init__(self, max_depth: int = 8, min_samples_split: int = 5,
                 n_classes: int = 5):
        self.max_depth          = max_depth
        self.min_samples_split  = min_samples_split
        self.n_classes          = n_classes
        self._root              = None
        self.feature_importances_: np.ndarray | None = None

    # ── Gini impurity ─────────────────────────────────────────────────────────

    @staticmethod
    def _gini(y: np.ndarray, n_classes: int) -> float:
        """Gini impurity of label array y."""
        n = len(y)
        if n == 0:
            return 0.0
        counts = np.bincount(y, minlength=n_classes)
        probs  = counts / n
        return 1.0 - float(np.sum(probs ** 2))

    # ── Best split ────────────────────────────────────────────────────────────

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """
        Finds the feature and threshold that minimise weighted Gini impurity.
        For binary features, the only meaningful threshold is 0.5.
        """
        n, n_features = X.shape
        best_gini     = float("inf")
        best_feat     = None
        best_thresh   = None

        parent_gini = self._gini(y, self.n_classes)

        for feat in range(n_features):
            # For binary inputs, unique thresholds are just 0.5
            # For safety, iterate over all mid-points between sorted unique values
            values    = np.unique(X[:, feat])
            thresholds = (values[:-1] + values[1:]) / 2 if len(values) > 1 else []

            for thresh in thresholds:
                left_mask  = X[:, feat] <= thresh
                right_mask = ~left_mask

                n_left  = left_mask.sum()
                n_right = right_mask.sum()

                if n_left == 0 or n_right == 0:
                    continue

                gini_left  = self._gini(y[left_mask],  self.n_classes)
                gini_right = self._gini(y[right_mask], self.n_classes)
                weighted   = (n_left * gini_left + n_right * gini_right) / n

                if weighted < best_gini:
                    best_gini  = weighted
                    best_feat  = feat
                    best_thresh = thresh

        gain = parent_gini - best_gini if best_feat is not None else 0.0
        return best_feat, best_thresh, gain

    # ── Recursive build ───────────────────────────────────────────────────────

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int,
               importances: np.ndarray) -> _Node:
        node           = _Node()
        node.n_samples = len(y)
        node.gini      = self._gini(y, self.n_classes)

        # Leaf conditions
        if (depth >= self.max_depth
                or len(y) < self.min_samples_split
                or node.gini == 0.0):
            node.label = int(np.bincount(y, minlength=self.n_classes).argmax())
            return node

        feat, thresh, gain = self._best_split(X, y)

        # No useful split found
        if feat is None:
            node.label = int(np.bincount(y, minlength=self.n_classes).argmax())
            return node

        # Accumulate feature importance (gain × fraction of samples)
        importances[feat] += gain * (len(y) / self._n_train)

        mask              = X[:, feat] <= thresh
        node.feature_idx  = feat
        node.threshold    = thresh
        node.left         = self._build(X[mask],  y[mask],  depth + 1, importances)
        node.right        = self._build(X[~mask], y[~mask], depth + 1, importances)
        return node

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the decision tree to labelled training data.
        Runs in under 1 second for n ≤ 5000 with 9 binary features.
        """
        self._n_train = len(y)
        importances   = np.zeros(X.shape[1])

        print(f"\n[DecisionTree] Training on {len(y)} samples "
              f"(max_depth={self.max_depth})...")
        self._root = self._build(X, y, depth=0, importances=importances)

        # Normalise importances to sum to 1
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances

        acc = self.evaluate(X, y)
        print(f"[DecisionTree] Training accuracy: {acc:.1f}%")
        print(f"[DecisionTree] Top features: "
              + ", ".join(
                  f"{FEATURE_NAMES[i]}({self.feature_importances_[i]:.2f})"
                  for i in np.argsort(self.feature_importances_)[::-1][:4]
              ))

    # ── Prediction ────────────────────────────────────────────────────────────

    def _traverse(self, node: _Node, x: np.ndarray) -> int:
        if node.label is not None:
            return node.label
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(node.left, x)
        return self._traverse(node.right, x)

    def predict(self, X: np.ndarray) -> str | list[str]:
        """
        Predicts severity label(s).
        Accepts a single 1-D vector or a 2-D batch.
        Returns a string for single input, list of strings for batch.
        """
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)
        labels = [SEVERITY_MAP[self._traverse(self._root, row)] for row in X]
        return labels[0] if single else labels

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns a soft probability vector by collecting leaf class counts.
        Mimics the MLP's predict_proba interface so the rest of the
        pipeline needs zero changes.
        """
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)

        def _leaf_counts(node: _Node, x: np.ndarray) -> _Node:
            if node.label is not None:
                return node
            if x[node.feature_idx] <= node.threshold:
                return _leaf_counts(node.left, x)
            return _leaf_counts(node.right, x)

        out = []
        for row in X:
            leaf   = _leaf_counts(self._root, row)
            # uniform distribution over predicted class (hard assignment)
            probs  = np.zeros(self.n_classes)
            probs[leaf.label] = 1.0
            out.append(probs)

        result = np.array(out)
        return result[0] if single else result

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Returns accuracy percentage on a labelled dataset."""
        preds     = self.predict(X)
        if isinstance(preds, str):
            preds = [preds]
        pred_ints = np.array([SEVERITY_REVERSE[p] for p in preds])
        return 100.0 * np.mean(pred_ints == y)

    # ── Tree visualisation (for your defence presentation) ───────────────────

    def print_tree(self, max_lines: int = 60) -> None:
        """
        Prints a text representation of the tree.
        Useful for explaining decisions during a presentation.

        Example output:
            [using_tool <= 0.5]
            ├── YES → [glass <= 0.5]
            │         ├── YES → STOP_WORK  (n=312)
            │         └── NO  → [glove <= 0.5] ...
            └── NO  → [hard_hat <= 0.5] ...
        """
        lines: list[str] = []

        def _recurse(node: _Node, prefix: str, is_left: bool, label: str) -> None:
            if len(lines) >= max_lines:
                return
            connector = "├── " if is_left else "└── "
            if node.label is not None:
                lines.append(f"{prefix}{connector}{label}"
                             f" → {SEVERITY_MAP[node.label]}  (n={node.n_samples})")
            else:
                feat_name = FEATURE_NAMES[node.feature_idx]
                lines.append(f"{prefix}{connector}{label}"
                             f" [{feat_name} <= {node.threshold:.1f}]"
                             f"  (n={node.n_samples}, gini={node.gini:.3f})")
                child_prefix = prefix + ("│   " if is_left else "    ")
                _recurse(node.left,  child_prefix, True,  "YES")
                _recurse(node.right, child_prefix, False, "NO ")

        feat_name = FEATURE_NAMES[self._root.feature_idx]
        lines.append(f"ROOT: [{feat_name} <= {self._root.threshold:.1f}]"
                     f"  (n={self._root.n_samples}, gini={self._root.gini:.3f})")
        child_prefix = "    "
        _recurse(self._root.left,  child_prefix, True,  "YES")
        _recurse(self._root.right, child_prefix, False, "NO ")

        print("\n[DecisionTree] Tree structure:")
        print("\n".join(lines))
        if len(lines) >= max_lines:
            print("    ... (truncated — increase max_lines to see more)")

    # ── Save / Load ───────────────────────────────────────────────────────────
    # The tree is serialised to a single .npz file using a flat array
    # encoding of the node structure (no pickle = no security concerns).

    def _encode(self) -> dict:
        """Encodes tree nodes into parallel flat arrays for npz storage."""
        nodes: list[_Node] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            nodes.append(node)
            if node.left  is not None: stack.append(node.left)
            if node.right is not None: stack.append(node.right)

        # Assign integer IDs
        id_map = {id(n): i for i, n in enumerate(nodes)}

        feature_idx = []
        threshold   = []
        left_id     = []
        right_id    = []
        label       = []
        n_samples   = []

        for node in nodes:
            feature_idx.append(node.feature_idx if node.feature_idx is not None else -1)
            threshold.append(node.threshold   if node.threshold   is not None else -1.0)
            left_id.append(id_map[id(node.left)]  if node.left  is not None else -1)
            right_id.append(id_map[id(node.right)] if node.right is not None else -1)
            label.append(node.label if node.label is not None else -1)
            n_samples.append(node.n_samples)

        return {
            "feature_idx":        np.array(feature_idx, dtype=np.int32),
            "threshold":          np.array(threshold,   dtype=np.float64),
            "left_id":            np.array(left_id,     dtype=np.int32),
            "right_id":           np.array(right_id,    dtype=np.int32),
            "label":              np.array(label,       dtype=np.int32),
            "n_samples":          np.array(n_samples,   dtype=np.int32),
            "feature_importances": self.feature_importances_,
            "meta": np.array([self.max_depth, self.min_samples_split, self.n_classes]),
        }

    def _decode(self, data: dict) -> None:
        """Reconstructs tree from flat arrays."""
        feature_idx = data["feature_idx"]
        threshold   = data["threshold"]
        left_id     = data["left_id"]
        right_id    = data["right_id"]
        label       = data["label"]
        n_samples   = data["n_samples"]

        n_nodes = len(feature_idx)
        nodes   = [_Node() for _ in range(n_nodes)]

        for i, node in enumerate(nodes):
            node.n_samples   = int(n_samples[i])
            node.feature_idx = int(feature_idx[i]) if feature_idx[i] != -1 else None
            node.threshold   = float(threshold[i])  if threshold[i]   != -1 else None
            node.left        = nodes[int(left_id[i])]  if left_id[i]  != -1 else None
            node.right       = nodes[int(right_id[i])] if right_id[i] != -1 else None
            node.label       = int(label[i]) if label[i] != -1 else None

        self._root               = nodes[0]
        self.feature_importances_ = data["feature_importances"]
        meta                     = data["meta"]
        self.max_depth           = int(meta[0])
        self.min_samples_split   = int(meta[1])
        self.n_classes           = int(meta[2])

    def save(self, filepath: str = "dt_model.npz") -> None:
        np.savez(filepath, **self._encode())
        print(f"[DecisionTree] Model saved to {filepath}")

    def load(self, filepath: str = "dt_model.npz") -> None:
        data = np.load(filepath, allow_pickle=False)
        self._decode(data)
        print(f"[DecisionTree] Model loaded from {filepath}")


# ── Standalone: train, evaluate, print tree ───────────────────────────────────
if __name__ == "__main__":
    try:
        from data_generator import generate_dataset, print_distribution
    except ImportError:
        from severity.data_generator import generate_dataset, print_distribution

    X, y = generate_dataset(n_samples=2000)
    print_distribution(y)

    split   = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    dt = DecisionTree(max_depth=8, min_samples_split=5)
    dt.train(X_train, y_train)

    test_acc = dt.evaluate(X_test, y_test)
    print(f"\n[DecisionTree] Test accuracy: {test_acc:.1f}%")

    dt.print_tree()
    dt.save("dt_model.npz")
