import numpy as np
from pathlib import Path

class MLPClassifier:
    def __init__(self, input_dim: int = 512, hidden_dim: int = 128, labels: list[str] | None = None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.labels = labels or []
        self.num_classes = len(self.labels)
        if self.num_classes > 0:
            self._init_weights()

    def _init_weights(self):
        self.W1 = (np.random.randn(self.input_dim, self.hidden_dim).astype(np.float32)
                    * np.sqrt(2.0 / self.input_dim))
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W2 = (np.random.randn(self.hidden_dim, self.num_classes).astype(np.float32)
                    * np.sqrt(2.0 / self.hidden_dim))
        self.b2 = np.zeros(self.num_classes, dtype=np.float32)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """(batch, 512) -> (batch, num_classes) probabilities."""
        self._z1 = X @ self.W1 + self.b1
        self._a1 = self._relu(self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        return self._softmax(self._z2)

    def predict(self, embeddings: list[list[float]]) -> list[str]:
        """Return predicted content_category for each embedding vector."""
        X = np.asarray(embeddings, dtype=np.float32)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        probs = self.forward(X)
        return [self.labels[i] for i in np.argmax(probs, axis=-1)]

    def predict_proba(self, embeddings: list[list[float]]) -> list[dict[str, float]]:
        """Return {label: probability} for each embedding vector."""
        X = np.asarray(embeddings, dtype=np.float32)
        if X.ndim == 1:
            X = X[np.newaxis, :]
        probs = self.forward(X)
        return [
            {label: float(p) for label, p in zip(self.labels, row)}
            for row in probs
        ]

    def loss(self, X: np.ndarray, y_idx: np.ndarray) -> float:
        """Average cross-entropy loss (no gradient, for eval)."""
        probs = self.forward(X)
        return float(-np.log(probs[np.arange(len(y_idx)), y_idx] + 1e-9).mean())

    def accuracy(self, X: np.ndarray, y_idx: np.ndarray) -> float:
        """Accuracy (no gradient, for eval)."""
        probs = self.forward(X)
        return float((np.argmax(probs, axis=-1) == y_idx).mean())

    def train(self, X: np.ndarray | list, y: list[str],
              epochs: int = 100, lr: float = 0.01, batch_size: int = 32,
              X_val: np.ndarray | None = None, y_val: list[str] | None = None,
              ) -> dict[str, list[float]]:
        """Train in-place. Returns {"train_loss": [...], "val_loss": [...]}."""
        X = np.asarray(X, dtype=np.float32)
        label_to_idx = {l: i for i, l in enumerate(self.labels)}
        y_idx = np.array([label_to_idx[label] for label in y])
        n = len(X)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val_idx = np.array([label_to_idx[label] for label in y_val])

        history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }
        log_every = max(1, epochs // 10)

        for epoch in range(epochs):
            perm = np.random.permutation(n)
            X_s, y_s = X[perm], y_idx[perm]
            epoch_loss = 0.0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                Xb, yb = X_s[start:end], y_s[start:end]
                bs = end - start

                # forward
                probs = self.forward(Xb)
                epoch_loss += -np.log(probs[np.arange(bs), yb] + 1e-9).sum()

                # backward (softmax + cross-entropy shortcut)
                dz2 = probs.copy()
                dz2[np.arange(bs), yb] -= 1
                dz2 /= bs

                dW2 = self._a1.T @ dz2
                db2 = dz2.sum(axis=0)

                da1 = dz2 @ self.W2.T
                dz1 = da1 * (self._z1 > 0).astype(np.float32)

                dW1 = Xb.T @ dz1
                db1 = dz1.sum(axis=0)

                # SGD step
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db2

            train_avg = epoch_loss / n
            train_acc = self.accuracy(X, y_idx)
            history["train_loss"].append(train_avg)
            history["train_acc"].append(train_acc)

            if has_val:
                val_avg = self.loss(X_val, y_val_idx)
                val_acc = self.accuracy(X_val, y_val_idx)
                history["val_loss"].append(val_avg)
                history["val_acc"].append(val_acc)

            if (epoch + 1) % log_every == 0:
                msg = f"epoch {epoch+1:>4}/{epochs}  train_loss={train_avg:.4f}  train_acc={train_acc:.4f}"
                if has_val:
                    msg += f"  val_loss={val_avg:.4f}  val_acc={val_acc:.4f}"
                print(msg)

        return history

    def save(self, path: str | Path) -> None:
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 labels=np.array(self.labels))

    @classmethod
    def load(cls, path: str | Path) -> "MLPClassifier":
        d = np.load(path, allow_pickle=True)
        labels = [str(l) for l in d["labels"]]
        m = cls(input_dim=d["W1"].shape[0], hidden_dim=d["W1"].shape[1], labels=labels)
        m.W1, m.b1, m.W2, m.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        return m


if __name__ == "__main__":
    from embed import embed_sync

    labels = ["news", "blog", "tutorial", "review", "discussion"]
    clf = MLPClassifier(input_dim=512, hidden_dim=128, labels=labels)

    fake_X = np.random.randn(20, 512).astype(np.float32)
    fake_y = [labels[i % len(labels)] for i in range(20)]
    clf.train(fake_X, fake_y, epochs=50, lr=0.05)

    preds = clf.predict(fake_X[:3])
    print(f"predictions: {preds}")

    vecs = embed_sync("Hello world")
    print(f"content_category: {clf.predict(vecs)[0]}")
