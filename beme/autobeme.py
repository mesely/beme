import numpy as np
import scipy.sparse
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from beme import BemeMarket, MultiLabelWrapper

class AutoBEME:
    """
    BEME (Bounty-Driven Evolutionary Market Ensembles) API Yönlendiricisi.
    Kullanıcının iş hedefine (Recall, Precision, Balanced, Custom) göre
    en uygun piyasa dinamiklerini ve ajan mimarilerini otomatik kurar.

    Auto-Caster: 1D hedef vektörlerini otomatik algılar ve 2D'ye dönüştürür.
    predict() çıktısı da orijinal forma (1D/2D) geri döndürülür.
    """
    def __init__(self, mode='balanced', base_model=None, n_funds=12):
        self.mode = mode.lower()
        self.base_model = base_model
        self.n_funds = n_funds
        self.market = None
        self._is_1d_target = False  # Auto-Caster flag

        if self.mode == 'recall':
            self.threshold = 0.30
        elif self.mode == 'precision':
            self.threshold = 0.65
        else:
            self.threshold = 0.45

        valid_modes = ['recall', 'precision', 'balanced', 'custom']
        if self.mode not in valid_modes:
            raise ValueError(f"Geçersiz mod! Lütfen şunlardan birini seçin: {valid_modes}")

        if self.mode == 'custom' and self.base_model is None:
            raise ValueError("⚠️ 'custom' modu seçildiğinde lütfen bir 'base_model' (Örn: Naive Bayes, LR) belirleyin!")

    def fit(self, X, y):
        # ── Auto-Caster: 1D detection & conversion ──────────────────
        if scipy.sparse.issparse(y):
            y = y.toarray()
        y = np.asarray(y)
        if y.ndim == 1 or (len(y.shape) > 1 and y.shape[1] == 1):
            self._is_1d_target = True
            y_flat = y.ravel()
            y = np.column_stack((1 - y_flat, y_flat))
        else:
            self._is_1d_target = False
        # ────────────────────────────────────────────────────────────

        n_classes = y.shape[1]

        if self.mode == 'recall':
            print("🛡️ AutoBEME: 'VANGUARD' Modu Aktif (Hedef: Maksimum Recall)")
            self.market = BemeMarket(n_funds=self.n_funds, leverage=35.0, bounty_map='auto')
            self.market.T = 1.05

            base_wrapper = MultiLabelWrapper(MultiOutputClassifier(LogisticRegression(solver='liblinear', class_weight='balanced')))
            self.market.fit(X, y, base_model=base_wrapper)

        elif self.mode == 'precision':
            print("🎯 AutoBEME: 'SOVEREIGN' Modu Aktif (Hedef: Maksimum Precision)")
            self.market = BemeMarket(n_funds=self.n_funds, leverage=20.0, bounty_map='auto')
            self.market.T = 0.7

            svm_core = CalibratedClassifierCV(LinearSVC(class_weight='balanced', dual='auto', max_iter=2000), cv=3)
            base_wrapper = MultiLabelWrapper(MultiOutputClassifier(svm_core))
            self.market.fit(X, y, base_model=base_wrapper)

        elif self.mode == 'balanced':
            print("☢️ AutoBEME: 'APOCALYPSE' Modu Aktif (Hedef: Maksimum F1 & SOTA)")
            self.market = BemeMarket(n_funds=self.n_funds, leverage=30.0, bounty_map='auto')
            self.market.T = 1.05

            svm_core = CalibratedClassifierCV(LinearSVC(class_weight={0:1, 1:4}, dual='auto', max_iter=2000), cv=3)
            lr_core = LogisticRegression(solver='liblinear', class_weight={0:1, 1:4})

            self.market.funds = []
            for i in range(self.n_funds):
                order = np.random.permutation(n_classes)
                chain = ClassifierChain(svm_core if i % 2 == 0 else lr_core, order=order)
                wrapped = MultiLabelWrapper(chain)

                idx = np.random.choice(X.shape[0], int(X.shape[0]*0.85), replace=False)
                # ── Sparse-safe bootstrap ────────────────────────────
                X_sub = X.tocsr()[idx] if scipy.sparse.issparse(X) else X[idx]
                # ────────────────────────────────────────────────────
                wrapped.fit(X_sub, y[idx])

                self.market.funds.append({
                    'id': i, 'model': wrapped, 'balance': 100.0,
                    'risk': 0.05, 'sector': "GENERAL", 'active_classes': list(range(n_classes))
                })
            print("⏱️ Apocalypse Market el yordamıyla kuruldu.")

        elif self.mode == 'custom':
            print("🛠️ AutoBEME: 'CUSTOM' Modu Aktif (Kullanıcı Tanımlı Ajanlar)")
            self.market = BemeMarket(n_funds=self.n_funds, leverage=25.0, bounty_map='auto')
            self.market.T = 1.0

            wrapped_base = MultiLabelWrapper(MultiOutputClassifier(self.base_model))
            self.market.fit(X, y, base_model=wrapped_base)

        return self  # sklearn chaining: fit(...).predict(...)

    def predict(self, X):
        # balanced: _hack_predict (ClassifierChain ensemblı, tüm sınıflar tek fonda)
        # diğerleri: market.predict (BemeMarket.fit sektörlere böler, active_classes değişir)
        if self.mode == 'balanced':
            predictions = self._hack_predict(X, threshold=self.threshold)
        else:
            predictions = self.market.predict(X)

        # ── Auto-Caster: inverse transform ──────────────────────────
        if self._is_1d_target:
            return predictions[:, 1]
        return predictions
        # ────────────────────────────────────────────────────────────

    def _hack_predict(self, X, threshold):
        n_samples = X.shape[0]
        n_classes = len(self.market.funds[0]['active_classes'])
        total_bal = sum(f['balance'] for f in self.market.funds)
        market_probs = np.zeros((n_samples, n_classes))

        for f in self.market.funds:
            wrapper = f['model']
            proba = None
            for attr in ['estimator', 'base_estimator', 'model', 'clf', '_estimator', 'classifier']:
                if hasattr(wrapper, attr):
                    inner = getattr(wrapper, attr)
                    if hasattr(inner, 'predict_proba'):
                        proba = inner.predict_proba(X)
                        break
            if proba is None:
                for key, val in vars(wrapper).items():
                    if hasattr(val, 'predict_proba'):
                        proba = val.predict_proba(X)
                        break

            if proba is None:
                raise ValueError(f"Ajan {f['id']} içinde olasılık üreten motor bulunamadı!")

            # Handle list output (e.g. MultiOutputClassifier) vs matrix output
            if isinstance(proba, list):
                proba = np.column_stack([
                    arr[:, 1] if arr.shape[1] >= 2 else np.zeros(n_samples)
                    for arr in proba
                ])

            weight = f['balance'] / total_bal
            market_probs += proba * weight

        return (market_probs > threshold).astype(int)
