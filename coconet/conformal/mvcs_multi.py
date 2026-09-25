"""Multi-norm MVCS baseline, run with the MVCS authors' own code (fetch it with
`bash scripts/get_mvcs.sh`, which clones their repository at a fixed commit into third_party/MVCS).

One global set, the same for every input: C(x) = F(x) + {r : s(r) <= q}. The authors' score rotates
r - mu, picks the orthant j of the first two rotated coordinates and returns ||D_j R (r - mu)||_{q_j}.
Everything about the fit is theirs (parameterization, loss, defaults, checkpoint rule), called as in
their notebooks with n_splits = 2. The shared pieces are the residuals it is trained on, the
split-conformal threshold every method gets, and the reported volume. The convex variant keeps every
exponent q_j >= 1 by projecting after each optimizer step; nothing else changes.
"""

import sys
from pathlib import Path

import numpy as np
import torch

AUTHORS_CODE = Path(__file__).resolve().parents[2] / "third_party" / "MVCS" / "experiments" / "one_matrix_MVCS" / "code"
N_SPLITS = 2


def _authors_modules():
    if not AUTHORS_CODE.is_dir():
        raise SystemExit("multi-norm MVCS needs the authors' code: run `bash scripts/get_mvcs.sh` first")
    if str(AUTHORS_CODE) not in sys.path:
        sys.path.insert(0, str(AUTHORS_CODE))
    import half_ellipsoid_predictor
    import utils
    return half_ellipsoid_predictor, utils.seed_everything


class MultiNormMvcs:
    def __init__(self, convex):
        self.convex = bool(convex)
        self.hep, self.seed_everything = _authors_modules()

    def fit(self, r_train, alpha, seed):
        self.seed_everything(seed)
        self.m = self.hep.HalfEllipsoidPredictor(N_SPLITS)
        adam = torch.optim.Adam
        if self.convex:
            class ProjectedAdam(adam):          # the exponents are the third parameter group of their optimizer
                def step(self, closure=None):
                    out = super().step(closure)
                    if len(self.param_groups) == 3:
                        with torch.no_grad():
                            for q in self.param_groups[2]["params"]:
                                q.clamp_(min=1.0)
                    return out
            torch.optim.Adam = ProjectedAdam
        try:
            self.m.fit(np.asarray(r_train, float), alpha, verbose=0)
        finally:
            torch.optim.Adam = adam
        return self

    def scores(self, r):
        """The authors' per-point score s(r)."""
        m, hep = self.m, self.hep
        x = torch.ones(len(r), r.shape[1])
        rt = torch.as_tensor(np.asarray(r, float), dtype=torch.float32)
        with torch.no_grad():
            f = m.model(x)
            rot = m.rotationMatrix_model(x)
            diags = [torch.stack([torch.diag(d) for d in dm(x)], 0) for dm in m.tab_diag_models]
            z = torch.bmm(rot, (rt - f).unsqueeze(-1)).squeeze(-1)
            idx = hep.get_idx_Lambdas(z, m.split)
            s = [hep.calculate_norm_q(diags[idx[i]][i] @ z[i], m.tab_q[idx[i]]).item() for i in range(len(rt))]
        self._diags = diags
        return np.asarray(s, float)

    def volume(self, q, d):
        """Closed-form volume of {s <= q} (standardized units), from the authors' helper."""
        diags = self._diags
        return float(self.hep.compute_volume_weighted_ellipsoids(
            [dg[:1] for dg in diags], [torch.tensor(float(v)) for v in self.m.tab_q], self.m.tab_weights,
            torch.tensor(q), d))

    def exponents(self):
        return [float(v) for v in self.m.tab_q]

    def centre(self):
        return [float(v) for v in self.m.mu]

    def gauge(self, d):
        """s(r + mu) as a float64 torch function of r: the set with its centre moved to the origin, which
        is what the shared volume estimator traces rays from (volume is translation invariant, and the
        score is positively 1-homogeneous around mu, so every ray crosses the boundary once)."""
        m = self.m
        one = torch.ones(1, d)
        with torch.no_grad():
            R = m.rotationMatrix_model(one)[0].double()
            D = torch.stack([dm(one)[0] for dm in m.tab_diag_models]).double()
        Q = torch.tensor(self.exponents(), dtype=torch.float64)
        bits = 2 ** torch.arange(N_SPLITS)

        def score(r):
            rr = r.double()
            z = rr @ R.to(rr.device).T
            j = ((z[:, :N_SPLITS] >= 0).long() * bits.to(rr.device)).sum(1)
            v = D.to(rr.device)[j] * z
            qj = Q.to(rr.device)[j]
            return (v.abs().clamp_min(1e-300).log().mul(qj[:, None]).logsumexp(1) / qj).exp().to(r.dtype)
        return score
