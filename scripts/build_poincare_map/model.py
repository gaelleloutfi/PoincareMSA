# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sklearn.metrics.pairwise import pairwise_distances
from torch.autograd import Function
from torch import nn
import numpy as np
import torch

eps = 1e-5
boundary = 1 - eps

def poincare_translation(v, x):
    """
    Computes the translation of x  when we move v to the center.
    Hence, the translation of u with -u should be the origin.
    """
    xsq = (x ** 2).sum(axis=1)
    vsq = (v ** 2).sum()
    xv = (x * v).sum(axis=1)
    a = np.matmul((xsq + 2 * xv + 1).reshape(-1, 1),
                  v.reshape(1, -1)) + (1 - vsq) * x
    b = xsq * vsq + 2 * xv + 1
    return np.dot(np.diag(1. / b), a)


def poincare_root(opt, labels, features):
    if opt.root is not None:
        head_idx = np.where(labels == opt.root)[0]

        if len(head_idx) > 1:
            # medoids in Euclidean space
            D = pairwise_distances(features[head_idx, :], metric='euclidean')
            return head_idx[np.argmin(D.mean(axis=0))]
        else:
            return head_idx[0]

    return -1


def grad(x, v, sqnormx, sqnormv, sqdist):
    alpha = (1 - sqnormx)
    beta = (1 - sqnormv)        
    z = 1 + 2 * sqdist / (alpha * beta)
    a = ((sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) /
            torch.pow(alpha, 2)).unsqueeze(-1).expand_as(x)
    a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
    z = torch.sqrt(torch.pow(z, 2) - 1)
    z = torch.clamp(z * beta, min=eps).unsqueeze(-1)
    return 4 * a / z.expand_as(x)


class PoincareDistance(Function):
    @staticmethod
    def forward(self, u, v):  
        self.save_for_backward(u, v)
        self.squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, boundary)
        self.sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, boundary)
        self.sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
        x = self.sqdist / ((1 - self.squnorm) * (1 - self.sqvnorm)) * 2 + 1
        # arcosh
        z = torch.sqrt(torch.pow(x, 2) - 1)
        return torch.log(x + z)

    @staticmethod
    def backward(self, g):    
        u, v = self.saved_tensors
        g = g.unsqueeze(-1)
        gu = grad(u, v, self.squnorm, self.sqvnorm, self.sqdist)
        gv = grad(v, u, self.sqvnorm, self.squnorm, self.sqdist)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv

    
def klSym(preds, targets):
    # preds = preds + eps
    # targets = targets + eps
    logPreds = preds.clamp(1e-20).log()
    logTargets = targets.clamp(1e-20).log()
    diff = targets - preds
    return (logTargets * diff - logPreds * diff).sum() / len(preds)


class PoincareEmbedding(nn.Module):
    def __init__(self,
                 size,
                 dim,
                 dist=PoincareDistance,
                 max_norm=1,
                 Qdist='laplace',
                 lossfn='klSym',
                 gamma=1.0,
                 cuda=0):
        # Use implicit super() for Python 3 compatibility and to avoid
        # issues if the class object is rebound in the importing module.
        super().__init__()

        self.dim = dim
        self.size = size
        self.lt = nn.Embedding(size, dim, max_norm=max_norm)

        ## pour ajout de points : initialiser ici avec les poids de l'ancien embedding ?
        self.lt.weight.data.uniform_(-1e-4, 1e-4)
        #####

        self.dist = dist
        self.Qdist = Qdist
        self.lossfnname = lossfn
        self.gamma = gamma

        self.sm = nn.Softmax(dim=1)
        self.lsm = nn.LogSoftmax(dim=1)

        if lossfn == 'kl':
            self.lossfn = nn.KLDivLoss()
        elif lossfn == 'klSym':
            self.lossfn = klSym
        elif lossfn == 'mse':
            self.lossfn = nn.MSELoss()
        else:
            raise NotImplementedError

        if cuda:
            self.lt.cuda()

    def forward(self, inputs):
        embs_all = self.lt.weight.unsqueeze(0)
        embs_all = embs_all.expand(len(inputs), self.size, self.dim)

        embs_inputs = self.lt(inputs).unsqueeze(1)
        embs_inputs = embs_inputs.expand_as(embs_all)

        dists = self.dist().apply(embs_inputs, embs_all).squeeze(-1)        

        if self.lossfnname == 'kl':
            if self.Qdist == 'laplace':
                return self.lsm(-self.gamma * dists)
            elif self.Qdist == 'gaussian':
                return self.lsm(-self.gamma * dists.pow(2))
            elif self.Qdist == 'student':
                return self.lsm(-torch.log(1 + self.gamma * dists))
            else:
                raise NotImplementedError
        elif self.lossfnname == 'klSym':
            if self.Qdist == 'laplace':
                return self.sm(-self.gamma * dists)
            elif self.Qdist == 'gaussian':
                return self.sm(-self.gamma * dists.pow(2))
            elif self.Qdist == 'student':
                return self.sm(-torch.log(1 + self.gamma * dists))
            else:
                raise NotImplementedError
        elif self.lossfnname == 'mse':
            return self.sm(-self.gamma * dists)
        else:
            raise NotImplementedError

    def infer_embedding_for_point(self,
                                  target,
                                  n_steps: int = 200,
                                  lr: float = 0.1,
                                  init: str = 'random',
                                  init_vec=None,
                                  device: str = None):
        """
        Infers an embedding vector for a single new point given its target
        similarity/distribution.

        This routine keeps the existing embeddings fixed and optimizes a
        single new embedding so that its predicted distribution (Q) matches
        the provided `target` distribution (length == current size).

        Args:
            target: 1D array-like (length == self.size) or torch tensor with
                similarity/probability scores between the new point and each
                existing item. The method will normalize it to sum to 1.
            n_steps: number of optimization steps.
            lr: learning rate for the per-point optimizer.
            init: 'random' (uniform small) or 'zeros'.
            device: torch device string, if None uses model parameters device.

        Returns:
            numpy array of shape (dim,) with the inferred embedding (inside
            the Poincaré ball, i.e. norm < 1).

        Notes:
            - This does not modify the model's existing embedding table.
            - The new point is optimized alone, using the current embeddings as
              fixed anchors (detached). This is a cheap way to "add" a point
              to an existing projection without retraining everything.
        """
        # device selection
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device('cpu')

        # prepare fixed existing embeddings (detached)
        with torch.no_grad():
            old_embs = self.lt.weight.detach().clone().to(device)

        # prepare target tensor (length == old_size)
        if not isinstance(target, torch.Tensor):
            target_t = torch.tensor(target, dtype=torch.float32, device=device)
        else:
            target_t = target.to(device).float()

        # Accept common 2D shapes (1, N) or (N, 1) by flattening to 1D.
        if target_t.ndim > 1:
            if target_t.shape[0] == 1:
                target_t = target_t.ravel()
            elif target_t.shape[1] == 1:
                target_t = target_t.ravel()

        if target_t.ndim != 1 or target_t.shape[0] != old_embs.shape[0]:
            raise ValueError(
                f"target must be 1D and length == {old_embs.shape[0]}; got shape {tuple(target_t.shape)}. "
                "If you computed target from distances, ensure it is a 1D array of length N (number of existing embeddings). "
                "Example: target = pairwise_distances(new_feat.reshape(1,-1), features).flatten()"
            )

        # normalize target to a probability vector
        if target_t.sum() <= 0:
            # avoid division by zero — use uniform small mass
            target_t = torch.ones_like(target_t) / float(target_t.shape[0])
        else:
            target_t = target_t / target_t.sum()

        # initialize new embedding parameter
        # init priority: init_vec (explicit) > 'barycenter' > random/zeros
        if init_vec is not None:
            # accept numpy array or torch tensor
            if not isinstance(init_vec, torch.Tensor):
                v = torch.tensor(init_vec, dtype=torch.float32, device=device).reshape(1, -1)
            else:
                v = init_vec.to(device).float().reshape(1, -1)
            v = self._project_to_ball(v)
            new = torch.nn.Parameter(v)
        elif init == 'barycenter':
            # compute top-k neighbors from target and compute barycenter as warm start
            old_size = old_embs.shape[0]
            k = min(50, old_size)
            topk = torch.topk(target_t, k=k).indices
            neighbor_embs = old_embs[topk]
            neighbor_w = target_t[topk]
            neighbor_w = neighbor_w / neighbor_w.sum()
            v = self.hyperbolic_barycenter(neighbor_embs, neighbor_w, n_steps=100, tol=1e-7, alpha=1.0, device=device)
            v = v.to(device).float()
            new = torch.nn.Parameter(v)
        else:
            new = torch.zeros((1, self.dim), dtype=torch.float32, device=device)
            if init == 'random':
                new.uniform_(-1e-4, 1e-4)
            elif init == 'zeros':
                pass
            else:
                raise ValueError("init must be 'random', 'zeros' or 'barycenter' (or provide init_vec)")
            new = torch.nn.Parameter(new)

        optim = torch.optim.SGD([new], lr=lr)

        lossfn = self.lossfn

        for _ in range(n_steps):
            optim.zero_grad()

            # build batch shapes compatible with forward
            # embs_all : (1, old_size + 1, dim)
            embs_all = torch.cat([old_embs, new.detach()], dim=0).unsqueeze(0)

            # embs_inputs : start from new shape (1, dim) -> unsqueeze to (1,1,dim)
            # then expand to (1, old_size + 1, dim) to match embs_all
            embs_inputs = new.unsqueeze(0).expand(1, embs_all.size(1), self.dim)

            dists = self.dist().apply(embs_inputs, embs_all).squeeze(-1)  # (1, N+1)

            # we only have target for the existing points; append self-similarity
            # for the new point (last entry) as a small epsilon so shapes match
            pad = torch.tensor([1e-12], dtype=torch.float32, device=device)
            target_ext = torch.cat([target_t, pad], dim=0).unsqueeze(0)
            # renormalize
            target_ext = target_ext / target_ext.sum(dim=1, keepdim=True)

            # compute Q as in forward
            if self.lossfnname == 'kl':
                if self.Qdist == 'laplace':
                    preds = self.lsm(-self.gamma * dists)
                elif self.Qdist == 'gaussian':
                    preds = self.lsm(-self.gamma * dists.pow(2))
                elif self.Qdist == 'student':
                    preds = self.lsm(-torch.log(1 + self.gamma * dists))
                else:
                    raise NotImplementedError
            elif self.lossfnname == 'klSym' or self.lossfnname == 'mse':
                if self.Qdist == 'laplace':
                    preds = self.sm(-self.gamma * dists)
                elif self.Qdist == 'gaussian':
                    preds = self.sm(-self.gamma * dists.pow(2))
                elif self.Qdist == 'student':
                    preds = self.sm(-torch.log(1 + self.gamma * dists))
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            loss = lossfn(preds, target_ext)
            loss.backward()

            # gradient step on the new embedding
            optim.step()

            # retract to Poincaré ball (ensure norm < 1 - eps)
            with torch.no_grad():
                norm = new.norm(p=2)
                if norm >= boundary:
                    new.mul_((boundary - 1e-8) / norm)

        return new.detach().cpu().numpy().reshape(self.dim,)

    def infer_batch_embedding(
        self,
        targets,
        n_steps: int = 500,
        lr: float = 0.05,
        init_vecs=None,
        device: str = None,
    ):
        """
        Jointly optimize positions for k new points via shared SGD.

        Unlike calling infer_embedding_for_point k times, here all k positions
        are free parameters optimized simultaneously. Each protein i's loss
        depends on the current positions of the other k-1 batch proteins
        (they appear as anchors in its embedding pool), so the gradients couple
        all k positions in each step.

        Parameters
        ----------
        targets : list of k array-like, each of length (N_existing + k - 1).
            targets[i] is the target RFA distribution for batch protein i
            against the N_existing base proteins followed by the k-1 other
            batch proteins in index order [j for j in range(k) if j != i].
            Must already be normalized to sum to 1.
        n_steps : int
            Number of SGD steps.
        lr : float
            Learning rate.
        init_vecs : array-like of shape (k, dim), optional
            Initial positions inside the Poincaré ball. If None, small random
            noise is used. Passing barycenter positions is recommended.
        device : str, optional

        Returns
        -------
        positions : np.ndarray of shape (k, dim)
        """
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device('cpu')

        k = len(targets)

        # Fixed base embeddings — no gradient needed
        with torch.no_grad():
            old_embs = self.lt.weight.detach().clone().to(device)  # (N_existing, dim)

        # Prepare and normalize targets
        target_tensors = []
        for t in targets:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, dtype=torch.float32, device=device)
            else:
                t = t.to(device).float()
            t = t / t.sum().clamp(min=1e-12)
            target_tensors.append(t)

        # Initialize k positions as a single (k, dim) Parameter
        if init_vecs is not None:
            if not isinstance(init_vecs, torch.Tensor):
                init_t = torch.tensor(init_vecs, dtype=torch.float32, device=device)
            else:
                init_t = init_vecs.to(device).float()
            init_t = init_t.reshape(k, self.dim)
            # per-row projection onto the open Poincaré ball
            norms = init_t.norm(p=2, dim=1, keepdim=True)
            over = (norms >= boundary).squeeze(1)
            if over.any():
                init_t[over] = init_t[over] / norms[over] * (boundary - 1e-8)
        else:
            init_t = torch.zeros(k, self.dim, dtype=torch.float32, device=device)
            init_t.uniform_(-1e-4, 1e-4)

        new_embs = torch.nn.Parameter(init_t)
        # Scale lr by 1/k: each parameter accumulates k gradient terms per step
        # (1 direct + k-1 indirect from coupling), so naive lr is k× too large.
        effective_lr = lr / max(k, 1)
        optim = torch.optim.SGD([new_embs], lr=effective_lr)
        lossfn = self.lossfn

        for _ in range(n_steps):
            optim.zero_grad()
            total_loss = torch.zeros(1, device=device)

            for i in range(k):
                # Anchor pool for protein i: base map + other k-1 batch proteins
                other_idx = [j for j in range(k) if j != i]
                other_new  = new_embs[other_idx]          # (k-1, dim) — keeps gradient
                embs_pool  = torch.cat([old_embs, other_new], dim=0)  # (N_existing+k-1, dim)
                N_pool     = embs_pool.size(0)

                pos_i        = new_embs[i:i+1]                                    # (1, dim)
                embs_all_i   = embs_pool.unsqueeze(0)                             # (1, N_pool, dim)
                embs_input_i = pos_i.unsqueeze(0).expand(1, N_pool, self.dim)     # (1, N_pool, dim)

                dists_i = self.dist().apply(embs_input_i, embs_all_i).squeeze(-1) # (1, N_pool)

                if self.lossfnname in ('klSym', 'mse'):
                    if self.Qdist == 'laplace':
                        preds_i = self.sm(-self.gamma * dists_i)
                    elif self.Qdist == 'gaussian':
                        preds_i = self.sm(-self.gamma * dists_i.pow(2))
                    else:
                        preds_i = self.sm(-torch.log(1 + self.gamma * dists_i))
                else:  # 'kl'
                    if self.Qdist == 'laplace':
                        preds_i = self.lsm(-self.gamma * dists_i)
                    elif self.Qdist == 'gaussian':
                        preds_i = self.lsm(-self.gamma * dists_i.pow(2))
                    else:
                        preds_i = self.lsm(-torch.log(1 + self.gamma * dists_i))

                total_loss = total_loss + lossfn(preds_i, target_tensors[i].unsqueeze(0))

            total_loss.backward()
            optim.step()

            # Per-row retraction onto the Poincaré ball
            with torch.no_grad():
                norms = new_embs.norm(p=2, dim=1)
                over  = norms >= boundary
                if over.any():
                    new_embs.data[over] = (
                        new_embs.data[over]
                        / norms[over].unsqueeze(1)
                        * (boundary - 1e-8)
                    )

        return new_embs.detach().cpu().numpy()

    def train_single_point(
        self,
        target,
        n_steps: int = 300,
        lr: float = 0.05,
        init: str = 'random',
        device: str = None,
        verbose: bool = False,
        k: int = 10,
        lambda_local: float = 1.0,
    ):
        """
        Infers a Poincaré embedding for a single new point while keeping
        existing embeddings fixed, with an explicit local attraction
        to nearest neighbors.

        Args:
            target: 1D array-like of length self.size
                    Similarity / probability distribution to existing points.
            n_steps: number of gradient steps.
            lr: learning rate.
            init: 'random' or 'zeros'.
            device: torch device (optional).
            verbose: print loss during optimization.
            k: number of nearest neighbors used for local attraction.
            lambda_local: weight of the local metric loss.

        Returns:
            new_embedding: numpy array of shape (dim,)
            losses: list of loss values
        """

        # device
        if device is None:
            device = next(self.parameters()).device

        # freeze existing embeddings
        with torch.no_grad():
            old_embs = self.lt.weight.detach().clone().to(device)

        # target distribution
        target = torch.tensor(target, dtype=torch.float32, device=device)
        if target.ndim != 1 or target.shape[0] != old_embs.shape[0]:
            raise ValueError(f"target must be 1D of length {old_embs.shape[0]}")

        if target.sum() <= 0:
            target = torch.ones_like(target) / target.numel()
        else:
            target = target / target.sum()

        # select top-k neighbors
        k = min(k, target.numel())
        topk = torch.topk(target, k=k).indices
        neighbor_embs = old_embs[topk]              # (k, dim)
        neighbor_w = target[topk]
        neighbor_w = neighbor_w / neighbor_w.sum()  # normalized weights

        # initialize new point
        # support init == 'random'|'zeros'|'barycenter' or an init_vec provided via closure (not arg here)
        if init == 'barycenter':
            # compute barycenter from top-k neighbors
            k_local = min(max(1, k), target.numel())
            topk = torch.topk(target, k=k_local).indices
            neighbor_embs = old_embs[topk]
            neighbor_w = target[topk]
            neighbor_w = neighbor_w / neighbor_w.sum()
            v = self.hyperbolic_barycenter(neighbor_embs, neighbor_w, n_steps=100, tol=1e-7, alpha=1.0, device=device)
            new = torch.nn.Parameter(v)
        else:
            new = torch.zeros((1, self.dim), device=device)
            if init == 'random':
                new.uniform_(-1e-4, 1e-4)
            elif init == 'zeros':
                pass
            else:
                raise ValueError("init must be 'random', 'zeros' or 'barycenter'")
            new = torch.nn.Parameter(new)
        optimizer = torch.optim.SGD([new], lr=lr)

        losses = []

        for step in range(n_steps):
            optimizer.zero_grad()

            # global distances (for KL term)
            embs_all = torch.cat([old_embs, new], dim=0).unsqueeze(0)
            embs_new = new.unsqueeze(0).expand_as(embs_all)
            dists = self.dist().apply(embs_new, embs_all).squeeze(-1)

            # target with self-distance padding
            pad = torch.tensor([1e-12], device=device)
            target_ext = torch.cat([target, pad]).unsqueeze(0)
            target_ext = target_ext / target_ext.sum(dim=1, keepdim=True)

            # predicted distribution (same logic as forward)
            if self.lossfnname == 'kl':
                if self.Qdist == 'laplace':
                    preds = self.lsm(-self.gamma * dists)
                elif self.Qdist == 'gaussian':
                    preds = self.lsm(-self.gamma * dists.pow(2))
                elif self.Qdist == 'student':
                    preds = self.lsm(-torch.log(1 + self.gamma * dists))
                else:
                    raise NotImplementedError
            else:  # klSym or mse
                if self.Qdist == 'laplace':
                    preds = self.sm(-self.gamma * dists)
                elif self.Qdist == 'gaussian':
                    preds = self.sm(-self.gamma * dists.pow(2))
                elif self.Qdist == 'student':
                    preds = self.sm(-torch.log(1 + self.gamma * dists))
                else:
                    raise NotImplementedError

            # global distribution loss
            loss_global = self.lossfn(preds, target_ext)

            # local metric attraction loss
            d_local = self.dist().apply(
                new.unsqueeze(0).expand(1, k, self.dim),
                neighbor_embs.unsqueeze(0)
            ).squeeze()

            loss_local = (neighbor_w * d_local).sum()

            # total loss
            loss = loss_global + lambda_local * loss_local
            loss.backward()
            optimizer.step()

            # retract into Poincaré ball
            with torch.no_grad():
                norm = new.norm(p=2)
                if norm >= boundary:
                    new.mul_((boundary - 1e-8) / norm)

            losses.append(loss.item())

            if verbose and step % 25 == 0:
                print(
                    f"step {step:4d} | "
                    f"loss={loss.item():.3e} | "
                    f"global={loss_global.item():.3e} | "
                    f"local={loss_local.item():.3e} | "
                    f"||x||={norm.item():.3f}"
                )

        return new.detach().cpu().numpy().reshape(self.dim,), losses
    def _squared_norm(self, x, dim=-1, keepdim=True):
        return (x * x).sum(dim=dim, keepdim=keepdim)

    def _mobius_add(self, x, y, eps_small=1e-8):
        x2 = self._squared_norm(x, keepdim=True)
        y2 = self._squared_norm(y, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * xy + y2) * x + (1 - x2) * y
        den = 1 + 2 * xy + x2 * y2
        return num / (den + eps_small)

    def _mobius_neg(self, x):
        return -x

    def _artanh(self, x):
        x = x.clamp(min=-1 + 1e-6, max=1 - 1e-6)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def _lambda_x(self, x):
        norm2 = self._squared_norm(x, keepdim=True)
        return 2.0 / (1.0 - norm2 + 1e-8)

    def _log_map(self, x, y):
        # x: (1, dim) ; y: (k, dim)
        u = self._mobius_add(self._mobius_neg(x), y)
        norm_u = u.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        lam = self._lambda_x(x)
        coef = (2.0 / lam) * self._artanh(norm_u) / (norm_u + 1e-8)
        return coef * u

    def _exp_map(self, x, v):
        norm_v = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        lam = self._lambda_x(x)
        second = torch.tanh(lam * norm_v / 2.0) * (v / (norm_v + 1e-8))
        return self._mobius_add(x, second)

    def _project_to_ball(self, x, max_norm=1 - 1e-6):
        norm = x.norm(p=2, dim=-1, keepdim=True)
        mask = norm >= max_norm
        if mask.any():
            x = x / norm * max_norm
        return x

    def hyperbolic_barycenter(self, points, weights=None, n_steps=100, tol=1e-6, alpha=1.0, device=None, method='karcher', lr=1e-2):
        """
        Compute weighted Fréchet mean in the Poincaré ball using log/exp maps.
        points: (k, dim) tensor, weights: (k,) tensor or None.
        Returns x of shape (1, dim)
        """
        # device and tensor setup
        if device is None:
            device = points.device
        points = points.to(device)
        # ensure points are 2D tensor (k, dim)
        if points.ndim == 1:
            points = points.unsqueeze(0)
        k, dim = points.shape

        # Project input points into the Poincaré ball if some lie outside.
        # This avoids surprising behaviour when embeddings are loaded from CSV
        # without being projected.
        with torch.no_grad():
            norms = points.norm(p=2, dim=-1)
            max_norm = 1.0 - 1e-6
            if (norms >= max_norm).any():
                points = self._project_to_ball(points, max_norm=max_norm)

        # weights: ensure a valid normalized weight vector
        if weights is None:
            weights = torch.ones(k, device=device) / float(k)
        else:
            weights = weights.to(device).float()
            if weights.sum() <= 0:
                weights = torch.ones_like(weights) / float(k)
            else:
                weights = weights / weights.sum()

        # Fast path: if only one point, return it (after projection)
        if k == 1:
            return points[:1]

        # Initialization: Euclidean weighted mean projected into the Poincaré ball.
        # This provides a stable starting point close to the true barycenter.
        x = (weights.view(-1, 1) * points).sum(dim=0, keepdim=True)
        x = self._project_to_ball(x)

        # Option: alternate method via autograd optimization (often more robust)
        if method == 'optim':
            # minimize F(x) = sum_i w_i * d(x, y_i)^2 by optimizing a tangent vector z at 0
            x0 = torch.zeros((1, dim), device=device)
            # parameter in tangent space at 0
            z = torch.zeros((1, dim), device=device, requires_grad=True)
            optim = torch.optim.Adam([z], lr=lr)
            for i in range(n_steps):
                optim.zero_grad()
                x_curr = self._exp_map(x0, z)  # map to manifold
                x_rep = x_curr.unsqueeze(0).expand(1, k, dim)
                pts_rep = points.unsqueeze(0)
                d = self.dist().apply(x_rep, pts_rep).squeeze(0)
                loss = (weights * d.pow(2)).sum()
                loss.backward()
                optim.step()

                # guard: if x goes outside ball, project back and update z accordingly
                with torch.no_grad():
                    x_curr = self._project_to_ball(x_curr)
                    # update z to log_0(x_curr)
                    z_new = self._log_map(x0, x_curr).reshape_as(z)
                    z.data.copy_(z_new.data)
            x = self._exp_map(x0, z).detach()
            x = self._project_to_ball(x)
            return x

        # fall through to 'karcher' (log/exp iterative) method
        if method != 'karcher':
            raise ValueError("method must be 'karcher' or 'optim'")

        # Iteratively compute the Riemannian (Fréchet) mean by mapping points
        # to the tangent space at the current estimate `x` via the log map,
        # taking the weighted Euclidean mean in that tangent space, and
        # mapping back to the manifold with the exp map. This is a standard
        # and stable procedure for barycenter computation on manifolds.
        for i in range(n_steps):
            # map each point y_i to the tangent space at x: v_i = log_x(y_i)
            v = self._log_map(x, points)  # (k, dim)

            # weighted average in the tangent space
            v_bar = (weights.view(-1, 1) * v).sum(dim=0, keepdim=True)

            # check convergence: if the mean tangent vector is near zero, stop
            norm_vbar = v_bar.norm()
            if norm_vbar < tol:
                break

            # move along the averaged tangent vector back to the manifold
            x_new = self._exp_map(x, alpha * v_bar)

            # numerical guard: ensure we stay inside the ball
            x_new = self._project_to_ball(x_new)
            x = x_new

        return x

