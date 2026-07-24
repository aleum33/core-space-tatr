"""
Shared, torch-free numpy implementation of the position-conditioned
interference-mitigation LoRA merge (Stages 1-3 of the experiment design).

Only needs numpy. No GPU, no torch, no PEFT -- everything here operates on
the small (r x n) / (m x r) LoRA factor matrices and the (Tr x Tr) core
matrices derived from them.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Generic subspace / CV utilities (shared with the earlier contamination
# diagnostic -- same definition, safe to reuse cached per_location_cv values).
# ---------------------------------------------------------------------------

def right_orth_basis(a):
    """Q = orth(A^T): orthonormal basis (cols) for the row space of A (r x d)."""
    _, _, vh = np.linalg.svd(a, full_matrices=False)  # vh: (r, d), rows orthonormal
    return vh.T  # (d, r)


def ratios_for_subspaces(q_list):
    """
    Given per-task orthonormal bases Q_t (d x r_t) living in a shared d-dim
    ambient space, return the flat list of ratio_m = con_m/own_m for every
    rank-1 unit (= basis column) of every task, plus the coefficient of
    variation across all of them.
    """
    all_ratios = []
    T = len(q_list)
    for own in range(T):
        q_own = q_list[own]
        r = q_own.shape[1]
        con = np.zeros(r)
        for s in range(T):
            if s == own:
                continue
            cross = q_list[s].T @ q_own  # (r_s, r_own)
            con += np.linalg.norm(cross, axis=0)
        own_norm = np.ones(r)  # exact by construction, see contamination_cv_diagnostic.py
        all_ratios.extend((con / own_norm).tolist())
    return all_ratios


def coefficient_of_variation(values):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean()
    return values.std() / mean if mean != 0 else float("nan")


# ---------------------------------------------------------------------------
# Stage 1: Core Space construction (paper Eq. 3 / 8)
# ---------------------------------------------------------------------------

def build_core_space(a_list, b_list):
    """
    a_list: list of T arrays, each (r, n)   -- LoRA A per task
    b_list: list of T arrays, each (m, r)   -- LoRA B per task
    Returns dict with U_ref (m,Tr), V_ref (n,Tr), Bc_list/Ac_list/M_list (per task).
    """
    T = len(a_list)
    r = a_list[0].shape[0]
    tr = T * r

    a_cat = np.vstack(a_list)  # (Tr, n)
    b_cat = np.hstack(b_list)  # (m, Tr)

    u_ref = np.linalg.svd(b_cat, full_matrices=False)[0][:, :tr]       # (m, Tr)
    v_ref = np.linalg.svd(a_cat.T, full_matrices=False)[0][:, :tr]     # (n, Tr)

    bc_list = [u_ref.T @ b for b in b_list]   # each (Tr, r)
    ac_list = [a @ v_ref for a in a_list]      # each (r, Tr)
    m_list = [bc @ ac for bc, ac in zip(bc_list, ac_list)]  # each (Tr, Tr)

    return {
        "U_ref": u_ref, "V_ref": v_ref,
        "Bc_list": bc_list, "Ac_list": ac_list, "M_list": m_list,
        "T": T, "r": r, "Tr": tr,
    }


def check_lossless(a_list, b_list, core, rtol=1e-4):
    """
    Per-task reconstruction error: ||U_ref M_t V_ref^T - B_t A_t||_inf / ||B_t A_t||_inf.
    Materializes the full (m x n) delta -- fine for small synthetic shapes,
    too slow for real (m,n) up to 14336. See check_lossless_fast for that.
    """
    errs = []
    u_ref, v_ref = core["U_ref"], core["V_ref"]
    for a, b, m in zip(a_list, b_list, core["M_list"]):
        recon = u_ref @ m @ v_ref.T
        orig = b @ a
        denom = np.max(np.abs(orig))
        err = np.max(np.abs(recon - orig)) / denom if denom > 0 else 0.0
        errs.append(err)
    return errs


def check_lossless_fast(a_list, b_list, core):
    """
    Algebraically equivalent to check_lossless but never materializes the
    (m x n) delta matrix. Since B_t's columns lie exactly in span(U_ref) and
    A_t's rows lie exactly in span(V_ref^T) (both are literal blocks of the
    stacks U_ref/V_ref were built from):

        U_ref M_t V_ref^T - B_t A_t
          = (U_ref U_ref^T) B_t A_t (V_ref V_ref^T) - B_t A_t
          = 0   exactly, in exact arithmetic

    so the only real signal is floating-point round-off in the two
    projections, captured on the much smaller (m,r) / (r,n) factors:
        ||U_ref @ Bc_t - B_t||_inf / ||B_t||_inf   (output-side projector)
        ||Ac_t @ V_ref^T - A_t||_inf / ||A_t||_inf   (input-side projector)
    Cost: O(m*Tr*r + n*Tr*r) instead of O(m*Tr*n) -- ~n/r times cheaper.
    """
    errs = []
    u_ref, v_ref = core["U_ref"], core["V_ref"]
    for a, b, bc, ac in zip(a_list, b_list, core["Bc_list"], core["Ac_list"]):
        b_recon = u_ref @ bc
        b_err = np.max(np.abs(b_recon - b)) / np.max(np.abs(b))
        a_recon = ac @ v_ref.T
        a_err = np.max(np.abs(a_recon - a)) / np.max(np.abs(a))
        errs.append(max(b_err, a_err))
    return errs


# ---------------------------------------------------------------------------
# Stage 3.1: rank-1 unit decomposition + per-task core-space input subspaces
# ---------------------------------------------------------------------------

def rank1_units(core):
    """units[i] (Tr,Tr) rank-1 matrices, owner[i] = task index."""
    units, owner = [], []
    for t in range(core["T"]):
        bc, ac = core["Bc_list"][t], core["Ac_list"][t]
        for k in range(core["r"]):
            units.append(np.outer(bc[:, k], ac[k, :]))
            owner.append(t)
    return units, owner


def core_input_subspaces(core):
    """Q_t = orth(Ac_t^T), shape (Tr, r), one per task -- data-free, gauge invariant."""
    return [right_orth_basis(ac) for ac in core["Ac_list"]]


# ---------------------------------------------------------------------------
# Stage 3.2/3.3: objective + closed-form solve
# ---------------------------------------------------------------------------

def build_G_C(units, owner, q_list, num_tasks):
    n = len(units)
    flat_u = np.stack([u.flatten() for u in units])  # (N, Tr*Tr)
    owner_arr = np.array(owner)

    gram_full = flat_u @ flat_u.T
    g = gram_full * (owner_arr[:, None] == owner_arr[None, :])

    c = np.zeros((n, n))
    for t in range(num_tasks):
        pi_t = q_list[t] @ q_list[t].T  # (Tr, Tr) projector
        proj = np.stack([(u @ pi_t).flatten() for u in units])  # (N, Tr*Tr)
        gram_t = proj @ proj.T
        mask_t = (owner_arr != t)[:, None] & (owner_arr != t)[None, :]
        c += gram_t * mask_t
    return g, c


def solve_phi(g, c, mu):
    n = g.shape[0]
    if mu == 0:
        # (G)(phi-1) = 0 is solved exactly and uniquely-intended by phi=1:
        # this is the non-destructiveness safety net, enforced directly
        # rather than via linear solve (G may be singular).
        return np.ones(n)
    a_mat = g + mu * c
    b_vec = g @ np.ones(n)
    try:
        return np.linalg.solve(a_mat, b_vec)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a_mat, b_vec, rcond=None)[0]


def proposed_merge(core, mu):
    units, owner = rank1_units(core)
    q_list = core_input_subspaces(core)
    g, c = build_G_C(units, owner, q_list, core["T"])
    phi = solve_phi(g, c, mu)
    m_merged = sum(phi[i] * units[i] for i in range(len(units)))
    return m_merged, phi, owner


# ---------------------------------------------------------------------------
# Baselines, operating directly on the (Tr x Tr) core matrices M_t
# ---------------------------------------------------------------------------

def ta_merge(m_list):
    return sum(m_list)


def ties_merge_np(m_list, keep_frac=0.2):
    shape = m_list[0].shape
    flat = np.stack([m.flatten() for m in m_list])  # (T, D)
    t, d = flat.shape
    k = max(1, int(d * keep_frac))
    order = np.argsort(-np.abs(flat), axis=1)
    mask = np.zeros_like(flat, dtype=bool)
    rows = np.arange(t)[:, None]
    mask[rows, order[:, :k]] = True
    trimmed = flat * mask

    sign = np.sign(trimmed.sum(axis=0))
    sign[sign == 0] = 1
    agree = np.where(sign[None, :] > 0, trimmed > 0, trimmed < 0)
    selected = trimmed * agree
    counts = np.clip(agree.sum(axis=0), 1, None)
    merged = selected.sum(axis=0) / counts
    return merged.reshape(shape)


def dare_ties_merge_np(m_list, drop_p=0.3, keep_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    dared = []
    for m in m_list:
        mask = rng.random(m.shape) > drop_p
        dared.append(m * mask / (1 - drop_p))
    return ties_merge_np(dared, keep_frac)


def isoc_merge_np(m_list):
    summed = sum(m_list)
    u, s, vh = np.linalg.svd(summed, full_matrices=False)
    s_iso = np.full_like(s, s.mean())
    return u @ np.diag(s_iso) @ vh
