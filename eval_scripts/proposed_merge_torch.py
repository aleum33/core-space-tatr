"""
Torch port of core_space_common.py's Stage 3 closed-form merge, for use
directly inside the real 8B-model pipeline (task_merger.py's
MatrixPerLayerMerger, merge_space='core'). CPU-numpy version already
validated in Stage 0-3 (see core_space_common.py / stage*.py) -- this is a
line-for-line translation, no new math.

Does NOT modify task_merger.py: callers get M_list/U_B_ref/Vh_A_ref from the
existing (untouched) get_core_matrices(), then recompute Bc_list/Ac_list
here (two lines, reusing U_B_ref/Vh_A_ref) and call proposed_merge_torch().
"""
import torch


def rank1_units_torch(bc_list, ac_list, r):
    units, owner = [], []
    for t, (bc, ac) in enumerate(zip(bc_list, ac_list)):
        for k in range(r):
            units.append(torch.outer(bc[:, k], ac[k, :]))
            owner.append(t)
    return units, owner


def right_orth_basis_torch(a):
    """Q = orth(A^T): (d, r) orthonormal columns spanning row(A), A is (r, d)."""
    _, _, vh = torch.linalg.svd(a, full_matrices=False)
    return vh.T


def core_input_subspaces_torch(ac_list):
    return [right_orth_basis_torch(ac) for ac in ac_list]


def build_G_C_torch(units, owner, q_list, num_tasks):
    device = units[0].device
    dtype = units[0].dtype
    n = len(units)
    flat_u = torch.stack([u.flatten() for u in units])  # (N, Tr*Tr)
    owner_t = torch.tensor(owner, device=device)

    gram_full = flat_u @ flat_u.T
    same_owner = (owner_t[:, None] == owner_t[None, :]).to(dtype)
    g = gram_full * same_owner

    c = torch.zeros((n, n), device=device, dtype=dtype)
    for t in range(num_tasks):
        pi_t = q_list[t] @ q_list[t].T  # (Tr, Tr) projector
        proj = torch.stack([(u @ pi_t).flatten() for u in units])  # (N, Tr*Tr)
        gram_t = proj @ proj.T
        mask_t = ((owner_t != t)[:, None] & (owner_t != t)[None, :]).to(dtype)
        c += gram_t * mask_t
    return g, c


def solve_phi_torch(g, c, mu):
    n = g.shape[0]
    if mu == 0:
        return torch.ones(n, device=g.device, dtype=g.dtype)
    a_mat = g + mu * c
    b_vec = g @ torch.ones(n, device=g.device, dtype=g.dtype)
    try:
        return torch.linalg.solve(a_mat, b_vec)
    except RuntimeError:
        return torch.linalg.lstsq(a_mat, b_vec.unsqueeze(-1)).solution.squeeze(-1)


def proposed_merge_torch(bc_list, ac_list, r, mu):
    """Returns M_merged (Tr, Tr) using the contamination-suppression closed form."""
    units, owner = rank1_units_torch(bc_list, ac_list, r)
    q_list = core_input_subspaces_torch(ac_list)
    g, c = build_G_C_torch(units, owner, q_list, len(bc_list))
    phi = solve_phi_torch(g, c, mu)
    m_merged = sum(phi[i] * units[i] for i in range(len(units)))
    return m_merged, phi


def load_location_cv_map(cv_json_path):
    """base_name (....lora_A.weight) -> CV -> re-keyed to key_base (....weight, no lora_A)."""
    import json
    with open(cv_json_path) as f:
        cv_result = json.load(f)
    per_location_cv = cv_result["per_location_cv"]
    out = {}
    suffix = ".lora_A.weight"
    for k, v in per_location_cv.items():
        assert k.endswith(suffix), f"unexpected key format: {k}"
        key_base = k[: -len(suffix)] + ".weight"
        out[key_base] = v
    return out


# ---------------------------------------------------------------------------
# Fast self-test (no model, no GPU needed strictly, but works on GPU too) --
# run this FIRST on the remote machine before touching the real 8B pipeline.
# Mirrors stage0_synthetic_sanity.py's checks exactly, in torch.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"self-test device: {device}")

    # (a) non-destructiveness at mu=0
    T, r, m, n = 4, 4, 32, 64
    a_list = [torch.randn(r, n, device=device, dtype=torch.float64) for _ in range(T)]
    b_list = [torch.randn(m, r, device=device, dtype=torch.float64) for _ in range(T)]

    a_cat = torch.vstack(a_list)
    b_cat = torch.hstack(b_list)
    tr = T * r
    u_ref = torch.linalg.svd(b_cat, full_matrices=False)[0][:, :tr]
    v_ref = torch.linalg.svd(a_cat.T, full_matrices=False)[0][:, :tr]
    bc_list = [u_ref.T @ b for b in b_list]
    ac_list = [a @ v_ref for a in a_list]
    m_list = [bc @ ac for bc, ac in zip(bc_list, ac_list)]

    # lossless check (fast projector-identity form)
    for a, b, bc, ac in zip(a_list, b_list, bc_list, ac_list):
        b_err = (u_ref @ bc - b).abs().max() / b.abs().max()
        a_err = (ac @ v_ref.T - a).abs().max() / a.abs().max()
        assert max(b_err, a_err) < 1e-8, f"lossless check failed: {b_err}, {a_err}"
    print("lossless reconstruction: PASS")

    m_merged, phi = proposed_merge_torch(bc_list, ac_list, r, mu=0.0)
    assert torch.allclose(phi, torch.ones_like(phi), atol=1e-10), f"phi != 1 at mu=0: max dev {(phi-1).abs().max()}"
    m_ta = sum(m_list)
    rel_err = (m_merged - m_ta).abs().max() / m_ta.abs().max()
    assert rel_err < 1e-8, f"M_merged != TA at mu=0: rel_err={rel_err}"
    print(f"non-destructiveness at mu=0: PASS (phi max dev={float((phi-1).abs().max()):.2e}, "
          f"M_merged rel err={float(rel_err):.2e})")

    # sanity: mu>0 actually changes something
    m_merged_mu1, phi_mu1 = proposed_merge_torch(bc_list, ac_list, r, mu=1.0)
    assert not torch.allclose(phi_mu1, torch.ones_like(phi_mu1)), "mu=1.0 had no effect -- suspicious"
    print(f"mu=1.0 sanity: phi std={float(phi_mu1.std()):.4f} (should be > 0)")

    print("\nALL SELF-TESTS PASSED")
