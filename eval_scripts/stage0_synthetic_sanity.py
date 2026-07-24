"""
Stage 0 -- synthetic sanity checks. No checkpoints, no GPU. Must pass before
touching real checkpoints (Stage 1).

(a) Non-destructiveness: at mu=0, phi* must be exactly 1 and M_merged must
    exactly equal Task Arithmetic (sum of M_t). Verified two ways: via the
    software shortcut (solve_phi's mu==0 special case) AND by forcing the
    general linear solve at a vanishingly small mu, to confirm the *algebra*
    itself (not just the shortcut) collapses to phi=1.
(b) Gate discriminability: build a synthetic "high-CV" location (two tasks
    deliberately share input directions) and a synthetic "low-CV" location
    (all tasks' input directions independent random) and confirm the CV gate
    at tau=0.4 fires True only on the high-CV one.
"""
import numpy as np

from core_space_common import (
    build_core_space, check_lossless, proposed_merge, ta_merge,
    right_orth_basis, ratios_for_subspaces, coefficient_of_variation,
)

RNG = np.random.default_rng(0)


def make_synthetic_location(t=4, r=4, m=32, n=64):
    a_list = [RNG.standard_normal((r, n)) for _ in range(t)]
    b_list = [RNG.standard_normal((m, r)) for _ in range(t)]
    return a_list, b_list


def check_non_destructiveness():
    a_list, b_list = make_synthetic_location()
    core = build_core_space(a_list, b_list)

    errs = check_lossless(a_list, b_list, core)
    assert max(errs) < 1e-4, f"Stage 1 lossless condition failed in synthetic setup: {errs}"

    # (a1) software shortcut path
    m_merged, phi, _ = proposed_merge(core, mu=0.0)
    assert np.allclose(phi, 1.0, atol=1e-10), f"phi* != 1 at mu=0 (shortcut path): max dev {np.abs(phi-1).max()}"
    ta = ta_merge(core["M_list"])
    rel_err = np.max(np.abs(m_merged - ta)) / np.max(np.abs(ta))
    assert rel_err < 1e-8, f"M_merged != Task Arithmetic at mu=0 (shortcut path): rel_err={rel_err}"

    # (a2) force the general linear solve (bypass the mu==0 shortcut) at a
    # vanishingly small mu, to confirm the underlying closed form itself
    # (not just the software special-case) collapses to phi=1.
    from core_space_common import rank1_units, core_input_subspaces, build_G_C
    units, owner = rank1_units(core)
    q_list = core_input_subspaces(core)
    g, c = build_G_C(units, owner, q_list, core["T"])
    mu_tiny = 1e-9
    a_mat = g + mu_tiny * c
    b_vec = g @ np.ones(g.shape[0])
    phi_general = np.linalg.lstsq(a_mat, b_vec, rcond=None)[0]
    assert np.allclose(phi_general, 1.0, atol=1e-4), \
        f"phi* != 1 at mu->0 (general solve path): max dev {np.abs(phi_general-1).max()}"

    print("(a) non-destructiveness: PASS")
    print(f"    lossless recon errs: max={max(errs):.2e}")
    print(f"    phi* max deviation from 1 (shortcut path): {np.abs(phi-1).max():.2e}")
    print(f"    M_merged vs Task Arithmetic rel err (shortcut path): {rel_err:.2e}")
    print(f"    phi* max deviation from 1 (general solve, mu={mu_tiny}): {np.abs(phi_general-1).max():.2e}")


def check_gate_discriminability(tau=0.4):
    # T=6 mirrors the real 6-NLI-task setup; large n so independent random
    # r-dim subspaces concentrate tightly around a small, uniform overlap
    # (low CV) instead of just "lower than the other case".
    t, r, n = 6, 8, 512

    # low-CV: fully independent random subspaces for every task
    a_low = [RNG.standard_normal((r, n)) for _ in range(t)]
    q_low = [right_orth_basis(a) for a in a_low]
    ratios_low = ratios_for_subspaces(q_low)
    cv_low = coefficient_of_variation(ratios_low)

    # high-CV: 3 of the 6 tasks share an IDENTICAL input subspace (full
    # r-dim overlap -> con_m saturates for that group's units), the other 3
    # stay fully independent (near-zero contamination) -> sharp bimodal
    # ratio distribution -> high CV, unlike the smoothly-random case above.
    shared_basis = right_orth_basis(RNG.standard_normal((r, n)))  # (n, r)
    a_high = []
    for i in range(t):
        if i < 3:
            # same subspace, different (random invertible) coordinates
            mix = RNG.standard_normal((r, r))
            a_high.append(mix @ shared_basis.T)
        else:
            a_high.append(RNG.standard_normal((r, n)))
    q_high = [right_orth_basis(a) for a in a_high]
    ratios_high = ratios_for_subspaces(q_high)
    cv_high = coefficient_of_variation(ratios_high)

    gated_low = cv_low > tau
    gated_high = cv_high > tau

    assert cv_high > cv_low, f"expected high-CV location to exceed low-CV: {cv_high} vs {cv_low}"
    assert gated_high and not gated_low, \
        f"gate did not discriminate at tau={tau}: gated_low={gated_low} (CV={cv_low:.3f}), " \
        f"gated_high={gated_high} (CV={cv_high:.3f})"

    print("(b) gate discriminability: PASS")
    print(f"    low-CV synthetic location:  CV={cv_low:.3f}  gated={gated_low}")
    print(f"    high-CV synthetic location: CV={cv_high:.3f}  gated={gated_high}")


if __name__ == "__main__":
    check_non_destructiveness()
    print()
    check_gate_discriminability()
