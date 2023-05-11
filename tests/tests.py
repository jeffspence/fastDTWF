import fastDTWF
import scipy.stats
import torch


# TODO
def test_coarse_grain():
    pass


def test_wright_fisher_ps_mutate_first():
    zero_tensor = torch.tensor(0, dtype=torch.float64)
    test_ps = fastDTWF.wright_fisher_ps_mutate_first(
        1000, zero_tensor, zero_tensor, zero_tensor
    )
    freqs = torch.arange(1001, dtype=torch.float64) / 1000
    assert torch.allclose(test_ps, freqs)

    one_tensor = torch.tensor(1, dtype=torch.float64)
    lower_ps = fastDTWF.wright_fisher_ps_mutate_first(
        1000, zero_tensor, zero_tensor, one_tensor*1e-5
    )
    assert torch.all(test_ps >= lower_ps)

    lowest_ps = fastDTWF.wright_fisher_ps_mutate_first(
        1000, zero_tensor, zero_tensor, one_tensor*1e-5, one_tensor*3e-5
    )
    assert torch.all(lower_ps >= lowest_ps)

    weird_ps = fastDTWF.wright_fisher_ps_mutate_first(
        1000, zero_tensor, zero_tensor, one_tensor, zero_tensor
    )
    assert torch.allclose(weird_ps, freqs**2 / (1 - 2*freqs + 2*freqs**2))

    mut_ps = fastDTWF.wright_fisher_ps_mutate_first(
        1000, one_tensor, zero_tensor, zero_tensor, zero_tensor
    )
    assert torch.all(mut_ps == 1)

    zero_ps = fastDTWF.wright_fisher_ps_mutate_first(
        1000, zero_tensor, one_tensor, zero_tensor, zero_tensor
    )
    assert torch.all(zero_ps == 0)

    nudge_ps = fastDTWF.wright_fisher_ps_mutate_first(
        1000, one_tensor*1e-5, zero_tensor, zero_tensor
    )
    assert torch.allclose(nudge_ps, 1e-5 + (1-1e-5)*freqs)

    other_nudge_ps = fastDTWF.wright_fisher_ps_mutate_first(
        1000, zero_tensor, one_tensor*1e-5, zero_tensor
    )
    assert torch.allclose(other_nudge_ps, (1-1e-5)*freqs)


# TODO
def test_numba_hyp_pmf_single():
    pass


# TODO
def test_numba_hyp_pmf():
    pass


# TODO
def test_torch_hyp_pmf():
    pass


# TODO
def test_numba_binom_pmf_single():
    pass


# TODO
def test_numba_binom_pmf():
    pass


# TODO
def test_torch_binom_pmf():
    pass


# TODO
def test_project_to_coarse():
    pass


# TODO
def test_hypergeometric_sample():
    pass


# TODO
def test_naive_hypergeometric_sample():
    pass


# TODO
def test_make_condensed_matrix():
    pass


# TODO
def test_diffusion_stationary_strong_s():
    pass


# TODO
def test_diffusion_sfs():
    pass


# TODO
def test_diffusion_stationary():
    pass


# TODO
def test_naive_multiply():
    pass


# TODO
def test_make_binom_coefficients():
    pass


# TODO
def test_mat_multiply_from_coarse():
    pass


# TODO
def test_naive_equilibrium_solve():
    pass


# TODO
def test_equilibrium_solve():
    pass


# TODO
def test_pick_eigenvec():
    pass


# TODO
def test_one_step_solve():
    pass


# TODO
def test_mat_multiply():
    pass


# TODO
def test_integrate_likelihood_constant_size():
    pass


# TODO
def test_get_likelihood():
    pass
