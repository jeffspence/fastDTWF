import fastDTWF.fastDTWF as fastDTWF
import scipy.stats
import scipy.special
import torch
import numpy as np
import warnings


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


def test_numba_hyp_pmf_single():
    check = fastDTWF._numba_hyp_pmf_single(3, 100, 45, 17)
    true = scipy.stats.hypergeom.pmf(3, 100, 17, 45)
    assert np.isclose(true, check)

    for _ in range(100):
        N = np.random.randint(10, 1000)
        K = np.random.randint(1, N)
        n = np.random.randint(1, N)
        k = np.random.randint(n)
        check = fastDTWF._numba_hyp_pmf_single(k, N, K, n)
        true = scipy.stats.hypergeom.pmf(k, N, n, K)
        assert np.isclose(true, check)


def test_numba_hyp_pmf():
    check = fastDTWF._numba_hyp_pmf(3, 10, 100, 17, 45)
    true = scipy.stats.hypergeom.pmf(range(3, 11), 100, 45, 17)
    assert np.allclose(true, check)

    for _ in range(100):
        N = np.random.randint(10, 1000)
        K = np.random.randint(1, N)
        n = np.random.randint(2, N)
        kmax = np.random.randint(1, n)
        kmin = np.random.randint(kmax)
        check = fastDTWF._numba_hyp_pmf(kmin, kmax, N, K, n)
        true = scipy.stats.hypergeom.pmf(range(kmin, kmax+1), N, n, K)
        assert np.allclose(true, check)


def test_torch_hyp_pmf():
    lfacts = torch.lgamma(torch.arange(1001, dtype=torch.float64) + 1)
    for _ in range(100):
        N = np.random.randint(10, 1000)
        K = np.random.randint(1, N)
        n = np.random.randint(2, N)
        kmax = np.random.randint(1, n)
        kmin = np.random.randint(kmax)
        check = fastDTWF._torch_hyp_pmf(kmin, kmax, N, K, n, lfacts)
        true = fastDTWF._numba_hyp_pmf(kmin, kmax, N, K, n)
        assert np.allclose(true, check.detach().numpy())


def test_numba_binom_pmf_single():
    assert fastDTWF._numba_binom_pmf_single(0, 55, 0) == 1.
    assert fastDTWF._numba_binom_pmf_single(1, 55, 0) == 0.
    assert fastDTWF._numba_binom_pmf_single(2, 55, 0) == 0.
    assert fastDTWF._numba_binom_pmf_single(55, 55, 0) == 0.
    assert fastDTWF._numba_binom_pmf_single(0, 55, 1) == 0.
    assert fastDTWF._numba_binom_pmf_single(1, 55, 1) == 0.
    assert fastDTWF._numba_binom_pmf_single(54, 55, 1) == 0.
    assert fastDTWF._numba_binom_pmf_single(55, 55, 1) == 1.

    for _ in range(100):
        p = np.random.random()
        N = np.random.randint(1, 1000)
        k = np.random.randint(N)
        # scipy.stats.binom.pmf throws a harmless division by zero runtime
        # warning
        with warnings.catch_warnings():
            assert np.isclose(
                fastDTWF._numba_binom_pmf_single(k, N, p),
                scipy.stats.binom.pmf(k, N, p)
            )
            assert np.isclose(
                fastDTWF._numba_binom_pmf_single(int(p*N), N, p),
                scipy.stats.binom.pmf(int(p*N), N, p)
            )


def test_numba_binom_pmf():
    check = fastDTWF._numba_binom_pmf(3, 10, 100, 0.1)
    true = scipy.stats.binom.pmf(range(3, 11), 100, 0.1)
    assert np.allclose(true, check)

    for _ in range(100):
        N = np.random.randint(10, 1000)
        p = np.random.random()
        kmax = np.random.randint(1, N)
        kmin = np.random.randint(kmax)
        check = fastDTWF._numba_binom_pmf(kmin, kmax, N, p)
        # scipy.stats.binom.pmf throws a harmless division by zero runtime
        # warning
        with warnings.catch_warnings():
            true = scipy.stats.binom.pmf(range(kmin, kmax+1), N, p)
        assert np.allclose(true, check)


def test_torch_binom_pmf():
    for _ in range(100):
        N = np.random.randint(10, 1000)
        bcoefs = fastDTWF._make_binom_coefficients(N)
        p = np.random.random()
        kmax = np.random.randint(1, N)
        kmin = np.random.randint(kmax)
        check = fastDTWF._torch_binom_pmf(
            kmin,
            kmax,
            N,
            torch.tensor(p, dtype=torch.float64),
            bcoefs
        )
        # scipy.stats.binom.pmf throws a harmless division by zero runtime
        # warning
        with warnings.catch_warnings():
            true = scipy.stats.binom.pmf(range(kmin, kmax+1), N, p)
        assert np.allclose(true, check.detach().numpy())


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


def test_make_binom_coefficients():
    for n in [5, 10, 50, 500]:
        true = np.log(scipy.special.binom(n, range(n+1)))
        check = fastDTWF._make_binom_coefficients(n)
        assert np.allclose(true, check.detach().numpy())


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
