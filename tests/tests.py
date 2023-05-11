import fastDTWF.fastDTWF as fastDTWF
import scipy.stats
import scipy.special
import torch
import numpy as np
import warnings


def test_coarse_grain_inner():
    for tv_sd in [1e-3, 1e-2, 1e-1, 0.25]:
        p_list = np.sort(np.random.random(1001))
        index_sets = fastDTWF._coarse_grain(
            p_list, 1000, tv_sd, False, False
        )
        for k in range(index_sets.max()+1):
            these_ps = p_list[index_sets == k]
            min_p = np.min(these_ps)
            max_p = np.max(these_ps)
            assert max_p <= min_p + tv_sd * np.sqrt(min_p*(1-min_p)/1002)
            assert min_p >= max_p - tv_sd * np.sqrt(max_p*(1-max_p)/1002)

        index_sets = fastDTWF._coarse_grain(
            p_list, 1000, tv_sd, True, False
        )
        assert (index_sets == index_sets[-1]).sum() == 1
        for k in range(index_sets.max()+1):
            these_ps = p_list[index_sets == k]
            min_p = np.min(these_ps)
            max_p = np.max(these_ps)
            assert max_p <= min_p + tv_sd * np.sqrt(min_p*(1-min_p)/1002)
            assert min_p >= max_p - tv_sd * np.sqrt(max_p*(1-max_p)/1002)

        index_sets = fastDTWF._coarse_grain(
            p_list, 1000, tv_sd, False, True
        )
        assert (index_sets == index_sets[-1]).sum() == 1
        assert (index_sets == index_sets[0]).sum() == 1
        for k in range(index_sets.max()+1):
            these_ps = p_list[index_sets == k]
            min_p = np.min(these_ps)
            max_p = np.max(these_ps)
            assert max_p <= min_p + tv_sd * np.sqrt(min_p*(1-min_p)/1002)
            assert min_p >= max_p - tv_sd * np.sqrt(max_p*(1-max_p)/1002)


def test_coarse_grain():
    for tv_sd in [1e-3, 1e-2, 1e-1, 0.25]:
        p_list = np.sort(np.random.random(1001))
        p_list_torch = torch.tensor(p_list, dtype=torch.float64)
        inner = fastDTWF._coarse_grain(
            p_list, 1000, tv_sd, False, False
        )
        outer = fastDTWF.coarse_grain(
            p_list_torch, 1000, tv_sd, False, False
        )
        assert np.all(inner == outer.detach().numpy())

        inner = fastDTWF._coarse_grain(
            p_list, 1000, tv_sd, True, False
        )
        outer = fastDTWF.coarse_grain(
            p_list_torch, 1000, tv_sd, True, False
        )
        assert np.all(inner == outer.detach().numpy())

        inner = fastDTWF._coarse_grain(
            p_list, 1000, tv_sd, False, True
        )
        outer = fastDTWF.coarse_grain(
            p_list_torch, 1000, tv_sd, False, True
        )
        assert np.all(inner == outer.detach().numpy())


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


def test_project_to_coarse():
    for _ in range(100):
        index_sets = np.random.randint(5, size=1000)
        vec = np.random.random(1000)
        sum_project = fastDTWF.project_to_coarse(
            torch.tensor(vec, dtype=torch.float64),
            torch.tensor(index_sets, dtype=torch.int64),
            None
        )
        true_sum = np.zeros(5)
        for i in range(5):
            true_sum[i] = vec[index_sets == i].sum()
        assert np.allclose(true_sum, sum_project.detach().numpy())

        normalizer = np.random.random(1000)
        norm_project = fastDTWF.project_to_coarse(
            torch.tensor(vec, dtype=torch.float64),
            torch.tensor(index_sets, dtype=torch.int64),
            torch.tensor(normalizer, dtype=torch.float64)
        )
        true_norm = np.zeros(5)
        for i in range(5):
            true_norm[i] = (
                vec[index_sets == i] * normalizer[index_sets == i]
            ).sum() / normalizer[index_sets == i].sum()
        assert np.allclose(true_norm, norm_project.detach().numpy())


def test_hypergeometric_sample():
    for _ in range(10):
        vec = np.random.random(500)
        vec /= vec.sum()
        check = fastDTWF.hypergeometric_sample(
            torch.tensor(vec, dtype=torch.float64),
            50,
            1e-16,
            1e-16,
            False
        )
        true = fastDTWF.naive_hypergeometric_sample(
            vec, 50
        )
        assert np.allclose(check.detach().numpy(), true)

        check = fastDTWF.hypergeometric_sample(
            torch.tensor(vec, dtype=torch.float64),
            50,
            0.05,
            1e-8,
            False
        )
        assert np.abs(check.detach().numpy() - true).sum() < 1e-4


def test_naive_hypergeometric_sample():
    for N in [51, 75, 100]:
        for K in [5, 10, 50]:
            dummy_vec = np.zeros(N+1)
            dummy_vec[K] = 1.
            check = fastDTWF.naive_hypergeometric_sample(
                dummy_vec, 25
            )
            true = scipy.stats.hypergeom.pmf(
                range(26), N, 25, K
            )
            assert np.allclose(check, true)


def test_make_condensed_matrix():
    v = torch.tensor(np.random.random(1001), dtype=torch.float64)
    v /= v.sum()
    zero_tensor = torch.tensor(0, dtype=torch.float64)
    p_list = fastDTWF.wright_fisher_ps_mutate_first(
        1000, zero_tensor, zero_tensor, zero_tensor
    )

    # First check standard process
    index_sets = fastDTWF.coarse_grain(p_list, 1000, 0.1, False, False)
    trunc_p_list = fastDTWF.project_to_coarse(p_list, index_sets, v)
    condensed_mat = fastDTWF.make_condensed_matrix(
        index_sets, trunc_p_list, 1000, 1e-8, False, False, 0.
    )
    v_proj = fastDTWF.project_to_coarse(v, index_sets)
    check = torch.matmul(condensed_mat.T, v_proj)
    true_1 = fastDTWF.mat_multiply(
        v, index_sets, p_list, 1000, 1e-8, False, False
    )
    true_1_proj = fastDTWF.project_to_coarse(true_1, index_sets)
    assert torch.allclose(check, true_1_proj)

    true_2 = fastDTWF.mat_multiply_from_coarse(
        v_proj, trunc_p_list, 1000, 1e-8, False, False, 0.
    )
    true_2_proj = fastDTWF.project_to_coarse(true_2, index_sets)
    assert torch.allclose(check, true_2_proj)

    # Now check conditioning on non-fixation
    v[-1] = 0.
    v /= v.sum()
    index_sets = fastDTWF.coarse_grain(p_list, 1000, 0.1, True, False)
    trunc_p_list = fastDTWF.project_to_coarse(p_list, index_sets, v)
    condensed_mat = fastDTWF.make_condensed_matrix(
        index_sets, trunc_p_list, 1000, 1e-8, True, False, 0.
    )
    v_proj = fastDTWF.project_to_coarse(v, index_sets)
    check = torch.matmul(condensed_mat.T, v_proj)
    check /= check.sum()
    true_1 = fastDTWF.mat_multiply(
        v, index_sets, p_list, 1000, 1e-8, True, False
    )
    true_1_proj = fastDTWF.project_to_coarse(true_1, index_sets)
    assert torch.allclose(check, true_1_proj)

    true_2 = fastDTWF.mat_multiply_from_coarse(
        v_proj, trunc_p_list, 1000, 1e-8, True, False, 0.
    )
    true_2_proj = fastDTWF.project_to_coarse(true_2, index_sets)
    assert torch.allclose(check, true_2_proj)

    # Now check SFS
    index_sets = fastDTWF.coarse_grain(p_list, 1000, 0.1, False, True)
    trunc_p_list = fastDTWF.project_to_coarse(p_list, index_sets, v)
    condensed_mat = fastDTWF.make_condensed_matrix(
        index_sets, trunc_p_list, 1000, 1e-8, False, True, 1.
    )
    v_proj = fastDTWF.project_to_coarse(v, index_sets)
    v_proj[0] = 1.
    check = torch.matmul(condensed_mat.T, v_proj)
    true_1 = fastDTWF.mat_multiply(
        v, index_sets, p_list, 1000, 1e-8, False, True, 1.
    )
    true_1_proj = fastDTWF.project_to_coarse(true_1, index_sets)
    assert torch.allclose(check[1:], true_1_proj[1:])

    true_2 = fastDTWF.mat_multiply_from_coarse(
        v_proj, trunc_p_list, 1000, 1e-8, False, True, 1.
    )
    true_2_proj = fastDTWF.project_to_coarse(true_2, index_sets)
    assert torch.allclose(check[1:], true_2_proj[1:])


# TODO
def test_diffusion_stationary_strong_s():
    pass


# TODO
def test_diffusion_sfs():
    pass


# TODO
def test_diffusion_stationary():
    pass


def test_naive_multiply():
    for N in [51, 75, 100]:
        for _ in range(5):
            p_list = np.sort(np.random.random(N+1))
            for i in range(N+1):
                vec = np.zeros(N+1)
                vec[i] = 1
                check = fastDTWF.naive_multiply(
                    vec, p_list, N, False, False, 0.0
                )
                with warnings.catch_warnings():
                    true = scipy.stats.binom.pmf(range(N+1), N, p_list[i])
                assert np.allclose(check, true)

                if i < N:
                    check = fastDTWF.naive_multiply(
                        vec, p_list, N, True, False, 0.0
                    )
                    with warnings.catch_warnings():
                        true = scipy.stats.binom.pmf(range(N+1), N, p_list[i])
                    true[-1] = 0
                    true /= true.sum()
                    assert np.allclose(check, true)

                if i > 0 and i < N:
                    check = fastDTWF.naive_multiply(
                        vec, p_list, N, False, True, 0.42
                    )
                    with warnings.catch_warnings():
                        true = scipy.stats.binom.pmf(range(N+1), N, p_list[i])
                    true[-1] = 0
                    true[0] = 0
                    true[1] += N * 0.42
                    assert np.allclose(check, true)


def test_make_binom_coefficients():
    for n in [5, 10, 50, 500]:
        true = np.log(scipy.special.binom(n, range(n+1)))
        check = fastDTWF._make_binom_coefficients(n)
        assert np.allclose(true, check.detach().numpy())


def test_mat_multiply_from_coarse():
    for _ in range(10):
        vec = np.random.random(501)
        vec /= vec.sum()
        p_list = np.sort(np.random.random(501))

        # standard, machine precision
        index_sets = fastDTWF.coarse_grain(
            torch.tensor(p_list, dtype=torch.float64),
            500,
            1e-16,
            False,
            False
        )
        trunc_mass = fastDTWF.project_to_coarse(
            torch.tensor(vec, dtype=torch.float64), index_sets
        )
        trunc_p_list = fastDTWF.project_to_coarse(
            torch.tensor(p_list, dtype=torch.float64),
            index_sets,
            torch.tensor(vec, dtype=torch.float64)
        )
        check = fastDTWF.mat_multiply_from_coarse(
            trunc_mass,
            trunc_p_list,
            500,
            1e-16,
            False,
            False,
            0.0,
            None
        )
        true = fastDTWF.naive_multiply(
            vec, p_list, 500, False, False, 0.0
        )
        assert np.allclose(true, check.detach().numpy())

        # standard, good approx
        index_sets = fastDTWF.coarse_grain(
            torch.tensor(p_list, dtype=torch.float64),
            500,
            0.1,
            False,
            False
        )
        assert index_sets.max() < 500
        trunc_mass = fastDTWF.project_to_coarse(
            torch.tensor(vec, dtype=torch.float64), index_sets
        )
        trunc_p_list = fastDTWF.project_to_coarse(
            torch.tensor(p_list, dtype=torch.float64),
            index_sets,
            torch.tensor(vec, dtype=torch.float64)
        )
        check = fastDTWF.mat_multiply_from_coarse(
            trunc_mass,
            trunc_p_list,
            500,
            1e-8,
            False,
            False,
            0.0,
            None
        )
        assert np.abs(true - check.detach().numpy()).sum() < 1e-3

        # non-fixation, exact
        index_sets = fastDTWF.coarse_grain(
            torch.tensor(p_list, dtype=torch.float64),
            100,
            1e-16,
            True,
            False
        )
        trunc_mass = fastDTWF.project_to_coarse(
            torch.tensor(vec, dtype=torch.float64), index_sets
        )
        trunc_p_list = fastDTWF.project_to_coarse(
            torch.tensor(p_list, dtype=torch.float64),
            index_sets,
            torch.tensor(vec, dtype=torch.float64)
        )
        check = fastDTWF.mat_multiply_from_coarse(
            trunc_mass,
            trunc_p_list,
            500,
            1e-16,
            True,
            False,
            0.0,
            None
        )
        true = fastDTWF.naive_multiply(
            vec, p_list, 500, True, False, 0.0
        )
        assert np.allclose(true, check.detach().numpy())

        # non-fixation, good approx
        index_sets = fastDTWF.coarse_grain(
            torch.tensor(p_list, dtype=torch.float64),
            500,
            0.1,
            True,
            False
        )
        trunc_mass = fastDTWF.project_to_coarse(
            torch.tensor(vec, dtype=torch.float64), index_sets
        )
        trunc_p_list = fastDTWF.project_to_coarse(
            torch.tensor(p_list, dtype=torch.float64),
            index_sets,
            torch.tensor(vec, dtype=torch.float64)
        )
        assert index_sets.max() < 500
        check = fastDTWF.mat_multiply_from_coarse(
            trunc_mass,
            trunc_p_list,
            500,
            1e-8,
            True,
            False,
            0.0,
            None
        )
        assert np.abs(true - check.detach().numpy()).sum() < 1e-3

        # SFS, exact
        vec[0] = 0
        vec[-1] = 0
        index_sets = fastDTWF.coarse_grain(
            torch.tensor(p_list, dtype=torch.float64),
            100,
            1e-16,
            False,
            True
        )
        trunc_mass = fastDTWF.project_to_coarse(
            torch.tensor(vec, dtype=torch.float64), index_sets
        )
        trunc_p_list = fastDTWF.project_to_coarse(
            torch.tensor(p_list, dtype=torch.float64),
            index_sets,
            torch.tensor(vec, dtype=torch.float64)
        )

        check = fastDTWF.mat_multiply_from_coarse(
            trunc_mass,
            trunc_p_list,
            500,
            1e-16,
            False,
            True,
            0.42,
            None
        )
        true = fastDTWF.naive_multiply(
            vec, p_list, 500, False, True, 0.42
        )
        assert np.allclose(true, check.detach().numpy())

        # SFS, good approx
        index_sets = fastDTWF.coarse_grain(
            torch.tensor(p_list, dtype=torch.float64),
            500,
            0.1,
            False,
            True
        )
        assert index_sets.max() < 500
        check = fastDTWF.mat_multiply(
            torch.tensor(vec, dtype=torch.float64),
            index_sets,
            torch.tensor(p_list, dtype=torch.float64),
            500,
            1e-8,
            False,
            True,
            0.42,
            None
        )
        assert np.abs(true - check.detach().numpy()).sum() < 1e-3


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


def test_mat_multiply():
    for _ in range(10):
        vec = np.random.random(501)
        vec /= vec.sum()
        p_list = np.sort(np.random.random(501))

        # standard, machine precision
        index_sets = fastDTWF.coarse_grain(
            torch.tensor(p_list, dtype=torch.float64),
            500,
            1e-16,
            False,
            False
        )
        check = fastDTWF.mat_multiply(
            torch.tensor(vec, dtype=torch.float64),
            index_sets,
            torch.tensor(p_list, dtype=torch.float64),
            500,
            1e-16,
            False,
            False,
            0.0,
            None
        )
        true = fastDTWF.naive_multiply(
            vec, p_list, 500, False, False, 0.0
        )
        assert np.allclose(true, check.detach().numpy())

        # standard, good approx
        index_sets = fastDTWF.coarse_grain(
            torch.tensor(p_list, dtype=torch.float64),
            500,
            0.1,
            False,
            False
        )
        assert index_sets.max() < 500
        check = fastDTWF.mat_multiply(
            torch.tensor(vec, dtype=torch.float64),
            index_sets,
            torch.tensor(p_list, dtype=torch.float64),
            500,
            1e-8,
            False,
            False,
            0.0,
            None
        )
        assert np.abs(true - check.detach().numpy()).sum() < 1e-3

        # non-fixation, exact
        index_sets = fastDTWF.coarse_grain(
            torch.tensor(p_list, dtype=torch.float64),
            100,
            1e-16,
            True,
            False
        )
        check = fastDTWF.mat_multiply(
            torch.tensor(vec, dtype=torch.float64),
            index_sets,
            torch.tensor(p_list, dtype=torch.float64),
            500,
            1e-16,
            True,
            False,
            0.0,
            None
        )
        true = fastDTWF.naive_multiply(
            vec, p_list, 500, True, False, 0.0
        )
        assert np.allclose(true, check.detach().numpy())

        # non-fixation, good approx
        index_sets = fastDTWF.coarse_grain(
            torch.tensor(p_list, dtype=torch.float64),
            500,
            0.1,
            True,
            False
        )
        assert index_sets.max() < 500
        check = fastDTWF.mat_multiply(
            torch.tensor(vec, dtype=torch.float64),
            index_sets,
            torch.tensor(p_list, dtype=torch.float64),
            500,
            1e-8,
            True,
            False,
            0.0,
            None
        )
        assert np.abs(true - check.detach().numpy()).sum() < 1e-3

        # SFS, exact
        index_sets = fastDTWF.coarse_grain(
            torch.tensor(p_list, dtype=torch.float64),
            100,
            1e-16,
            False,
            True
        )
        check = fastDTWF.mat_multiply(
            torch.tensor(vec, dtype=torch.float64),
            index_sets,
            torch.tensor(p_list, dtype=torch.float64),
            500,
            1e-16,
            False,
            True,
            0.42,
            None
        )
        true = fastDTWF.naive_multiply(
            vec, p_list, 500, False, True, 0.42
        )
        assert np.allclose(true, check.detach().numpy())

        # SFS, good approx
        index_sets = fastDTWF.coarse_grain(
            torch.tensor(p_list, dtype=torch.float64),
            500,
            0.1,
            False,
            True
        )
        assert index_sets.max() < 500
        check = fastDTWF.mat_multiply(
            torch.tensor(vec, dtype=torch.float64),
            index_sets,
            torch.tensor(p_list, dtype=torch.float64),
            500,
            1e-8,
            False,
            True,
            0.42,
            None
        )
        assert np.abs(true - check.detach().numpy()).sum() < 1e-3


# TODO
def test_integrate_likelihood_constant_size():
    pass


# TODO
def test_get_likelihood():
    pass
