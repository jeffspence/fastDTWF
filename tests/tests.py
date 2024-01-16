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


def test_wright_fisher_ps_mutate_first_x_chr():
    zero_tensor = torch.tensor(0, dtype=torch.float64)
    test_ps = fastDTWF.wright_fisher_ps_mutate_first_x_chr(
        1000, zero_tensor, zero_tensor, zero_tensor, zero_tensor, zero_tensor
    )
    freqs = torch.arange(1001, dtype=torch.float64) / 1000
    assert torch.allclose(test_ps, freqs)

    one_tensor = torch.tensor(1, dtype=torch.float64)
    test_ps = fastDTWF.wright_fisher_ps_mutate_first_x_chr(
        1000,
        one_tensor*1e-5,
        one_tensor*1e-5,
        one_tensor*1e-5,
        one_tensor*2e-5,
        one_tensor*1e-5
    )
    assert torch.all(test_ps >= zero_tensor)
    assert torch.all(test_ps <= one_tensor)

    nudge_1 = fastDTWF.wright_fisher_ps_mutate_first_x_chr(
        1000,
        one_tensor*1e-5 - 1e-7,
        one_tensor*1e-5,
        one_tensor*1e-5,
        one_tensor*2e-5,
        one_tensor*1e-5
    )
    nudge_2 = fastDTWF.wright_fisher_ps_mutate_first_x_chr(
        1000,
        one_tensor*1e-5,
        one_tensor*1e-5 + 1e-7,
        one_tensor*1e-5,
        one_tensor*2e-5,
        one_tensor*1e-5
    )
    nudge_3 = fastDTWF.wright_fisher_ps_mutate_first_x_chr(
        1000,
        one_tensor*1e-5,
        one_tensor*1e-5,
        one_tensor*1e-5 + 1e-7,
        one_tensor*2e-5 + 2e-7,
        one_tensor*1e-5
    )
    nudge_4 = fastDTWF.wright_fisher_ps_mutate_first_x_chr(
        1000,
        one_tensor*1e-5,
        one_tensor*1e-5,
        one_tensor*1e-5,
        one_tensor*2e-5 + 1e-7,
        one_tensor*1e-5
    )
    nudge_5 = fastDTWF.wright_fisher_ps_mutate_first_x_chr(
        1000,
        one_tensor*1e-5,
        one_tensor*1e-5,
        one_tensor*1e-5,
        one_tensor*2e-5,
        one_tensor*1e-5 + 1e-7
    )

    assert torch.all(nudge_1 <= test_ps)
    assert torch.all(nudge_2 <= test_ps)
    assert torch.all(nudge_3 <= test_ps), torch.max(nudge_3 - test_ps)
    assert torch.all(nudge_4 <= test_ps)
    assert torch.all(nudge_5 <= test_ps)

    recessive = fastDTWF.wright_fisher_ps_mutate_first_x_chr(
        1000,
        one_tensor*1e-3,
        one_tensor*1e-3,
        zero_tensor,
        one_tensor*1e-5,
        one_tensor*1e-5
    )
    true_recessive = fastDTWF.wright_fisher_ps_mutate_first(
        1000,
        one_tensor*1e-3,
        one_tensor*1e-3,
        zero_tensor,
        one_tensor*1e-5
    )

    assert torch.allclose(recessive, true_recessive)


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
            warnings.simplefilter("ignore")
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
            warnings.simplefilter("ignore")
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
            warnings.simplefilter("ignore")
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

    v_proj[0] = 0.
    true_2 = fastDTWF.mat_multiply_from_coarse(
        v_proj, trunc_p_list, 1000, 1e-8, False, True, 1.
    )
    true_2_proj = fastDTWF.project_to_coarse(true_2, index_sets)
    assert torch.allclose(check[1:], true_2_proj[1:])


def test_diffusion_stationary_strong_s():
    for s_het in [0.005, 0.01, 0.05, 0.1]:
        check = fastDTWF.diffusion_stationary_strong_s(
            1000,
            torch.tensor(s_het, dtype=torch.float64),
            torch.tensor(1e-10, dtype=torch.float64)
        ).detach().numpy()[1:10]
        check /= check.sum()
        sfs = fastDTWF.diffusion_sfs(1000, s_het)[1:10]
        sfs /= sfs.sum()
        assert np.all(np.abs(sfs - check) / sfs <= 1e-2)


def test_diffusion_stationary():
    # Low mutation rate limit should roughly match SFS
    for s_het in [0.000001, 0.00001, 0.0001, 0.001]:
        check = fastDTWF.diffusion_stationary(
            1000,
            torch.tensor(s_het, dtype=torch.float64),
            torch.tensor(1e-10, dtype=torch.float64),
            torch.tensor(1e-10, dtype=torch.float64)
        ).detach().numpy()[1:10]
        check /= check.sum()
        sfs = fastDTWF.diffusion_sfs(1000, s_het)[1:10]
        sfs /= sfs.sum()
        assert np.all(np.abs(sfs - check) / sfs <= 1e-2)

    # Should just be beta binomial, can work out by hand
    for _ in range(10):
        alpha = np.random.random() * 1e-10
        beta = np.random.random() * 1e-10
        check = fastDTWF.diffusion_stationary(
            1000,
            torch.tensor(1e-30, dtype=torch.float64),
            torch.tensor(alpha, dtype=torch.float64),
            torch.tensor(beta, dtype=torch.float64),
        ).detach().numpy()
        assert np.isclose(check[0], 1./(1. + alpha/beta))
        assert np.isclose(check[-1], 1./(beta/alpha + 1.))
        interior = np.arange(1, 1000)
        interior = 1000 / ((1000 - interior) * interior)
        interior /= (1/(2*1000*alpha) + 1/(2*1000*beta))
        assert np.allclose(check[1:-1], interior)


def test_diffusion_sfs():
    # Compare to analytical neutral result
    check_neutral = fastDTWF.diffusion_sfs(100, 0)
    check_nearly_neutral = fastDTWF.diffusion_sfs(100, 1e-30)
    true = 1 / (1e-100 + np.arange(101))
    true[0] = 0
    true[-1] = 0
    true /= true.sum()
    assert np.allclose(check_neutral, true)
    assert np.allclose(check_nearly_neutral, true)

    # Make sure that selection shifts you to lower frequencies
    curr_sfs = check_nearly_neutral
    for s_het in 10**np.linspace(-10, -1, num=10):
        new_sfs = fastDTWF.diffusion_sfs(100, s_het)
        curr_cumsum = curr_sfs.cumsum()
        assert np.all(curr_cumsum <= new_sfs.cumsum() + 1e-5*curr_cumsum)
        curr_sfs = new_sfs


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
                    warnings.simplefilter("ignore")
                    true = scipy.stats.binom.pmf(range(N+1), N, p_list[i])
                assert np.allclose(check, true)

                if i < N:
                    check = fastDTWF.naive_multiply(
                        vec, p_list, N, True, False, 0.0
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        true = scipy.stats.binom.pmf(range(N+1), N, p_list[i])
                    true[-1] = 0
                    true /= true.sum()
                    assert np.allclose(check, true)

                if i > 0 and i < N:
                    check = fastDTWF.naive_multiply(
                        vec, p_list, N, False, True, 0.42
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
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


def test_naive_equilibrium_solve():
    for _ in range(10):
        mu_1 = np.random.random() * 1e-8
        mu_2 = np.random.random() * 1e-8
        s_het = np.random.random() * 1e-3
        ps = fastDTWF.wright_fisher_ps_mutate_first(
            1000,
            torch.tensor(mu_1, dtype=torch.float64),
            torch.tensor(mu_2, dtype=torch.float64),
            torch.tensor(s_het, dtype=torch.float64)
        ).detach().numpy()
        eq_check = fastDTWF.naive_equilibrium_solve(
            ps,
            1000,
            False,
            False
        )
        next_gen = fastDTWF.naive_multiply(
            eq_check,
            ps,
            1000,
            False,
            False,
            0.
        )
        assert np.allclose(eq_check, next_gen)
        eq_check = fastDTWF.naive_equilibrium_solve(
            ps,
            1000,
            True,
            False,
            0.
        )
        next_gen = fastDTWF.naive_multiply(
            eq_check,
            ps,
            1000,
            True,
            False,
            0.
        )
        assert np.allclose(eq_check[1:], next_gen[1:])
        eq_check = fastDTWF.naive_equilibrium_solve(
            ps,
            1000,
            False,
            True,
            0.42
        )
        next_gen = fastDTWF.naive_multiply(
            eq_check,
            ps,
            1000,
            False,
            True,
            0.42
        )
        assert np.allclose(eq_check[1:], next_gen[1:])


def test_equilibrium_solve():
    for _ in range(10):
        mu_1 = np.random.random() * 1e-8
        mu_2 = np.random.random() * 1e-8
        s_het = np.random.random() * 1e-3
        ps = fastDTWF.wright_fisher_ps_mutate_first(
            1000,
            torch.tensor(mu_1, dtype=torch.float64),
            torch.tensor(mu_2, dtype=torch.float64),
            torch.tensor(s_het, dtype=torch.float64)
        )
        index_sets = fastDTWF.coarse_grain(
            ps, 1000, 0.1, False, False
        )
        eq_check = fastDTWF.equilibrium_solve(
            index_sets,
            ps,
            1000,
            1e-8,
            False,
            False
        )
        next_gen = fastDTWF.mat_multiply(
            eq_check,
            index_sets,
            ps,
            1000,
            1e-8,
            False,
            False,
            0.
        )
        assert np.allclose(eq_check, next_gen)

        eq_check = fastDTWF.equilibrium_solve(
            index_sets,
            ps,
            1000,
            1e-8,
            False,
            False,
            None,
            True
        )
        next_gen = fastDTWF.mat_multiply(
            eq_check,
            index_sets,
            ps,
            1000,
            1e-8,
            False,
            False,
            0.
        )
        assert np.allclose(eq_check, next_gen)

        index_sets = fastDTWF.coarse_grain(
            ps, 1000, 0.1, True, False
        )
        eq_check = fastDTWF.equilibrium_solve(
            index_sets,
            ps,
            1000,
            1e-8,
            True,
            False
        )
        next_gen = fastDTWF.mat_multiply(
            eq_check,
            index_sets,
            ps,
            1000,
            1e-8,
            True,
            False,
            0.
        )
        assert np.allclose(eq_check, next_gen)

        eq_check = fastDTWF.equilibrium_solve(
            index_sets,
            ps,
            1000,
            1e-8,
            True,
            False,
            None,
            True
        )
        next_gen = fastDTWF.mat_multiply(
            eq_check,
            index_sets,
            ps,
            1000,
            1e-8,
            True,
            False,
            0.
        )
        assert np.allclose(eq_check, next_gen)

        index_sets = fastDTWF.coarse_grain(
            ps, 1000, 0.1, False, True
        )
        eq_check = fastDTWF.equilibrium_solve(
            index_sets,
            ps,
            1000,
            1e-8,
            False,
            True,
            0.42
        )
        next_gen = fastDTWF.mat_multiply(
            eq_check,
            index_sets,
            ps,
            1000,
            1e-8,
            False,
            True,
            0.42
        )
        assert np.allclose(eq_check, next_gen)


def test_pick_eigenvec():
    for _ in range(10):
        t_mat = np.random.random((100, 100))
        t_mat /= t_mat.sum(axis=1, keepdims=True)
        t_mat = torch.tensor(t_mat, dtype=torch.float64)
        eigvals, eigvecs = torch.linalg.eig(t_mat.T)
        eigvec_check = fastDTWF._pick_eigenvec(eigvals, eigvecs)
        true_idx = torch.argmax(eigvals.real)
        eigvec_true = torch.abs(eigvecs[:, true_idx])
        assert torch.allclose(eigvec_check, eigvec_true)

        # should be fine under arbitrary complex rotations
        eigvecs_i = eigvecs * torch.complex(
            torch.tensor(0, dtype=torch.float64),
            torch.tensor(1, dtype=torch.float64)
        )
        eigvec_check = fastDTWF._pick_eigenvec(eigvals, eigvecs_i)
        assert torch.allclose(eigvec_check, eigvec_true)

        eigvecs_neg = -1 * eigvecs
        eigvec_check = fastDTWF._pick_eigenvec(eigvals, eigvecs_neg)
        assert torch.allclose(eigvec_check, eigvec_true)

        eigvecs_mi = -eigvecs_i
        eigvec_check = fastDTWF._pick_eigenvec(eigvals, eigvecs_mi)
        assert torch.allclose(eigvec_check, eigvec_true)

        eigvecs_comp = eigvecs * torch.complex(
            torch.tensor(np.sqrt(1/2), dtype=torch.float64),
            torch.tensor(np.sqrt(1/2), dtype=torch.float64)
        )
        eigvec_check = fastDTWF._pick_eigenvec(eigvals, eigvecs_comp)
        assert torch.allclose(eigvec_check, eigvec_true)


def test_one_step_solve():
    for _ in range(10):
        mu_1 = np.random.random() * 1e-8
        mu_2 = np.random.random() * 1e-8
        s_het = np.random.random() * 1e-3
        ps = fastDTWF.wright_fisher_ps_mutate_first(
            1000,
            torch.tensor(mu_1, dtype=torch.float64),
            torch.tensor(mu_2, dtype=torch.float64),
            torch.tensor(s_het, dtype=torch.float64)
        )
        index_sets = fastDTWF.coarse_grain(
            ps, 1000, 0.1, False, False
        )
        vec = torch.tensor(np.random.random(1001), dtype=torch.float64)
        vec /= vec.sum()
        trunc_ps = fastDTWF.project_to_coarse(ps, index_sets, vec)
        eq_check = fastDTWF._one_step_solve(
            index_sets,
            trunc_ps,
            1000,
            1e-8,
            False,
            False,
        )
        proj_eq = fastDTWF.project_to_coarse(eq_check, index_sets)
        eq_next = fastDTWF.mat_multiply_from_coarse(
            proj_eq,
            trunc_ps,
            1000,
            1e-8,
            False,
            False,
            0.
        )
        assert torch.allclose(eq_check, eq_next)

        index_sets = fastDTWF.coarse_grain(
            ps, 1000, 0.1, True, False
        )
        vec = torch.tensor(np.random.random(1001), dtype=torch.float64)
        vec[-1] = 0.
        vec /= vec.sum()
        trunc_ps = fastDTWF.project_to_coarse(ps, index_sets, vec)
        eq_check = fastDTWF._one_step_solve(
            index_sets,
            trunc_ps,
            1000,
            1e-8,
            True,
            False,
        )
        proj_eq = fastDTWF.project_to_coarse(eq_check, index_sets)
        eq_next = fastDTWF.mat_multiply_from_coarse(
            proj_eq,
            trunc_ps,
            1000,
            1e-8,
            True,
            False,
            0.
        )
        assert torch.allclose(eq_check, eq_next)

        index_sets = fastDTWF.coarse_grain(
            ps, 1000, 0.1, False, True
        )
        vec = torch.tensor(np.random.random(1001), dtype=torch.float64)
        vec[0] = 0.
        vec[-1] = 0.
        vec /= vec.sum()
        trunc_ps = fastDTWF.project_to_coarse(ps, index_sets, vec)
        eq_check = fastDTWF._one_step_solve(
            index_sets,
            trunc_ps,
            1000,
            1e-8,
            False,
            True,
            0.42
        )
        proj_eq = fastDTWF.project_to_coarse(eq_check, index_sets)
        eq_next = fastDTWF.mat_multiply_from_coarse(
            proj_eq,
            trunc_ps,
            1000,
            1e-8,
            False,
            True,
            0.42
        )
        assert torch.allclose(eq_check, eq_next)


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


def test_integrate_likelihood_constant_size():
    for _ in range(10):
        v = torch.tensor(np.random.random(101), dtype=torch.float64)
        v /= v.sum()

        s_het = torch.tensor(np.random.random()*1e-2, dtype=torch.float64)
        mu_0_to_1 = torch.tensor(np.random.random()*1e-3, dtype=torch.float64)
        mu_1_to_0 = torch.tensor(np.random.random()*1e-3, dtype=torch.float64)

        for no_fix, sfs in [(False, False), (True, False), (False, True)]:
            for use_condensed in [False, True]:
                if use_condensed:
                    refresh = [5, 10, 50, float('inf')]
                else:
                    refresh = [float('inf')]
                for refresh_gens in refresh:
                    if sfs:
                        injection_rate = 0.42
                    else:
                        injection_rate = 0.

                    check = fastDTWF._integrate_likelihood_constant_size(
                        vec=v,
                        interval_pop_size=100,
                        interval_length=50,
                        s_het=s_het,
                        mu_0_to_1=mu_0_to_1,
                        mu_1_to_0=mu_1_to_0,
                        dtwf_tv_sd=0.1,
                        dtwf_row_eps=1e-8,
                        no_fix=no_fix,
                        sfs=sfs,
                        injection_rate=injection_rate,
                        s_hom=None,
                        use_condensed=use_condensed,
                        refresh_gens=refresh_gens
                    )
                    true = v.detach().clone()
                    ps = fastDTWF.wright_fisher_ps_mutate_first(
                        100,
                        mu_0_to_1,
                        mu_1_to_0,
                        s_het
                    )
                    index_sets = fastDTWF.coarse_grain(
                        ps,
                        100,
                        0.1,
                        no_fix,
                        sfs
                    )
                    for i in range(50):
                        true = fastDTWF.mat_multiply(
                            true,
                            index_sets,
                            ps,
                            100,
                            1e-8,
                            no_fix,
                            sfs,
                            injection_rate
                        )
                    assert torch.allclose(true, check)


def test_get_likelihood():
    s_het = torch.tensor(1e-3, dtype=torch.float64)
    mu_0_to_1 = torch.tensor(1e-4, dtype=torch.float64)
    mu_1_to_0 = torch.tensor(2.3e-4, dtype=torch.float64)

    for no_fix, sfs in [(False, False), (True, False), (False, True)]:
        for use_condensed in [False, True]:
            if use_condensed:
                refresh = [5, 10, 50, float('inf')]
            else:
                refresh = [float('inf')]
            for refresh_gens in refresh:
                if sfs:
                    injection_rate = 0.42
                else:
                    injection_rate = 0.
                # check that continuing on is close to equilibrium
                check = fastDTWF.get_likelihood(
                    pop_size_list=[100, 100],
                    switch_points=[50, 0],
                    sample_size=100,
                    s_het=s_het,
                    mu_0_to_1=mu_0_to_1,
                    mu_1_to_0=mu_1_to_0,
                    dtwf_tv_sd=0.1,
                    dtwf_row_eps=1e-8,
                    sampling_tv_sd=0.05,
                    sampling_row_eps=1e-8,
                    no_fix=no_fix,
                    sfs=sfs,
                    injection_rate=injection_rate,
                    s_hom=None,
                    use_condensed=use_condensed,
                    refresh_gens=refresh_gens
                )
                ps = fastDTWF.wright_fisher_ps_mutate_first(
                    100,
                    mu_0_to_1,
                    mu_1_to_0,
                    s_het,
                    None
                )
                index_sets = fastDTWF.coarse_grain(
                    ps,
                    100,
                    0.1,
                    no_fix,
                    sfs
                )
                true = fastDTWF.equilibrium_solve(
                    index_sets,
                    ps,
                    100,
                    1e-8,
                    no_fix,
                    sfs,
                    injection_rate
                )
                assert torch.allclose(check, true)

                # check that the downsampling is okay
                check = fastDTWF.get_likelihood(
                    pop_size_list=[100, 100],
                    switch_points=[50, 0],
                    sample_size=10,
                    s_het=s_het,
                    mu_0_to_1=mu_0_to_1,
                    mu_1_to_0=mu_1_to_0,
                    dtwf_tv_sd=0.1,
                    dtwf_row_eps=1e-8,
                    sampling_tv_sd=0.05,
                    sampling_row_eps=1e-8,
                    no_fix=no_fix,
                    sfs=sfs,
                    injection_rate=injection_rate,
                    s_hom=None,
                    use_condensed=use_condensed,
                    refresh_gens=refresh_gens
                )
                true_down = fastDTWF.hypergeometric_sample(
                    true, 10, 0.05, 1e-8, sfs
                )
                assert torch.allclose(check, true_down)

                # check that running for a long time brings you close to
                # another equilibrium
                time_needed = 200 if (sfs or no_fix) else 1000
                check = fastDTWF.get_likelihood(
                    pop_size_list=[20, 10],
                    switch_points=[time_needed, 0],
                    sample_size=10,
                    s_het=s_het,
                    mu_0_to_1=mu_0_to_1,
                    mu_1_to_0=mu_1_to_0,
                    dtwf_tv_sd=0.1,
                    dtwf_row_eps=1e-8,
                    sampling_tv_sd=0.05,
                    sampling_row_eps=1e-8,
                    no_fix=no_fix,
                    sfs=sfs,
                    injection_rate=injection_rate,
                    s_hom=None,
                    use_condensed=use_condensed,
                    refresh_gens=refresh_gens
                )
                ps = fastDTWF.wright_fisher_ps_mutate_first(
                    10,
                    mu_0_to_1,
                    mu_1_to_0,
                    s_het,
                    None
                )
                index_sets = fastDTWF.coarse_grain(
                    ps,
                    10,
                    0.1,
                    no_fix,
                    sfs
                )
                true = fastDTWF.equilibrium_solve(
                    index_sets,
                    ps,
                    10,
                    1e-8,
                    no_fix,
                    sfs,
                    injection_rate
                )
                assert torch.allclose(check, true, rtol=1e-2)


def check_differentiable():
    for no_fix, sfs in [(False, False), (True, False), (False, True)]:
        injection_rate = 0.42 if sfs else 0.
        for use_condensed in [False, True]:
            if use_condensed:
                refresh = [5, 10, 50, float('inf')]
            else:
                refresh = [float('inf')]
            for refresh_gens in refresh:
                s_het = torch.tensor(
                    1e-3, dtype=torch.float64, requires_grad=True
                )
                mu_0_to_1 = torch.tensor(
                    1e-4, dtype=torch.float64, requires_grad=True
                )
                mu_1_to_0 = torch.tensor(
                    2.3e-4, dtype=torch.float64, requires_grad=True
                )
                check = fastDTWF.get_likelihood(
                    pop_size_list=[100, 100],
                    switch_points=[50, 0],
                    sample_size=10,
                    s_het=s_het,
                    mu_0_to_1=mu_0_to_1,
                    mu_1_to_0=mu_1_to_0,
                    dtwf_tv_sd=0.1,
                    dtwf_row_eps=1e-8,
                    sampling_tv_sd=0.05,
                    sampling_row_eps=1e-8,
                    no_fix=no_fix,
                    sfs=sfs,
                    injection_rate=injection_rate,
                    s_hom=None,
                    use_condensed=use_condensed,
                    refresh_gens=refresh_gens
                )
                val = check[1]
                val.backward()
                s_het_grad = s_het.grad
                mu_0_to_1_grad = mu_0_to_1.grad
                mu_1_to_0_grad = mu_1_to_0.grad

                # compare to numerical gradients
                s_het_eps = torch.tensor(
                    1e-3 + 1e-6, dtype=torch.float64
                )
                check_s_eps = fastDTWF.get_likelihood(
                    pop_size_list=[100, 100],
                    switch_points=[50, 0],
                    sample_size=10,
                    s_het=s_het_eps,
                    mu_0_to_1=mu_0_to_1,
                    mu_1_to_0=mu_1_to_0,
                    dtwf_tv_sd=0.1,
                    dtwf_row_eps=1e-8,
                    sampling_tv_sd=0.05,
                    sampling_row_eps=1e-8,
                    no_fix=no_fix,
                    sfs=sfs,
                    injection_rate=injection_rate,
                    s_hom=None,
                    use_condensed=use_condensed,
                    refresh_gens=refresh_gens
                )
                assert torch.isclose(
                    s_het_grad, (check_s_eps[1] - val) / 1e-6, rtol=1e-4
                )

                mu_0_to_1_eps = torch.tensor(
                    1e-4 + 1e-8, dtype=torch.float64
                )
                check_mu_0_to_1_eps = fastDTWF.get_likelihood(
                    pop_size_list=[100, 100],
                    switch_points=[50, 0],
                    sample_size=10,
                    s_het=s_het,
                    mu_0_to_1=mu_0_to_1_eps,
                    mu_1_to_0=mu_1_to_0,
                    dtwf_tv_sd=0.1,
                    dtwf_row_eps=1e-8,
                    sampling_tv_sd=0.05,
                    sampling_row_eps=1e-8,
                    no_fix=no_fix,
                    sfs=sfs,
                    injection_rate=injection_rate,
                    s_hom=None,
                    use_condensed=use_condensed,
                    refresh_gens=refresh_gens
                )
                assert torch.isclose(
                    mu_0_to_1_grad,
                    (check_mu_0_to_1_eps[1] - val) / 1e-8,
                    rtol=1e-4
                )

                mu_1_to_0_eps = torch.tensor(
                    2.3e-4 + 1e-8, dtype=torch.float64
                )
                check_mu_1_to_0_eps = fastDTWF.get_likelihood(
                    pop_size_list=[100, 100],
                    switch_points=[50, 0],
                    sample_size=10,
                    s_het=s_het,
                    mu_0_to_1=mu_0_to_1,
                    mu_1_to_0=mu_1_to_0_eps,
                    dtwf_tv_sd=0.1,
                    dtwf_row_eps=1e-8,
                    sampling_tv_sd=0.05,
                    sampling_row_eps=1e-8,
                    no_fix=no_fix,
                    sfs=sfs,
                    injection_rate=injection_rate,
                    s_hom=None,
                    use_condensed=use_condensed,
                    refresh_gens=refresh_gens
                )
                assert torch.isclose(
                    mu_1_to_0_grad,
                    (check_mu_1_to_0_eps[1] - val) / 1e-8,
                    rtol=1e-4
                )
