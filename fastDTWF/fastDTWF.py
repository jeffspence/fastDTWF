"""Compute likelihoods under the Discrete-Time Wright Fisher model"""
from __future__ import annotations
import math
import logging
import numpy as np
import numpy.typing as npt
import scipy.stats
import scipy.integrate
from numba import njit
import torch


@njit("int64[:](float64[:], int64, float64, boolean, boolean)")
def _coarse_grain(
    p_list: npt.NDArray[np.float64],
    pop_size: int,
    tv_sd: float,
    no_fix: bool,
    sfs: bool
) -> npt.NDArray[np.int64]:
    """numba compilable portion of coarse_grain"""

    # If only one state, it is in its own index set, and it is the only one
    if len(p_list) == 1:
        return np.zeros(1, dtype=np.int64)

    # Initialize index sets, and set first entry to belong to the first set
    index_sets = np.zeros(p_list.shape[0], dtype=np.int64)
    curr_idx_set = 0
    i = 0
    index_sets[i] = curr_idx_set

    # Guarantees that first element (fixed for ancestral)
    # will be in its own coarse graining set
    if sfs:
        curr_idx_set += 1
        i += 1
        index_sets[i] = curr_idx_set

    # We add as many success probabilities as we can before we are too far away
    # from the current one.  We start at the smallest and move toward 0.5
    curr_p = p_list[i]
    curr_sd = np.sqrt(p_list[i] * (1 - p_list[i]) / (pop_size + 2)) * tv_sd
    while p_list[i] <= 0.5 and i < len(p_list) - (2 if (no_fix or sfs) else 1):
        # If the next success probability is too far from the previous one,
        # begin a new set, and determine how far away we can go
        if (p_list[i] - curr_p) > curr_sd:
            curr_p = p_list[i]
            curr_sd = tv_sd * np.sqrt(
                1.0 / (pop_size + 2) * p_list[i] * (1 - p_list[i])
            )
            curr_idx_set += 1
        index_sets[i] = curr_idx_set
        i += 1

    # We're now down with all of the success probabilities < 0.5
    # So we move on to the next index set, and restart from
    # the other end, moving from the highest success probabilities down until
    # we hit 0.5 from the other side. Here we start with the last success
    # probability
    curr_idx_set += 1
    i = len(p_list) - 1
    index_sets[i] = curr_idx_set

    # Guarantees that last element (fixed for derived)
    # will be in its own coarse graining set. Important if we are computing the
    # SFS, where we want to ignore fixed mutations, or when we condition on
    # non-fixation, where this entry is special.
    if no_fix or sfs:
        i -= 1
        curr_idx_set += 1
        index_sets[i] = curr_idx_set

    # Same as above but going backward from 1 to 0.5
    curr_p = p_list[i]
    curr_sd = np.sqrt(p_list[i] * (1 - p_list[i]) / (pop_size + 2)) * tv_sd
    i -= 1
    while p_list[i] > 0.5 and i >= 0:
        if (curr_p - p_list[i]) > curr_sd:
            curr_p = p_list[i]
            curr_sd = tv_sd * np.sqrt(
                1.0 / (pop_size + 2) * p_list[i] * (1 - p_list[i])
            )
            curr_idx_set += 1
        index_sets[i] = curr_idx_set
        i -= 1

    return index_sets


def coarse_grain(
    p_list: torch.DoubleTensor,
    pop_size: int,
    tv_sd: float,
    no_fix: bool,
    sfs: bool
) -> torch.LongTensor:
    """
    Coarse grains a Binomial transition matrix by combining rows

    Given a vector of success probabilities, we need to assign each success
    probability to a set such that success probabilities assigned to the same
    set are "close" in the sense that their Binomial (or Hypergeometic)
    distributions are close in total variation distance. We refer to the
    resulting sets as index sets.

    Args:
        p_vec: A torch.DoubleTensor containing the success probabilities for
            each row of the matrix to be coarse grained. This must be
            non-decreasing.
        N: The population size in the next generation (i.e., N in the
            binomial or hypergeometric distribution for each row). That is, the
            matrix to be coarse grained has N+1 columns
        tv_sd: The number of standard deviations away that two success
            probabilities can be before being put in different index sets
        no_fix: Boolean representing whether we want to use these index sets
            for the case where we condition on non-fixation of the derived
            allele. In this case, we need the element corresponding to fixation
            to be in its own index set.
        sfs: Boolean representing whether we want to use these index sets for
            the case where we are computing the frequency spectrum under an
            infinite sites model. In this case, we need the element
            corresponding to fixation to be in its own index set, and we also
            need the element corresponding to loss to be in its own index set.

    Returns:
        index_sets: A torch.LongTensor where entry i is the index set that the
            i^th success probability belongs to. That is, if entry i and entry
            j are the same, then those two success probabilities belong to the
            same index set, and in the coarse grained model will be treated as
            being identical.
    """

    # Check to make sure that the success probabilities are valid and that
    # they are sorted
    assert torch.all(p_list >= 0)
    assert torch.all(p_list <= 1)
    if sfs:
        assert torch.all(torch.diff(p_list[1:-1]) >= -1e-16)
    elif no_fix:
        assert torch.all(torch.diff(p_list[:-1]) >= -1e-16)
    else:
        assert torch.all(torch.diff(p_list) >= -1e-16)

    return torch.LongTensor(
        _coarse_grain(p_list.detach().numpy(), pop_size, tv_sd, no_fix, sfs)
    )


def wright_fisher_ps_mutate_first_x_chr(
    num_haps: int,
    mu_0_to_1: torch.DoubleTensor,
    mu_1_to_0: torch.DoubleTensor,
    s_het: torch.DoubleTensor,
    s_hom: torch.DoubleTensor,
    s_hemi: torch.DoubleTensor
) -> torch.DoubleTensor:
    """
    Compute expected allele frequences on the X chromosome for DTWF models

    We use an approximation where we assume that all mutation and selection
    happens in an infinitely large pool at hardy weinberg equilibrium and then
    chromosomes are pulled from this pool to form the next generation. In
    particular, all mutation and selection happens durung gamete formation, and
    so, for example, selection happens in the males of the previous generation.
    The ordering of these events matters slightly, and this is a particular
    choice.

    Args:
        num_haps: The number of haploids in the current (parental) generation.
            Note that this should be 3/4 of the actual population size since
            males only carry one X chromosome.
        mu_0_to_1: The probability that a "0" allele mutates to a "1" allele.
            Represented as a torch.DoubleTensor scalar
        mu_1_to_0: The probability that a "1" allele mutates to a "0" allele.
            Represented as a torch.DoubleTensor scalar
        s_het: The strength of selection acting against the "1" allele in
            heterozygous females. In particular, the fitness of homozygous "0"
            individuals is 1, the fitness of heterozygous females is
            1-s_het, and the fitness of homozygous "1" females is 1-s_hom.
            Can be set to negative values for positive
            selection. Represented as a torch.DoubleTensor scalar
        s_hom: The strength of selection acting against females homozygous
            for the "1" allele. Represented as a torch.DoubleTensor scalar.
        s_hemi: The strength of selection acting against males with the "1"
            allele.  Represented as a torch.DoubleTensor scalar.

    Returns:
        A torch.DoubleTensor containing the expected frequencies in the next
        generation given a current frequency. In partciular given a current
        frequncy of i/num_haps, the ith entry of the result will contain the
        expected frequency in the next generation.
    """
    assert num_haps == int(num_haps)

    hap_freqs = torch.arange(
        num_haps + 1, dtype=torch.float64
    ) / num_haps

    # mutation
    mutated_freqs = (
        hap_freqs * (1 - mu_1_to_0)
        + (1 - hap_freqs) * mu_0_to_1
    )

    # selection in paternal gametes
    paternal_0 = (1 - mutated_freqs)
    paternal_1 = mutated_freqs * (1 - s_hemi)
    total_living_paternal = paternal_0 + paternal_1
    total_living_paternal[total_living_paternal == 0] = 1.0
    paternal_0 = paternal_0 / total_living_paternal
    paternal_1 = paternal_1 / total_living_paternal
    paternal_freqs = paternal_1

    # selection in maternal gametes
    maternal_00 = (1 - mutated_freqs)**2
    maternal_01 = 2 * mutated_freqs * (1 - mutated_freqs) * (1 - s_het)
    maternal_11 = mutated_freqs**2 * (1-s_hom)
    total_living_maternal = maternal_00 + maternal_01 + maternal_11
    total_living_maternal[total_living_maternal == 0] = 1.0
    maternal_00 = maternal_00 / total_living_maternal
    maternal_01 = maternal_01 / total_living_maternal
    maternal_11 = maternal_11 / total_living_maternal
    maternal_freqs = 0.5*maternal_01 + maternal_11

    # combined frequencies
    return (2/3) * maternal_freqs + (1/3) * paternal_freqs


def wright_fisher_ps_mutate_first(
    pop_size: int,
    mu_0_to_1: torch.DoubleTensor,
    mu_1_to_0: torch.DoubleTensor,
    s_het: torch.DoubleTensor,
    s_hom: torch.DoubleTensor = None,
) -> torch.DoubleTensor:
    """
    Compute expected allele frequencies in next generation for DTWF models

    We use an approximation where we assume that all mutation and selection
    happens in an infinitely large pool at hardy weinberg equilibrium and then
    chromosomes are pulled from that pool to form the next generation. In
    particular, all of the effects of mutation and selection are felt in
    changing success probabilities for the binomial sampling of the subsequent
    generation. We assume that mutation happens before selection. The ordering
    of the two processes results in slightly different results (except for
    selection coefficients very close to 1 and/or very large mutation rates).

    Args:
        pop_size: The number of individuals in the current (parental)
            generation
        mu_0_to_1: The probability that a "0" allele mutates to a "1" allele.
            Represented as a torch.DoubleTensor scalar
        mu_1_to_0: The probability that a "1" allele mutates to a "0" allele.
            Represented as a torch.DoubleTensor scalar
        s_het: The strength of selection acting against the "1" allele in
            heterozygotes. In particular, the fitness of homozygous "0"
            individuals is 1, the fitness of heterozygous individuals is
            1-s_het, and the fitness of homozygous "1" individuals is 1-s_hom.
            If s_hom is not provided, defaults to 2*s_het (i.e., additivity)
            unless s_het > 0.5, in which case s_hom is 1 (so that fitness is
            always nonnegative). Can be set to negative values for positive
            selection. Represented as a torch.DoubleTensor scalar
        s_hom: The strength of selection acting against individuals homozygous
            for the "1" allele. If not provided, defaults to an additive model
            with s_hom being min([2*s_het, 1]). Represented as a
            torch.DoubleTensor scalar

    Returns:
        A torch.DoubleTensor containing the expected frequencies in the next
        generation given a current frequency. In partciular given a current
        frequncy of i/pop_size, the ith entry of the result will contain the
        expected frequency in the next generation.
    """

    # Make sure provided arguments are valid
    assert pop_size == int(pop_size)

    # if s_hom is not provided, assume additivity, with s_hom = 2*s_het, but
    # care must be taken to make sure that fitness is nonnegative
    if s_hom is None:
        s_hom = torch.minimum(s_het * 2, torch.ones(1, dtype=torch.float64))

    # vector of possible allele frequencies in the parental generation
    parental_allele_freqs = torch.arange(
        pop_size + 1, dtype=torch.float64
    ) / pop_size

    # Frequencies after mutations occur
    mut_freqs = (
        parental_allele_freqs * (1 - mu_1_to_0)
        + (1 - parental_allele_freqs) * mu_0_to_1
    )

    # Pair gametes according to Hardy Weinberg, and then multiply them by their
    # relative fitnesses
    living_00 = (1 - mut_freqs) ** 2
    living_01 = 2 * mut_freqs * (1 - mut_freqs) * (1 - s_het)
    living_11 = mut_freqs**2 * (1 - s_hom)

    # Calculate how big the total pool is after selection
    total_living = living_00 + living_01 + living_11

    # Prevent division by zero
    total_living[total_living == 0] = 1.0

    # Convert to proportions
    living_00 = living_00 / total_living
    living_01 = living_01 / total_living
    living_11 = living_11 / total_living

    # Half of the alleles in 01 individuals are 1 and all of the alleles in 11
    # individuals are 1.
    return 0.5 * living_01 + living_11


@njit("float64(int64, int64, int64, int64)")
def _numba_hyp_pmf_single(k: int, N: int, K: int, n: int) -> float:
    """Compute the PMF of a Hypergeometric(k; N, K, n)"""

    # Boundary cases are set to zero
    if k < 0:
        return 0.0
    if k > n:
        return 0.0
    if k > K:
        return 0.0
    if k < n - (N - K):
        return 0.0

    # log(K choose k)
    num1 = math.lgamma(K + 1) - math.lgamma(k + 1) - math.lgamma(K - k + 1)

    # log(N - K choose n - k)
    num2 = (
        math.lgamma(N - K + 1)
        - math.lgamma(n - k + 1)
        - math.lgamma(N - K - (n - k) + 1)
    )

    # log(N choose n)
    den = math.lgamma(N + 1) - math.lgamma(n + 1) - math.lgamma(N - n + 1)

    # (K choose k) * (N - K choose n - k) / (N choose n)
    return np.exp(num1 + num2 - den)


@njit("float64[:](int64, int64, int64, int64, int64)")
def _numba_hyp_pmf(
    kmin: int, kmax: int, N: int, K: int, n: int
) -> npt.NDArray[np.float64]:
    """Compute the PMF of a Hypergeometric(k; N, K, n) for k in kmin...kmax"""

    to_return = np.empty(kmax - kmin + 1, dtype=np.float64)
    for k in range(kmin, kmax + 1):
        to_return[k - kmin] = _numba_hyp_pmf_single(k, N, K, n)
    return to_return


def _torch_hyp_pmf(
    kmin: int, kmax: int, N: int, K: int, n: int, lfacts: torch.DoubleTensor
) -> torch.DoubleTensor:
    """
    Compute the PMF of a Hypergeometric(k; N, K, n) for k in kmin...kmax

    Args:
        kmin: The lowest value (inclusive) of k at which to compute the PMF
        kmax: The highest value (inclusive) of k at which to compute the PMF
        N: The Hypergeometric N parameter
        K: The Hypergeometric K parameter
        n: The Hypergeometric n parameter
        lfacts: A torch.DoubleTensor where entry i is the log of i factorial

    Returns:
        A torch.DoubleTensor where entry i is the PMF
        Hypergeometric(kmin+i; N, K, n)
    """

    # Make sure parameters are valid
    assert kmax >= kmin
    assert N >= 0
    assert K >= 0
    assert K <= N
    assert n >= 0
    assert n <= N
    assert kmin >= 0
    assert kmax <= n

    to_return = torch.zeros(kmax - kmin + 1, dtype=torch.float64)

    # Only compute the Hypergeometric PMF for values where it is valid
    kmin_real = int(kmin)
    kmin_real = max(kmin_real, 0)
    kmin_real = max(int(n + K - N), kmin_real)
    kmax_real = int(kmax)
    kmax_real = min(kmax_real, n)
    kmax_real = min(kmax_real, K)

    # If there are no valid values, return 0 tensor
    if kmin_real > kmax_real:
        return to_return

    k_tensor = torch.arange(kmin_real, kmax_real + 1)

    # log(K choose k)
    num1 = lfacts[K] - lfacts[k_tensor] - lfacts[K - k_tensor]
    # log(N - K choose n - k)
    num2 = lfacts[N - K] - lfacts[n - k_tensor] - lfacts[N - K - n + k_tensor]
    # log(N choose n)
    den = lfacts[N] - lfacts[n] - lfacts[N - n]

    # Fill in (K choose k) * (N - K choose n - k) / (N choose n) for the values
    # of k that were valid.
    to_return[(kmin_real - kmin) : (kmax_real - kmin + 1)] = torch.exp(
        num1 + num2 - den
    )
    return to_return


@njit("float64(int64, int64, float64)")
def _numba_binom_pmf_single(k: int, N: int, p: int) -> float:
    """Compute the PMF of a Binomial(k; N, p)"""

    # Boundary cases where p is 0 or 1 cause problems for logs
    if p == 0:
        if k == 0:
            return 1.0
        return 0.0
    if p == 1:
        if k == N:
            return 1.0
        return 0.0

    # log(N choose k)
    binom_coef = (math.lgamma(N + 1)
                  - math.lgamma(k + 1)
                  - math.lgamma(N - k + 1))

    # (N choose k) * p**k * (1-p)**(N-k)
    return np.exp(binom_coef + k * np.log(p) + (N - k) * np.log(1 - p))


@njit("float64[:](int64, int64, int64, float64)")
def _numba_binom_pmf(
    kmin: int, kmax: int, N: int, p: float
) -> npt.NDArray[np.float64]:
    """Compute the PMF of a Binomial(k; N, p) for k in kmin...kmax"""

    to_return = np.empty(kmax - kmin + 1, dtype=np.float64)
    for k in range(kmin, kmax + 1):
        to_return[k - kmin] = _numba_binom_pmf_single(k, N, p)
    return to_return


def _torch_binom_pmf(
    kmin: int, kmax: int, N: int, p: torch.Tensor, bcoefs: torch.DoubleTensor
) -> torch.DoubleTensor:
    """
    Compute the PMF of a Binomial(k; N, p) for k in kmin...kmax

    Args:
        kmin: The lowest value (inclusive) of k at which to compute the PMF
        kmax: The highest value (inclusive) of k at which to compute the PMF
        N: The Binomial N parameter
        p: The Binomial success probability parameter, p. Represented as a
            torch.DoubleTensor scalar.
        bcoefs: A torch.DoubleTensor, where entry i is the log of N choose i.

    Returns:
        A torch.DoubleTensor where entry i is the PMF
        Binomial(kmin+i; N, p)
    """

    # Make sure arguments are valid
    assert kmax >= kmin
    assert p >= 0
    assert p <= 1
    assert N >= 0
    assert len(bcoefs) == N + 1
    assert kmin >= 0
    assert kmax <= N

    # Boundary cases where p is 0 or 1 cause problems for logs
    if p == 0:
        to_return = torch.zeros(kmax - kmin + 1, dtype=torch.float64)
        if kmin == 0:
            to_return[0] = 1.0
        return to_return
    if p == 1:
        to_return = torch.zeros(kmax - kmin + 1, dtype=torch.float64)
        if kmax == N:
            to_return[-1] = 1.0
        return to_return

    k_tensor = torch.arange(kmin, kmax + 1, dtype=torch.float64)
    binom_coef = bcoefs[kmin : (kmax + 1)]
    return torch.exp(
        binom_coef
        + k_tensor * torch.log(p)
        + (N - k_tensor) * torch.log(1 - p)
    )


def project_to_coarse(
    vec: torch.DoubleTensor,
    index_sets: torch.LongTensor,
    normalizer: torch.DoubleTensor = None,
) -> torch.DoubleTensor:
    """
    Combine vector entries whose indices belong to the same index set.

    Args:
        vec: Vector represented as a torch.DoubleTensor whose entries should be
            combined.
        index_sets: Index sets where entries of vec whose indices are in the
            same index set will be combined. Index sets are represented as a
            torch.LongTensor of the same shape as vec, where the value for
            each index is the label of that index's index set.
        normalizer: A vector represented as a torch.DoubleTensor of the same
            shape as vec. If provided, then a weighted average of all of the
            entries of vec whose indices belong to the same index set is
            performed, where the weights are proportional to the corresponding
            entries of normalizer. If no normalizer is provided, or if
            normalizer is None,  then the entries are simple summed, not
            averaged. If not None, normalizer must be nonnegative.

    Returns:
        A torch.DoubleTensor of shape index_sets.max()+1 where entry i is the
        combined value (either sum or weighted average --- see the normalizer
        arg) for all of the indices belonging to index set i.
    """

    # We can represent this coarse graining as a matrix vector multiplication
    # where the matrix is incredibly sparse (i.e., 1 non-zero entry per row).
    # Entry [i, j] of this matrix is 1 if index j belongs the the ith index set
    # and is zero otherwise.
    indices = torch.vstack([index_sets, torch.arange(len(index_sets))])
    values = torch.ones(len(vec), dtype=torch.float64)
    sizes = [int(index_sets.max() + 1), len(vec)]
    proj = torch.sparse_coo_tensor(indices, values, sizes)

    # If no normalizer is constructed, then we simply sum up all of the entries
    # in the same index set, which is exactly multiplying by this matrix
    if normalizer is None:
        return proj.matmul(vec)

    # Normalizer must be non-negative to be interpreted as a weighted average
    assert torch.all(normalizer >= 0)

    # Multiply each entry of the vector by the normalizer, and then divide by
    # the total amount of normalizer in each index set so that the weights sum
    # to one.
    res = proj.matmul(vec * normalizer)
    proj_normalizer = proj.matmul(normalizer)

    # If the total weight in an index set is 0, normalizing it will result in a
    # divide by zero, so instead we set it to 1. This causes the resulting
    # projected vector to be zero for this index set as all of the weights will
    # be zero.
    proj_normalizer[proj_normalizer == 0] = 1.0

    return res / proj_normalizer


def hypergeometric_sample(
    vec: torch.DoubleTensor,
    sample_size: int,
    tv_sd: float,
    row_eps: float,
    sfs: bool = False
) -> torch.DoubleTensor:
    """
    Compute sample frequency probabilities from population probabilities

    Args:
        vec: Vector of population probabilities represented as a
            torch.DoubleTensor
        n: New sample size
        tv_sd: The number of standard deviatons away that two success
            probabilities (K/N) can be befor being put in separate index sets.
        row_eps: The amount of mass you are willing to neglect when sparsifying
            the Hypergeometric distributions.
        sfs: Boolean representing whether we are computing the SFS (infinite
            sites assumption) or not.

    Returns:
        A torch.DoubleTensor of shape (n+1,) containing the sample
        probabilities
    """

    pop_size = len(vec) - 1

    # "success probabilities" are just K/N for Hypergeometric
    p_vec = torch.arange(pop_size + 1, dtype=torch.float64) / pop_size

    # Coarse grain and project the success probabilities and population
    # probabilities to the coarse grained space
    index_sets = coarse_grain(p_vec, sample_size, tv_sd, False, False)
    trunc_K_vec = pop_size * project_to_coarse(p_vec, index_sets, vec)
    trunc_mass = project_to_coarse(vec, index_sets)

    # How many k must we include on either side of the mean to capture at least
    # 1 - row_eps mass (Hoeffding)
    sd_val = math.sqrt(-sample_size * np.log(row_eps) / 2)

    to_return = torch.zeros(sample_size + 1, dtype=torch.float64)

    # Precompute log factorials
    lfacts = torch.lgamma(torch.arange(pop_size + 1, dtype=torch.float64) + 1)

    for i in range(len(trunc_mass)):
        # if the success probability is not equal to K/N for some K, then the
        # hypergeometric distribution is not well-defined. Instead do a
        # weighted average of the two closest K's
        raw_K = float(trunc_K_vec[i])
        Klow = int(raw_K)
        Khigh = Klow + 1
        plow = Khigh - raw_K
        if Khigh > pop_size:
            Khigh = Klow
            plow = 1.0

        # get the range of k we need to consider
        mean_val = sample_size * raw_K / pop_size
        kmin = int(mean_val - sd_val)
        kmax = int(mean_val + sd_val) + 1
        kmin = max(kmin, 0)
        kmax = min(kmax, sample_size)

        to_add = plow * _torch_hyp_pmf(
            kmin, kmax, pop_size, Klow, sample_size, lfacts
        )
        to_add = to_add + (1 - plow) * _torch_hyp_pmf(
            kmin, kmax, pop_size, Khigh, sample_size, lfacts
        )

        # Condition on landing in the range of k we consider, to make sure it's
        # a valid probability mass function
        to_add = to_add / to_add.sum()

        to_return[kmin : (kmax + 1)] += to_add * trunc_mass[i]

    # If this is an SFS we ignore monomorphic sites
    if sfs:
        to_return[0] = 0.0
        to_return[-1] = 0.0
        return to_return

    return to_return / to_return.sum()


@njit("float64[:](float64[:], int64)")
def naive_hypergeometric_sample(
    vec: npt.NDArray[np.float64], sample_size: int
) -> npt.NDArray[np.float64]:
    """
    "Exact" computation of hypergeometric projection

    Args:
        vec: Vector of population probabilities represented as a numpy array
        n: New sample size

    Returns:
        A numpy ndarray of shape (n+1,) containing the sample probabilities
    """
    to_return = np.zeros(sample_size + 1, dtype=np.float64)
    for K in range(len(vec)):
        to_return += vec[K] * _numba_hyp_pmf(
            0, sample_size, len(vec) - 1, K, sample_size
        )
    return to_return


def make_condensed_matrix(
    index_sets: torch.LongTensor,
    trunc_p_list: torch.DoubleTensor,
    pop_size: int,
    row_eps: float,
    no_fix: bool,
    sfs: bool,
    injection_rate: float,
    bcoefs: torch.DoubleTensor = None,
) -> torch.DoubleTensor:
    """
    Make a coarse grained transition matrix

    Specifically, if one has K condensed states, make a matrix that is K by K
    dimensional where entry [i][j] is the probability of going from "condensed"
    state i to "condensed" state j in one generation.

    Args:
        index_sets: The index sets indicating how to coarse grain represented
            as a torch.LongTensor where if index_sets[i] is k, then state i in
            the original process will be part of state k in the condensed
            process.
        trunc_p_list: The success probabilities to be used for each of the
            condensed states. trunc_p_list[k] is the success probability of
            condensed state k. Represented as a torch.DoubleTensor
        pop_size: The number of haploids in the population
        row_eps: The amound of mass you are willing to neglect when sparsifying
            the Binomial distributions.
        no_fix: Boolean representing whether we want to condition this process
            on non-fixation.
        sfs: Boolean representing whether we are computing the SFS (infinite
            sites assumption) or not.
        injection_rate: If using the SFS, the rate (per individual) at
            which mutations enter the population.
        bcoefs: A torch.DoubleTensor, where entry i is the log of pop_size
            choose i.

    Returns:
        A torch.DoubleTensor where entry [i][j] is the probability of going
        from "condensed" state i to "condensed" state j in one generation.
    """

    # Make sure arguments are valid
    assert len(trunc_p_list) == index_sets.max()+1
    assert len(index_sets) == pop_size+1
    assert bcoefs is None or len(bcoefs) == pop_size+1

    to_return = torch.zeros(
        (len(trunc_p_list), len(trunc_p_list)), dtype=torch.float64
    )

    # Precompute log of binomial coefficients
    if bcoefs is None:
        bcoefs = _make_binom_coefficients(pop_size)

    # Create transition mass function for each condensed row
    for i, prob in enumerate(trunc_p_list):

        # Find out how far away from the mean we need to consider to keep 1 -
        # row_eps mass
        mean_val = torch.round(pop_size * prob)
        sd_val = np.sqrt(-pop_size * np.log(row_eps) / 2)
        kmin = int(mean_val - sd_val)
        if kmin < 0:
            kmin = 0
            if sfs:
                kmin = 1
        kmax = int(mean_val + sd_val) + 1
        if kmax > pop_size:
            kmax = pop_size
            if sfs or no_fix:
                kmax = pop_size - 1

        # Get out the corresponding entries of the binomial PMF
        pmf = _torch_binom_pmf(kmin, kmax, pop_size, prob, bcoefs)

        # Normalize rows
        # If conditoning on non-fixation, then things should lose mass
        # proportional to how much mass would have gone to fixation.
        # fixation. If doing the SFS, then we can lose mass
        # at either fixation or loss.
        den = pmf.sum()
        normalizer = 1.0
        if sfs:
            normalizer = (
                normalizer
                - _torch_binom_pmf(0, 0, pop_size, prob, bcoefs)
            )
            normalizer = (
                normalizer
                - _torch_binom_pmf(pop_size, pop_size, pop_size, prob, bcoefs)
            )
        elif no_fix:
            normalizer = (
                normalizer
                - _torch_binom_pmf(pop_size, pop_size, pop_size, prob, bcoefs)
            )
        if den <= row_eps:
            assert normalizer <= row_eps
            den = 1.0
            normalizer = 0.0
        pmf = pmf / den * normalizer

        # All of the transitions above are indexed by the original process, not
        # the condensed process. We need to add each of these to their
        # corresponding index set.
        to_return[i].index_add_(0, index_sets[kmin : kmax + 1], pmf)

    # If we want to repeatedly do matrix-vector multiplication, we need to be a
    # bit careful in the SFS case --- we want to keep adding mass to entry 1
    # each generation.
    # The code below is a trick where if the zero entry of the vector is 1,
    # then after matrix-vector multiplication, it will remain 1, but will also
    # add the correct number of new mutants into the population.
    if sfs:
        to_return[index_sets[0], :] = 0
        to_return[index_sets[0], index_sets[0]] = 1.0
        to_return[index_sets[0], index_sets[1]] = injection_rate * pop_size
    return to_return


# stationary is prop to
# exp(-2*N*s*y) * (1-y)**(2*N*mu_1_to_0-1) * y**(2*N*mu_0_to_1-1)
# unfortunately, integrating this is pretty numerically unstable
# but, closed forms exist (it's gamma) for each term if we series
# expand the exponential.  Furthermore, each of these is conjugate
# to the binomial that we need to sample from to get the actual
# vector of probabilities so it should be possible to get a
# closed form that is exact up to truncation error.
# Catastrophic cancellation results in numerical instability
# when Ns is large.  Fortunately, in this regime, it is
# unlikely that variants will reach high frequency. This
# allows us to ignore terms involving (1-frequency) as they
# will be essentially 1. This ends up changing the density
# to a Gamma distribution.  Similarly, in this regime
# we can replace binomial sampling by poisson sampling
# which lets us solve in closed form.
def diffusion_stationary_strong_s(
    pop_size: int, s_het: float, mu_0_to_1: float
) -> torch.DoubleTensor:
    """
    Compute the stationary distribution of the WF diffusion for strong genic s

    For recurrent mutation and non-zero selection, the stationary distribution
    is proportional to
    exp(-2*N*s-het*y) * (1-y)**(2*N*mu_1_to_0-1) * y**(2*N*mu_0_to_1-1).
    Numerically integrating this is hard. There are two
    tricks to approximate this, however. In the strong selection regime,
    frequencies are unlikely to grow much above 0. Ignoring the (1-y) term
    results in an analytically tractable gamma integral. To go from the
    continuous function to probabilities for the population, one then uses the
    Poisson approximation to the binomial (since the success probability will
    be small with high probability), so the resulting distribution is gamma
    Poisson.

    Args:
        pop_size: The number of haploids in the population
        s_het: The strength of genic selection against the "1" allele
        mu_0_to_1: The rate of mutation from the 0 allele to the 1 allele

    Returns:
        A torch.DoubleTensor where entry i is the probability of observing i
        "1" alleles in the population under this strong selection diffusion
        approximaton.
    """

    if pop_size * s_het < 10:
        logging.warning(
            "Selection is probably too weak for the strong approximation "
            "to work"
        )
    to_return = torch.zeros(pop_size + 1, dtype=torch.float64)
    k_tensor = torch.arange(pop_size + 1, dtype=torch.float64)

    # This is the PMF of the gamma poisson distribution you get by doing out
    # the integral described above
    to_return = torch.exp(
        k_tensor * np.log(pop_size)
        - torch.lgamma(k_tensor + 1)
        + 2 * pop_size * mu_0_to_1 * torch.log(pop_size * s_het)
        - torch.lgamma(2 * pop_size * mu_0_to_1)
        + torch.lgamma(2 * pop_size * mu_0_to_1 + k_tensor)
        - ((2 * pop_size * mu_0_to_1 + k_tensor)
           * torch.log(pop_size * s_het + pop_size))
    )

    return to_return / to_return.sum()


def diffusion_stationary(
    pop_size: int,
    s_het: torch.DoubleTensor,
    mu_0_to_1: torch.DoubleTensor,
    mu_1_to_0: torch.DoubleTensor
) -> torch.DoubleTensor:
    """
    Compute the stationary distribution of the WF diffusion

    For recurrent mutation and non-zero selection, the stationary distribution
    is proportional to
    exp(-2*N*s-het*y) * (1-y)**(2*N*mu_1_to_0-1) * y**(2*N*mu_0_to_1-1).
    Numerically integrating this is hard.  There are two tricks to approximate
    this, however. This function calls diffusion_stationary_strong_s is s_het
    is sufficiently strong. See that function for a description of the
    analytical approximation used there. Here, if s is small, then the taylor
    series for the exponential term converges quickly, so we can include a
    small number of terms, each of which ends up being a Beta integral. To
    obtain the probabilities in a finite population, we then have a mixture of
    Beta-Binomial distributions, which is analytically tractable.

    Args:
        pop_size: The number of haploids in the population
        s_het: The strength of geneic selection against the "1" allele
        mu_0_to_1: The rate at which "0" alleles mutate to "1" alleles
        mu_1_to_0: The rrate at which "1" alleles mutate to "0" alleles

    Returns:
        A torch.DoubleTensor where entry i is the probability of observing i
        "1" alleles in the population under the diffusion approximation
    """

    # If s_het is zero we end up taking log(0) below, but the result is smooth
    # in s_het, so we can just use a tiny s_het instead
    if s_het == 0:
        s_het = torch.tensor(1e-300, dtype=torch.float64)

    # If s_het is sufficiently strong, this approximation becomes numerically
    # unstable, but we can use the other approximation instead
    if pop_size * s_het > 2:
        return diffusion_stationary_strong_s(pop_size, s_het, mu_0_to_1)

    # Determine how many terms we need to include before encountering
    # negligible error
    m_max = 1
    error = (
        m_max * torch.log(pop_size * s_het)
        - scipy.special.gammaln(m_max + 1)
    )
    while error > np.log(1e-10):
        m_max += 1
        error = (
            m_max * torch.log(pop_size * s_het)
            - scipy.special.gammaln(m_max + 1)
        )
    m_max += 1

    # We are assuming that after truncating this series it is proportional to a
    # valid density. As such we need to obtain the normalizing constant, which
    # is a sum of the normalizing constants of each of the resulting Beta
    # integrals
    normalizer = sum(
        [
            torch.exp(
                p * torch.log(pop_size * s_het)
                - scipy.special.gammaln(p + 1)
                + torch.lgamma(2 * pop_size * mu_0_to_1 + p)
                + torch.lgamma(2 * pop_size * mu_1_to_0)
                - torch.lgamma(2 * pop_size * mu_0_to_1
                               + 2 * pop_size * mu_1_to_0 + p)
            )
            * (-1) ** p
            for p in range(m_max + 1)
        ]
    )
    normalizer = 1.0 / normalizer

    # Now for each number of "1" alleles in the population, we have to evaluate
    # this mixture (with alternating signs) of Beta Binomial PMFs
    to_return = torch.zeros(pop_size + 1, dtype=torch.float64)
    for k in range(pop_size + 1):
        to_return[k] = normalizer * sum(
            [
                torch.exp(
                    scipy.special.gammaln(pop_size + 1)
                    - scipy.special.gammaln(k + 1)
                    - scipy.special.gammaln(pop_size - k + 1)
                    + torch.lgamma(k + 2 * pop_size * mu_0_to_1 + p)
                    + torch.lgamma(pop_size - k + 2 * pop_size * mu_1_to_0)
                    - torch.lgamma(
                        pop_size
                        + 2 * pop_size * mu_0_to_1
                        + 2 * pop_size * mu_1_to_0
                        + p
                    )
                    - scipy.special.gammaln(p + 1)
                    + p * torch.log(pop_size * s_het)
                )
                * (-1) ** p
                for p in range(m_max + 1)
            ]
        )

    # The above is a bit numerically unstable, and so some entries might be
    # slightly negative.
    to_return[to_return < 0.] = 0.

    # Normalize the result to make sure its a real mass function
    return to_return / to_return.sum()


def diffusion_sfs(pop_size: int, s_het: float) -> npt.NDArray[np.float64]:
    """
    Compute the SFS under the WF diffusion

    This is taken from Bustamante et al. 2001, and is similar to what is
    implemented in Krukov and Gravel 2021. Essentially, the infinite population
    size SFS at stationarity is known, but one then needs to numerically
    integrate that against the binomial PMF to obtain the finite population
    size SFS.

    Args:
        pop_size: The number of haploids in the population
        s_het: The strength of genic selection against the "1" allele

    Returns:
        A numpy.ndarray where entry i is the probability of seeing i "1"
        alleles in the population, conditioned on the position being
        segregating, under the infinite sites assumption.i
    """

    # Can use the classic 1 / k result for neutrality
    if s_het == 0:
        raw = np.zeros(pop_size+1)
        raw[1:] = 1 / np.arange(1, pop_size + 1)
        raw[-1] = 0
        return raw / raw.sum()

    scaled_s_het = pop_size * s_het
    to_return = np.zeros(pop_size + 1)

    # For each entry we have to integrate the PMF Binomial(k; N, p) against the
    # analytical formula for the continuous infinite population size function
    # for p
    for k in range(1, pop_size):

        # Define the analytical result for the continous function
        def fun(freq):
            # pylint: disable=cell-var-from-loop
            first_bit = np.exp(
                math.lgamma(pop_size + 1)
                - math.lgamma(pop_size - k + 1)
                - math.lgamma(k + 1)
                + (k - 1) * np.log(freq)
                + (pop_size - k - 1) * np.log(1 - freq)
            )
            # pylint: enable=cell-var-from-loop

            # If scaled_s_het > 500 we start running into overflow issues
            # as we are compute exp(scaled_s_het * (1-x)) - 1 in the numerator
            # and exp(scaled_s_het) - 1 in the denominator.
            # But in that case, we expect exp(scaled_s_het * (1-x)) >> 1 and
            # also exp(scaled_s_het) >> 1, in which case expm1 is essentially
            # equal to just exp, and then we can analytically cancel out the
            # numerator and denominator, resulting in just exp(-scaled_s_het*x)
            if scaled_s_het < 500:
                second_bit = np.expm1(scaled_s_het * (1 - freq))
                third_bit = np.expm1(scaled_s_het)
            else:
                second_bit = np.exp(-scaled_s_het * freq)
                third_bit = 1.0
            return first_bit * second_bit / third_bit

        # Make sure there are enough points near the edges where things get
        # weird in the integration
        pts = (10 ** np.linspace(-10, -5)).tolist()
        pts += np.linspace(10**-5, 1-10**-5).tolist()
        pts += (1 - (10 ** np.linspace(-10, -5))).tolist()

        # Perform the integral, but change the defaults so that it tries harder
        # and is more accurate
        to_return[k] = scipy.integrate.quad(
            fun, 0, 1, limit=500, maxp1=500, limlst=500, points=pts
        )[0]

    return to_return / to_return.sum()


@njit(
    "float64[:](float64[:], float64[:], int64, boolean, boolean, float64)"
)
def naive_multiply(
    vec: npt.NDArray[np.float64],
    p_list: npt.NDArray[np.float64],
    pop_size: int,
    no_fix: bool,
    sfs: bool = False,
    injection_rate: float = 0.0,
) -> npt.NDArray[np.float64]:
    """
    Compute the transition of one DTWF generation exactly

    This code performs matrix-vector multiplication under the DTWF model
    without making any approximations and hence does not scale to large
    population sizes. It should only be used to check the accuracy of other
    methods for small population sizes.

    Args:
        vec: The vector to be evolved forward according to the DTWF transition
            matrix. Represented as a np.ndarray, where entry i is the
            probability (or count) associated with having i "1" alleles in the
            population.
        p_list: The vector of success probabilities for each of the possible
            population allele frequencies. Represented as a np.ndarray
        pop_size: The size of the population in the next generation in number
            of haploids
        no_fix: Boolean indicating whether to condition on non-fixation of the
            "1" allele
        sfs: Boolean indicating whether we are computing the SFS (infinite
            sites assumption) or not.
        injection_rate: If using the SFS, the rate (per individual) at which
            mutations enter the population.
    """

    # If computing the SFS, we want to ignore monomorphic sites
    if sfs:
        assert not no_fix
        vec = vec.copy()
        vec[0] = 0.0
        vec[-1] = 0.0

    # Transition each entry in the vector according to a Binomial PMF with the
    # corresponding success probability
    to_return = np.zeros_like(vec)
    for i in range(len(to_return)):
        prob = p_list[i]
        binom = _numba_binom_pmf(0, pop_size, pop_size, prob)
        to_return += binom * vec[i]

    # If computing the SFS, remove any mutations that became fixed or lost and
    # then inject singletons.
    if sfs:
        to_return[0] = 0.0
        to_return[-1] = 0.0
        to_return[1] += pop_size * injection_rate

    # If conditioning on non-fixation, remove the alleles that fixed and then
    # renormalize.
    if no_fix:
        to_return[-1] = 0
        to_return = to_return / to_return.sum()

    return to_return


def _make_binom_coefficients(N: int) -> torch.DoubleTensor:
    """Create the log binomial coefficients (N choose k) for k in 1...N"""

    Nlgamma = torch.lgamma(torch.tensor(N + 1, dtype=torch.float64))
    k_tensor = torch.arange(N + 1, dtype=torch.float64)
    return (
        Nlgamma - torch.lgamma(k_tensor + 1) - torch.lgamma(N - k_tensor + 1)
    )


def mat_multiply_from_coarse(
    trunc_mass: torch.DoubleTensor,
    trunc_p_list: torch.DoubleTensor,
    pop_size: int,
    row_eps: float,
    no_fix: bool,
    sfs: bool,
    injection_rate: float,
    bcoefs: torch.DoubleTensor = None,
) -> torch.DoubleTensor:
    """
    Compute the DTWF transition of a coarse grained vector

    This takes a vector that has been coarse grained and then transitions it
    back into the original space.  That is, it takes a vector of size K
    corresponding to K index sets, and then evolves it forward according to the
    DTWF model into a vector of size pop_size + 1.

    Args:
        trunc_mass: The coarse grained vector to be multiplied by the DTWF
            transition matrix. That is, trunc_mass[k] is the total amount of
            mass in the original vector in states that belong to index set k.
            Represented as a torch.DoubleTensor
        trunc_p_list: A vector of representative success probabilities to use
            for each of the coarse grained states. That is, trunc_p_list[k] is
            the success probabilities to use for all of the states belonging to
            index set k. Represented as a torch.DoubleTensor
        pop_size: The number of haploids in the next generation
        row_eps: The amount of mass you are willing to neglect when sparsifying
            the Binomial distributions
        no_fix: Boolean representing whether we want to condition this process
            on non-fixation
        sfs: Boolean representing whether we are computing the SFS (infinite
            sites assumption) or not.
        injection_rate: If using the SFS, the rate (per individual) at which
            mutations enter the population.
        bcoefs: A torch.DoubleTensor, where entry i is the log of pop_size
            choose i.

    Returns:
        A torch.DoubleTensor where entry i is the value of the transitioned
        vector for the state of having i "1" alleles.
    """

    if sfs:
        assert not no_fix
        assert trunc_mass[0] == 0.

    to_return = torch.zeros(pop_size + 1, dtype=torch.float64)

    # How many entries away from the mean do we need to go for each row?
    sd_val = np.sqrt(-pop_size * np.log(row_eps) / 2)

    # Precompute log binomial coefficients
    if bcoefs is None:
        bcoefs = _make_binom_coefficients(pop_size)

    # Transition each row
    for i, prob in enumerate(trunc_p_list):

        # Find out which entries we need to consider
        mean_val = pop_size * prob
        kmin = int(mean_val - sd_val)
        kmin = max(kmin, 0)
        kmax = int(mean_val + sd_val) + 1
        kmax = min(kmax, pop_size)

        # Compute the PMF for those entries, and then normalize to conditon on
        # landing within this set of values
        pmf = _torch_binom_pmf(kmin, kmax, pop_size, prob, bcoefs)
        to_return[kmin : (kmax + 1)] += pmf * trunc_mass[i] / pmf.sum()

    # If computing the SFS, we want to ignore mutations that have been fixed or
    # lost, and also inject singletons
    if sfs:
        to_return[0] = 0.0
        to_return[1] += injection_rate * pop_size
        to_return[-1] = 0.0

    # If conditioning on non-fixation we want to drop mutations that fixed and
    # then renormalize
    if no_fix:
        to_return[-1] = 0
        to_return = to_return / to_return.sum()

    return to_return


def naive_equilibrium_solve(
    p_list: npt.NDArray[np.float64],
    pop_size: int,
    no_fix: bool,
    sfs: bool = False,
    injection_rate: float = None,
) -> npt.NDArray[np.float64]:
    """
    Compute the equilibrium of a DTWF model

    The behavior of this function changes depending on whether sfs or no_fix
    are specified. If sfs is True, then this computes the stationary SFS, where
    alleles that are fixed or lost are ignored, and at equilibrium as exactly
    counterbalanced by the influx of singletons at rate injection_rate. If
    no_fix is true, then this computes the stationary distribution of the
    process conditioned on non-fixation. This is an exact, but slow
    implementation of equilibrium_solve, and so should only be used for small
    population sizes to check accuracy.

    Args:
        ps: a numpy.ndarray where entry i is the success probability for the
            Binomial distribution when there are i "1" alleles in the
            population.
        pop_size: The number of haploid individuals in the population.
        no_fix: Boolean representing whether we want to condition this process
            on non-fixation.
        sfs: Boolean representing whether we are computing the SFS (infinite
            sites assumption) or not.
        injection_rate: If using the SFS, the rate (per individual) at which
            mutations enter the population.

    Returns:
        A np.ndarray where entry i is the stationary probability of having i
        "1" alleles, or the entry of the SFS corresponding to i "1" alleles.
    """

    # Make sure provided arguments are valid
    assert np.all(p_list >= 0)
    assert np.all(p_list <= 1)
    if sfs:
        assert np.all(np.diff(p_list[1:-1]) >= -1e-16)
    elif no_fix:
        assert np.all(np.diff(p_list[:-1]) >= -1e-16)
    else:
        assert np.all(np.diff(p_list) >= -1e-16)

    # Construct the transition matrix
    matrix = np.zeros((pop_size + 1, pop_size + 2))
    for idx, prob in enumerate(p_list):
        matrix[idx, :-1] = _numba_binom_pmf(0, pop_size, pop_size, prob)

    # In the case of the SFS we want to ignore all lost mutations, and then we
    # want an injection of singletons at each generation. This can be
    # accomplished by imagining that there is always mass of 1 on the 0
    # frequency state, and then those remain unchanged by setting entry [0][0]
    # to 1, but then they also result in injected singletons by setting [0][1]
    # to the injection rate. Then, one wants to find a vector that remains
    # unchanged after multiplication by this matrix, which can be found by
    # subtracting one on the diagonal (except for [0][0]) and then finding the
    # vector that will result in the e_1 vector (by subtracting one from the
    # diagonal except for [0][0] the stationary SFS will have 1 in the 0 entry
    # and then be zero everywhere else since it is unchanged by matrix
    # multiplication).
    if sfs:
        matrix[:, 0] = 0
        matrix[0, 0] = 1.0
        matrix[0, 1] = injection_rate * pop_size
        matrix = matrix[:-1, :-2]
        matrix[np.arange(1, pop_size), np.arange(1, pop_size)] -= 1
        e1 = np.zeros(pop_size)
        e1[0] = 1.0
        res = np.array(np.linalg.solve(matrix.T, e1).tolist() + [0.0])
        res[0] = 0.0
        return res

    # In the case of conditioning on non-fixation, we want to take the subset
    # of the matrix where we ignore stuff that gets lost or fixed. Then we
    # expect that at equilibrium after multiplying by this matrix, we get back
    # the same vector, except that it has lost mass corresponding to fixation
    # and loss. Unfortunately, to turn this into matrix equation, it becomes
    # nonlinear since the amount of mass lost also depends on the unknown
    # stationary vector. But, what is described above is obviously an
    # eigenvector of the matrix (since multiplying by the matrix results in a
    # scaled version of the same vector). Since we also expect that only a
    # little bit of mass will be lost, we can probably just take the
    # eigenvector corresponding to the largest eigenvalue
    if no_fix:
        matrix = matrix[:-1, :-2]
        eigvec = np.linalg.eig(matrix.T)[1][:, 0].real
        eigvec = eigvec / eigvec.sum()
        assert np.all(eigvec >= 0)
        return np.array(eigvec.tolist() + [0.0])

    # Otherwise, we're just computing the regular old stationary distribution.
    # In this case, we can subtract off the diagonal and want to solve for the
    # vector in the null space.  Unfortunately, the zero vector is in the null
    # space, so we include a constraint that the entries must sum to 1.
    matrix[np.arange(pop_size + 1), np.arange(pop_size + 1)] -= 1
    matrix[:, -1] = 1
    e_last = np.zeros(pop_size + 2)
    e_last[-1] = 1.0
    return np.linalg.lstsq(matrix.T, e_last, rcond=None)[0]


def equilibrium_solve(
    index_sets: torch.LongTensor,
    p_list: torch.DoubleTensor,
    pop_size: int,
    row_eps: float,
    no_fix: bool,
    sfs: bool = False,
    injection_rate: float = None,
) -> torch.DoubleTensor:
    """
    Quickly compute the equilibrium of a DTWF model

    The behavior of this function changes depending on whether sfs or no_fix
    are specified. If sfs is True, then this computes the stationary SFS, where
    alleles that are fixed or lost are ignored, and at equilibrium as exactly
    counterbalanced by the influx of singletons at rate injection_rate. If
    no_fix is true, then this computes the stationary distribution of the
    process conditioned on non-fixation.

    Args:
        index_sets: The index sets indicating how to coarse grain represented
            as a torch.LongTensor where if index_sets[i] is k, then state i in
            the original process will be part of state k in the condensed
            process.
        p_list: The vector of success probabilities for each of the possible
            population allele frequencies. Represented as a torch.DoubleTensor
        pop_size: The size of the population in number of haploids
        row_eps: The amount of mass you are willing to neglect when sparsifying
            the Binomial distributions
        no_fix: Boolean indicating whether to condition on non-fixation of the
            "1" allele
        sfs: Boolean indicating whether we are computing the SFS (infinite
            sites assumption) or not.
        injection_rate: If using the SFS, the rate (per individual) at which
            mutations enter the population.

    Returns:
        A torch.DoubleTensor where entry i is the stationary probability of
        having i "1" alleles, or the entry of the SFS corresponding to i "1"
        alleles.
    """

    # We need to pick representative success probabilities for the coarse
    # grained states, but we have a chicken/egg problem in that the
    # representative probabilities we want should be a weighted average of the
    # success probabilities weighted by the equilibrium values we are trying to
    # compute. Here we do something iterative, where we start with an
    # unweighted average, compute the equilibrium and then repeat that until it
    # converges.
    err = float("inf")
    curr_trunc_ps = project_to_coarse(
        p_list, index_sets, torch.ones_like(p_list)
    )
    idx = 0
    old_res = torch.zeros(pop_size + 1, dtype=torch.float64)

    # This loop computes the new equilibrium, and the new representative
    # success probabilities and then computes how much they've changed
    while err > 1e-8 and idx < 15:
        res = _one_step_solve(
            index_sets,
            curr_trunc_ps,
            pop_size,
            row_eps,
            no_fix,
            sfs,
            injection_rate
        )
        new_trunc_ps = project_to_coarse(p_list, index_sets, res)
        err = torch.sum(torch.abs(res - old_res))
        old_res = res
        curr_trunc_ps = new_trunc_ps
        idx += 1

    return res


def _pick_eigenvec(
    eigvals: torch.Tensor, eigvecs: torch.Tensor
) -> torch.DoubleTensor:
    """
    Pick the "best" eigenvector

    When computing stationary distributions we have to solve an eigenvalue
    problem, and sometimes those eigenvalue problems result in several
    plausible solutions. We know that the solution we want should have a real
    eigenvalue and also that all of the entries in the solution should have the
    same sign (since they will correspond to a probability distribution).

    Args:
        eigvals: Complex eigenvalues of a matrix, represented as a torch.Tensor
        eigvecs: Complex eigenvectors of a matrix, represented as a
            torch.Tensor

    Returns:
        A torch.DoubleTensor containing the eigenvector that has a large
        enough, purely real eigenvalue such that all of the entries have the
        same sign
    """

    # Eigenvalue should be real, and should be close to 1.
    kept_vals = np.where(
        torch.logical_and(torch.abs(eigvals.imag) < 1e-12, eigvals.real > 0.9)
    )[0]
    eigvals = eigvals[kept_vals]
    eigvecs = eigvecs[:, kept_vals]

    # Check if all of the entries have the same sign -- if they have the same
    # sign then the absolute value of the sum should be equal to the sum of the
    # absolute values.  Otherwise, the sum of the absolute values will be
    # larger.
    best = torch.argmin(
        torch.abs(eigvecs).sum(dim=0) - torch.abs(eigvecs.sum(dim=0))
    )

    eigvec = torch.abs(eigvecs[:, best])
    return eigvec


def _one_step_solve(
    index_sets: torch.LongTensor,
    trunc_p_list: torch.DoubleTensor,
    pop_size: int,
    row_eps: float,
    no_fix: bool,
    sfs: bool = False,
    injection_rate: float = None,
) -> torch.DoubleTensor:
    """
    Get the equilibrium of a condensed matrix

    This is a sort of "condensed" equivalent of naive_equilibrium_solve where
    we obtain the stationary distribution (or SFS) of a given coarse grained
    process, and then project that back out to the space of the original
    process. Since the condensed matrix depends on the current equilbirium (or
    SFS) this method is intended to be used iteratively.

    Args:
        index_sets: The index sets indicating how to coarse grain represented
            as a torch.LongTensor where if index_sets[i] is k, then state i in
            the original process will be part of state k in the condensed
            process.
        trunc_p_list: A vector of representative success probabilities to use
            for each of the coarse grained states. That is, trunc_p_list[k] is
            the success probabilities to use for all of the states belonging to
            index set k. Represented as a torch.DoubleTensor
        pop_size: The size of the population in number of haploids
        row_eps: The amount of mass you are willing to neglect when sparsifying
            the Binomial distributions
        no_fix: Boolean indicating whether to condition on non-fixation of the
            "1" allele
        sfs: Boolean indicating whether we are computing the SFS (infinite
            sites assumption) or not.
        injection_rate: If using the SFS, the rate (per individual) at which
            mutations enter the population.

    Returns:
        A torch.DoubleTensor where entry i is the stationary probbility
    """

    mat = make_condensed_matrix(
        index_sets,
        trunc_p_list,
        pop_size,
        row_eps,
        no_fix,
        sfs,
        injection_rate
    )

    # For the three cases below the logic is very similar to that described in
    # the comments of naive_equilibrium solve. We just need to take care to
    # think about monomorphic sites in the condensed process, which are now
    # represented by index_sets[0] and index_sets[-1]
    if sfs:
        mat[:, index_sets[0]] = 0
        mat[index_sets[0], index_sets[0]] = 1.0
        mat[index_sets[0], index_sets[1]] = injection_rate * pop_size
        keep = torch.arange(mat.shape[0]) != index_sets[-1]
        mat = mat[keep, :]
        mat = mat[:, keep]
        diag = torch.arange(mat.shape[0]) != index_sets[0]
        mat[diag, diag] -= 1
        e1 = torch.zeros(mat.shape[0], dtype=torch.float64)
        e1[index_sets[0]] = 1.0
        res = torch.linalg.solve(mat.T, e1)
        vec = torch.zeros(mat.shape[0] + 1, dtype=torch.float64)
        vec[keep] = res
        vec[index_sets[0]] = 0

    elif no_fix:
        keep = torch.arange(mat.shape[0]) != index_sets[-1]
        mat = mat[keep, :]
        mat = mat[:, keep]
        eigvals, eigvecs = torch.linalg.eig(mat.T)
        eigvec = _pick_eigenvec(eigvals, eigvecs)
        eigvec = eigvec / eigvec.sum()
        vec = torch.zeros(mat.shape[0] + 1, dtype=torch.float64)
        vec[keep] = eigvec

    else:
        eigvals, eigvecs = torch.linalg.eig(mat.T)
        eigvec = _pick_eigenvec(eigvals, eigvecs)
        vec = eigvec / eigvec.sum()

    # Regardless of we got the stationary, we now need to convert the
    # stationary of the coarse grained process to the stationary of the
    # original process
    res = mat_multiply_from_coarse(
        vec, trunc_p_list, pop_size, row_eps, no_fix, sfs, injection_rate
    )

    return res


def mat_multiply(
    vec: torch.DoubleTensor,
    index_sets: torch.LongTensor,
    p_list: torch.DoubleTensor,
    pop_size: int,
    row_eps: float,
    no_fix: bool,
    sfs: bool = False,
    injection_rate: float = 0.0,
    bcoefs: torch.DoubleTensor = None,
) -> torch.DoubleTensor:
    """
    Compute a matrix-vector multiplication for a Binomial transition matrix

    Args:
        vec: The vector, represented as torch.DoubleTensor, to be multiplied.
        index_sets: The index sets indicating how to coarse grain represented
            as a torch.LongTensor where if index_sets[i] is k, then state i in
            the original process will be part of state k in the condensed
            process.
        p_list: The vector of success probabilities representing the binomial
            transition matrix. Represented as a torch.DoubleTensor
        pop_size: The size of the population in number of haploids
        row_eps: The amount of mass you are willing to neglect when sparsifying
            the Binomial distributions
        no_fix: Boolean indicating whether to condition on non-fixation of the
            "1" allele
        sfs: Boolean indicating whether we are computing the SFS (infinite
            sites assumption) or not.
        injection_rate: If using the SFS, the rate (per individual) at which
            mutation enter the population
        bcoefs: A torch.DoubleTensor, where entry i is the log of pop_size
            choose i.
    """

    # If computing the SFS, we want to ignore monomorphic sites
    if sfs:
        vec[0] = 0.
        vec[-1] = 0.

    # Coarse grain the success probabilities and the vector we want to evolve
    # forward.
    trunc_p_list = project_to_coarse(p_list, index_sets, vec)
    trunc_mass = project_to_coarse(vec, index_sets)

    # We are now in the case covered by mat_multiply_from_coarse
    return mat_multiply_from_coarse(
        trunc_mass,
        trunc_p_list,
        pop_size,
        row_eps,
        no_fix,
        sfs,
        injection_rate,
        bcoefs
    )


def _integrate_likelihood_constant_size(
    vec: torch.DoubleTensor,
    interval_pop_size: int,
    interval_length: int,
    s_het: torch.DoubleTensor,
    mu_0_to_1: torch.DoubleTensor,
    mu_1_to_0: torch.DoubleTensor,
    dtwf_tv_sd: float,
    dtwf_row_eps: float,
    no_fix: bool,
    sfs: bool,
    injection_rate: float,
    s_hom: torch.DoubleTensor = None,
    use_condensed: bool = False,
    refresh_gens: float = float('inf'),
) -> torch.DoubleTensor:
    """
    Evolve a vector forward through several DTWF generations

    Performs all of the matrix-vector multiplications required to evolve a
    vector forward through a number of generations of constant population size.

    Args:
        vec: Vector of population probabilities represented as a
            torch.DoubleTensor
        interval_pop_size: The number of haploids in the population for
            this time interval
        interval_length: The number of generations in this time interval
        s_het: The strength of selection acting against the "1" allele in
            heterozygotes. In particular, the fitness of homozygous "0"
            individuals is 1, the fitness of heterozygous individuals is
            1-s_het, and the fitness of homozygous "1" individuals is 1-s_hom.
            If s_hom is not provided, defaults to 2*s_het (i.e., additivity)
            unless s_het > 0.5, in which case s_hom is 1 (so that fitness is
            always nonnegative). Can be set to negative values for positive
            selection. Represented as a torch.DoubleTensor scalar
        mu_0_to_1: The probability that a "0" allele mutates to a "1" allele.
            Represented as a torch.DoubleTensor scalar
        mu_1_to_0: The probability that a "1" allele mutates to a "0" allele.
            Represented as a torch.DoubleTensor scalar
        dtwf_tv_sd: The number of standard deviations away that two success
            probabilities can be before being put in different index sets
        dtwf_row_eps: The amount of mass you are willing to neglect when
            sparsifying the Binomial distributions.
        no_fix: Boolean representing whether we want to condition this process
            on non-fixation.
        sfs: Boolean representing whether we are computing the SFS (infinite
            sites assumption) or not.
        injection_rate: If using the SFS, the rate (per individual) at
            which mutations enter the population.
        s_hom: The strength of selection acting against individuals homozygous
            for the "1" allele. If not provided, defaults to an additive model
            with s_hom being min([2*s_het, 1]). Represented as a
            torch.DoubleTensor scalar
        use_condensed: Boolean representing whether or not to perform
            transitions in the "condensed" space. This can result in
            significant speedups but can result in greater errors over long
            time scales
        refresh_gens: If use_condensed is true, then every refresh_gens
            generations, the "condensed" transition matrix will be recomputed.
            This trades off speed and accuracy, with higher values being less
            accurate but faster.

    Returns:
        A torch.DoubleTensor where entry i is the value of the transitioned
        vector for the state of having i "1" alleles, transitioned to the end
        of this time interval.
    """
    # not worth using condensed matrix if the interval
    # is very short
    if (refresh_gens <= 5 or interval_length <= 5) and use_condensed:
        return _integrate_likelihood_constant_size(
            vec,
            interval_pop_size,
            interval_length,
            s_het,
            mu_0_to_1,
            mu_1_to_0,
            dtwf_tv_sd,
            dtwf_row_eps,
            no_fix,
            sfs,
            injection_rate,
            s_hom,
            False,
            refresh_gens
        )

    assert interval_length > 0

    # Precompute binomial coefficients
    curr_b_coefs = _make_binom_coefficients(interval_pop_size)

    # Start the countdown!
    num_gens_left = interval_length

    # Project from current size to new size
    curr_ps = wright_fisher_ps_mutate_first(
        len(vec)-1, mu_0_to_1, mu_1_to_0, s_het, s_hom
    )
    index_sets = coarse_grain(
        curr_ps, interval_pop_size, dtwf_tv_sd, no_fix, sfs
    )
    vec = mat_multiply(
        vec,
        index_sets,
        curr_ps,
        interval_pop_size,
        dtwf_row_eps,
        no_fix,
        sfs,
        injection_rate,
        curr_b_coefs
    )
    num_gens_left -= 1

    # If only one generation in the interval, then that projection was all that
    # we needed.
    if num_gens_left == 0:
        return vec

    # Otherwise, we're now ready to compute the matrix for while we have the
    # same constant size
    curr_ps = wright_fisher_ps_mutate_first(
        interval_pop_size, mu_0_to_1, mu_1_to_0, s_het, s_hom
    )
    index_sets = coarse_grain(
        curr_ps, interval_pop_size, dtwf_tv_sd, no_fix, sfs
    )

    # Compute the condensed transition matrix and coarse-grained probabilities
    # if we want
    if use_condensed:
        trunc_p_list = project_to_coarse(curr_ps, index_sets, vec)
        condensed_matrix = make_condensed_matrix(
            index_sets,
            trunc_p_list,
            interval_pop_size,
            dtwf_row_eps,
            no_fix,
            sfs,
            injection_rate,
            curr_b_coefs
        )
        condensed_prob_vec = project_to_coarse(vec, index_sets)
        if sfs:
            condensed_prob_vec[index_sets[0]] = 1.
        staleness = 0

    while num_gens_left > 0:
        if use_condensed:
            # If there is only one generation left, or we want to update the
            # condensed matrix, then we will need to project from the
            # coarse-grained probabilities to the actual probabilities
            if num_gens_left == 1 or staleness >= refresh_gens:
                if sfs:
                    condensed_prob_vec[index_sets[0]] = 0.
                vec = mat_multiply_from_coarse(
                    condensed_prob_vec,
                    trunc_p_list,
                    interval_pop_size,
                    dtwf_row_eps,
                    no_fix,
                    sfs,
                    injection_rate,
                    curr_b_coefs
                )
                # If the condensed matrix is stale, we will remake it using
                # the up-to-date probability vector. But if there are only
                # a few generations left, we might as well just use the
                # uncondensed algorithm until the end
                if num_gens_left > 5:
                    staleness = 0
                    condensed_prob_vec = project_to_coarse(vec,
                                                           index_sets)
                    if sfs:
                        condensed_prob_vec[index_sets[0]] = 1.
                    trunc_p_list = project_to_coarse(curr_ps,
                                                     index_sets,
                                                     vec)
                    condensed_matrix = make_condensed_matrix(
                        index_sets,
                        trunc_p_list,
                        interval_pop_size,
                        dtwf_row_eps,
                        no_fix,
                        sfs,
                        injection_rate,
                        curr_b_coefs
                    )
                else:
                    use_condensed = False
            # If we still like our condensed matrix and we have generations
            # remaining, we can obtain the next generation by a standard matrix
            # multiplication in the condensed space.
            else:
                condensed_prob_vec = torch.matmul(condensed_matrix.T,
                                                  condensed_prob_vec)
                staleness += 1  # but the matrix becomes less trustworthy

        # Otherwise, we're not doing the condensed stuff, and so we just have
        # to do a normal matrix-vector multiplication
        else:
            vec = mat_multiply(
                vec,
                index_sets,
                curr_ps,
                interval_pop_size,
                dtwf_row_eps,
                no_fix,
                sfs,
                injection_rate,
                curr_b_coefs
            )
        num_gens_left -= 1

    return vec


def get_likelihood(
    pop_size_list: list[int],
    switch_points: list[int],
    sample_size: int,
    s_het: torch.DoubleTensor,
    mu_0_to_1: torch.DoubleTensor,
    mu_1_to_0: torch.DoubleTensor,
    dtwf_tv_sd: float,
    dtwf_row_eps: float,
    sampling_tv_sd: float,
    sampling_row_eps: float,
    no_fix: bool,
    sfs: bool,
    injection_rate: float,
    s_hom: torch.DoubleTensor = None,
    use_condensed: bool = False,
    refresh_gens: float = float('inf'),
) -> torch.DoubleTensor:
    """
    Compute sample likelihoods under the DTWF model

    Computes the likelihood under piecewise constant population size models.
    The demography is represented as a pair of lists. pop_size_list contains
    the populations sizes (in number of haploids) and the switch_points list
    contains the generations at which population sizes change. For example, if
    pop_size_list = [10000, 100] and switch_points = [50, 0], then in the
    ancient past, the population size was 10000 haploids, until 50 generations
    ago, at which point the population size became 100, and that size persisted
    until present. By convention, switch_points[-1] should be 0 (i.e., sampling
    at present), but if it is not, then sample likelihoods will be computed at
    that time.

    Args:
        pop_size_list: A list of population sizes with pop_size_list[0] being
            the most ancient size and pop_size_list[-1] being the most recent
            size. Sizes are measured in number of haploids
        switch_points: A list of generations ago at which the population sizes
            changed. That is, at switch_points[0] generations ago, the
            population changed from size pop_size_list[0] to pop_size_list[1].
            switch_points[-1] should be set to zero for this interpretation. If
            switch_points[-1] then all of the switch points can be interpreted
            relative to the present, but then the individuals in the sample
            were obtained at switch_points[-1] generations ago.
        sample_size: The number of haploids in the sample taken at time
            switch_points[-1].
        s_het: The strength of selection acting against the "1" allele in
            heterozygotes. In particular, the fitness of homozygous "0"
            individuals is 1, the fitness of heterozygous individuals is
            1-s_het, and the fitness of homozygous "1" individuals is 1-s_hom.
            If s_hom is not provided, defaults to 2*s_het (i.e., additivity)
            unless s_het > 0.5, in which case s_hom is 1 (so that fitness is
            always nonnegative). Can be set to negative values for positive
            selection. Represented as a torch.DoubleTensor scalar
        mu_0_to_1: The probability that a "0" allele mutates to a "1" allele.
            Represented as a torch.DoubleTensor scalar
        mu_1_to_0: The probability that a "1" allele mutates to a "0" allele.
            Represented as a torch.DoubleTensor scalar
        dtwf_tv_sd: The number of standard deviations away that two success
            probabilities can be before being put in different index sets for
            the DTWF transition matrix.
        dtwf_row_eps: The amount of mass you are willing to neglect when
            sparsifying the Binomial distributions in the DTWF transition
            matrix.
        sampling_tv_sd: The number of standard deviations away that two success
            probabilities can be before being put in different index sets for
            the Hypergeometric sampling matrix.
        sampling_row_eps: The amount of mass you are willing to neglect when
            sparsifying the Hypergeometric distributions in the sampling
            matrix.
        no_fix: Boolean representing whether we want to condition this process
            on non-fixation.
        sfs: Boolean representing whether we are computing the SFS (infinite
            sites assumption) or not.
        injection_rate: If using the SFS, the rate (per individual) at
            which mutations enter the population.
        s_hom: The strength of selection acting against individuals homozygous
            for the "1" allele. If not provided, defaults to an additive model
            with s_hom being min([2*s_het, 1]). Represented as a
            torch.DoubleTensor scalar
        use_condensed: Boolean representing whether or not to perform
            transitions in the "condensed" space. This can result in
            significant speedups but can result in greater errors over long
            time scales
        refresh_gens: If use_condensed is true, then every refresh_gens
            generations, the "condensed" transition matrix will be recomputed
            This trades off speed and accuracy, with higher values being less
            accurate but faster.

    Returns:
        A torch.DoubleTensor where entry i is the value of the sample
        probability of having i "1" alleles in the sample at switch_points[-1]
        generations before present.
    """

    # Check provided arguments
    assert np.all(np.array(pop_size_list) > 0)
    assert np.all(np.diff(switch_points) < 0)
    assert mu_0_to_1 >= 0
    assert mu_1_to_0 >= 0
    assert sample_size >= 0
    assert sample_size <= pop_size_list[-1]
    assert s_het <= 1
    assert dtwf_tv_sd >= 0
    assert dtwf_row_eps >= 0
    assert dtwf_row_eps < 1
    assert sampling_tv_sd >= 0
    assert sampling_row_eps >= 0
    assert sampling_row_eps < 1
    assert not (sfs and no_fix)
    assert s_hom is None or s_hom <= 1

    # Compute the equilibrium population distribution
    curr_ps = wright_fisher_ps_mutate_first(
        pop_size_list[0], mu_0_to_1, mu_1_to_0, s_het, s_hom
    )
    index_sets = coarse_grain(
        curr_ps, pop_size_list[0], dtwf_tv_sd, no_fix, sfs
    )
    vec = equilibrium_solve(
        index_sets,
        curr_ps,
        pop_size_list[0],
        dtwf_row_eps,
        no_fix,
        sfs,
        injection_rate,
    )

    # Evolve the population distributions through the intervals up to the end
    # of the last one
    for interval_pop_size, negative_interval_length in zip(
        pop_size_list[1:], np.diff(switch_points)
    ):
        vec = _integrate_likelihood_constant_size(
            vec,
            interval_pop_size,
            -negative_interval_length,
            s_het,
            mu_0_to_1,
            mu_1_to_0,
            dtwf_tv_sd,
            dtwf_row_eps,
            no_fix,
            sfs,
            injection_rate,
            s_hom,
            use_condensed,
            refresh_gens
        )

    # If we have sampled the whole population, we are done
    if sample_size == pop_size_list[-1]:
        return vec

    # Otherwise, we need to take a hypergeometric from the whole population to
    # obtain our sample
    return hypergeometric_sample(
        vec, sample_size, sampling_tv_sd, sampling_row_eps, sfs
    )
