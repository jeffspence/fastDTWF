# fastDTWF

Tools to compute likelihoods under the discrete-time Wright-Fisher model.

This package is meant to be used as an API.
Install the package and then import wfmoments and use the functions
to compute population and sample likelihoods under equilibrium and
non-equilibrium demographies.

### Installation
Install using `pip`. `fastDTWF` requires `numpy`, `scipy`, `torch`, and `numba`.
These will all be installed or updated if you have not yet installed them. To
install `fastDTWF`, simply type:

`pip install git+https://github.com/jeffspence/fastDTWF`

### Usage

The code itself is throughly documented, so feel free to explore the source code.

Here we briefly describe some of the main uses:

```
import fastDTWF
import torch

# Strength of selection against the "1" allele in
# heterozygotes
s_het = torch.tensor(1e-4, dtype=torch.float64)

# Per-generation mutation rate
mu = torch.tensor(1e-8, dtype=torch.float64)

# Compute the likelihood of observing different 
# allele frequencies in a population of size
# 100
freq_probs = fastDTWF.get_likelihood(
    pop_size_list=[100],    # population size of 10000
    switch_points=[0],        # sample individuals at present
    sample_size=100,        # sample the whole population
    s_het=s_het,
    mu_0_to_1=mu,             # rate of mutating from "0" to "1"
    mu_1_to_0=mu,             # rate of mutating from "1" to "0"
    dtwf_tv_sd=0.1,           # controls accuracy of approximation
    dtwf_row_eps=1e-8,        # controls accuracy of approximation
    sampling_tv_sd=0.05,      # controls accuracy of approximation
    sampling_row_eps=1e-8,    # controls accuracy of approximation
    no_fix=False,             # Whether to condition on non-fixation
    sfs=False,                # Whether to use an infinite sites model
    injection_rate=0.,        # Mutation rate under infinite sites model
)

# Probability of observing 0 "1" alleles in the population
freq_probs[0]

# Probability of observing 5 "1" alleles in the population
freq_probs[5]

# Compute the likelihood for a sample of size 10
# from the same population:
sample_probs = fastDTWF.get_likelihood(
    pop_size_list=[100],
    switch_points=[0],
    sample_size=10,           # sample only 10 haploids
    s_het=s_het,
    mu_0_to_1=mu,
    mu_1_to_0=mu,
    dtwf_tv_sd=0.1,
    dtwf_row_eps=1e-8,
    sampling_tv_sd=0.05,
    sampling_row_eps=1e-8,
    no_fix=False,
    sfs=False,
    injection_rate=0.,
)

# Probability of observing 1 "1" allele in the sample
sample_probs[1]

# Alternatively, one can directly downsample the population likelihoods:
sample_probs_direct = fastDTWF.hypergeometric_sample(
    vec=freq_probs,
    n=10,
    tv_sd=0.05,
    row_eps=1e-8,
    sfs=False
)

# Check that the two are equivalent
torch.allclose(sample_probs, sample_probs_direct)

# We can handle non-equilibrium demography by passing a list of
# population sizes and switch points.  In this case, the population
# consisted of 100 haploids until 50 generations when the population
# increased to 100000 haploids
freq_probs_non_eq = fastDTWF.get_likelihood(
    pop_size_list=[100, 100000],   # population sizes for the two epochs
    switch_points=[50, 0],         # population sizes switch 50 generations ago
    sample_size=10000,             # sample 10000 individuals
    s_het=s_het,
    mu_0_to_1=mu,
    mu_1_to_0=mu,
    dtwf_tv_sd=0.1,
    dtwf_row_eps=1e-8,
    sampling_tv_sd=0.05,
    sampling_row_eps=1e-8,
    no_fix=False,
    sfs=False,
    injection_rate=0.,
    use_condensed=True,            # Makes things slightly more approximate, but faster
)

# We can also compute transition mass functions (TMF) --- 
# the probability of transitioning from some number of alleles at
# one time point to a different number at some point in the future.
# In this case, we assume a population of 1000 haploids, and an
# initial frequency of 10% (100 out of 1000)
transition_distribution = torch.zeros(1001, dtype=torch.float64)
transition_distribution[100] = 1.

# To compute the TMF we will need the expected allele frequencies
# for each given allele frequency:
expected_allele_freqs = fastDTWF.wright_fisher_ps_mutate_first(
    pop_size=1000,
    mu_0_to_1=mu,
    mu_1_to_0=mu,
    s_het=s_het
)

# We will also need to "coarse grain" these
index_sets = fastDTWF.coarse_grain(
    p_list=expected_allele_freqs,
    pop_size=1000,
    tv_sd=0.1,
    no_fix=False,
    sfs=False    
)

# Let's see what the distribution looks like after 100 generations
for _ in range(100):
    transition_distribution = fastDTWF.mat_multiply(
        vec=transition_distribution,
        index_sets=index_sets,
        p_list=expected_allele_freqs,
        pop_size=1000,
        row_eps=1e-8,
        no_fix=False,
        sfs=False
    )

# Probability of transition from 10% frequency to 20% frequency in
# 100 generations:
transition_distribution[200]
```
