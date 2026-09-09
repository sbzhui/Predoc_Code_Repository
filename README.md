
README.md
# Bayesian Estimation of a Small-Scale New Keynesian DSGE Model

This repository contains a self-contained Python implementation of Bayesian estimation for a small-scale New Keynesian (NK) DSGE model. The code was developed for my master's thesis, *Overparameterized Neural Forecasting with DSGE Priors*, in which the estimated model provides a structural prior for a macroeconomic forecasting exercise.

The script estimates 13 structural parameters from three quarterly U.S. macroeconomic series: output growth (`YGR`), annualized inflation (`INFL`), and the annualized short-term nominal interest rate (`INT`).

## What the code does

`Solution_estimation_mcmc.py` implements the complete estimation pipeline:

- expresses the three-equation NK model in Sims's (2002) canonical linear rational-expectations form;
- solves the model with an ordered QZ decomposition and checks the Blanchard-Kahn conditions;
- constructs the linear Gaussian state-space representation;
- evaluates the likelihood with a Kalman filter, including support for missing observations;
- specifies economically motivated priors for the 13 structural parameters;
- samples from the posterior using multi-chain Random-Walk Metropolis-Hastings; and
- saves posterior draws, acceptance rates, summary statistics, and optional convergence diagnostics.

The sampler operates in an unconstrained parameter space: positive parameters use log transformations, persistence parameters use logit transformations, and the transformed posterior includes the required log-Jacobian correction.

## Model

The model consists of an intertemporal IS equation, a New Keynesian Phillips curve, and a Taylor rule with interest-rate smoothing. It includes monetary-policy, government-spending, and technology-growth shocks. The observables are linked to the latent states through measurement equations for output growth, inflation, and the nominal interest rate.

## Requirements

Core dependencies:

```bash
pip install numpy scipy pandas tqdm
```

Optional dependencies for trace plots and convergence diagnostics:

```bash
pip install matplotlib arviz
```

## Quick start

Prepare a CSV file containing `YGR`, `INFL`, and `INT` in that order, then run:

```python
import pandas as pd

from Solution_estimation_mcmc import run_random_walk_mh, save_mcmc_results

data = pd.read_csv("your_observed_data.csv")
Y = data[["YGR", "INFL", "INT"]].to_numpy(dtype=float)

results = run_random_walk_mh(
    Y,
    n_chains=2,
    n_draws=2_000,
    burnin=500,
    thin=2,
    seed=2026,
)

print(results["summary"])
save_mcmc_results(results, output_dir="mcmc_output")
```

The short configuration above is intended as a test run. The thesis results use four chains with 100,000 draws per chain, 5,000 burn-in draws, and a thinning interval of four. For substantive inference, proposal scales should be tuned and convergence should be assessed using acceptance rates, trace plots, effective sample sizes, and R-hat statistics.

## Main outputs

`save_mcmc_results` writes raw chains, retained posterior draws in both transformed and structural parameterizations, log-posterior values, the proposal covariance matrix, acceptance rates, and a CSV posterior summary.

## References

- Herbst, E. P., and Schorfheide, F. (2015). *Bayesian Estimation of DSGE Models*. Princeton University Press.
- Sims, C. A. (2002). "Solving Linear Rational Expectations Models." *Computational Economics*, 20, 1-20.

## Author

Yun Dou  
M.A. in Economics, University of Chicago
