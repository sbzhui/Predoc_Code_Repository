"""
Random-Walk Metropolis-Hastings estimation for a small-scale NK DSGE model.

keep the model solution, prior, and Kalman likelihood code, and replacing posterior-mode
optimization with MCMC in transformed/unconstrained parameter space.

Important: the MCMC target includes the log-Jacobian correction for the
transformation from unconstrained parameters to original structural parameters.
"""

import numpy as np
from dataclasses import dataclass
from scipy.linalg import ordqz, solve


# ============================================================
# 1. Parameter container
# ============================================================

@dataclass
class NKParams:
    """
    Small-scale New Keynesian DSGE model parameters.

    This follows the empirical parameterization in
    Herbst and Schorfheide, Bayesian Estimation of DSGE Models.

    Important:
    - r_A, pi_A, and gamma_Q are empirical steady-state parameters.
    - beta is NOT directly estimated here.
    - beta is implied by r_A:

          beta = 1 / (1 + r_A / 400)

    where r_A is annualized and measured in percent.
    """

    # Endogenous propagation parameters
    tau: float       # inverse intertemporal elasticity of substitution = risk aversion of household
    kappa: float     # slope of the New Keynesian Phillips curve, the function of v and phi.
    psi1: float      # Taylor-rule response to inflation
    psi2: float      # Taylor-rule response to output gap
    rho_R: float     # interest-rate smoothing parameter

    # Exogenous shock persistence
    rho_g: float     # government-spending shock persistence
    rho_z: float     # technology-growth shock persistence

    # Empirical steady-state parameters
    r_A: float        # annualized steady-state real-rate component, in percent
    pi_A: float       # annualized steady-state inflation, in percent
    gamma_Q: float    # quarterly steady-state growth rate, in percent

    # Structural shock standard deviations
    sigma_R: float    # monetary policy shock standard deviation
    sigma_g: float    # government-spending shock standard deviation
    sigma_z: float    # technology-growth shock standard deviation

    @property
    def beta(self):
        """
        Discount factor implied by r_A.

        The book parameterizes the model with r_A instead of beta.
        Therefore, beta is derived from r_A.
        """
        return 1.0 / (1.0 + self.r_A / 400.0)


# ============================================================
# 2. Build Sims (2002) canonical LRE system
# ============================================================

def build_sims_matrices(p: NKParams):
    """
    Build the Sims (2002) canonical linear rational expectations system:

        Gamma0 * s_t = Gamma1 * s_{t-1} + Psi * eps_t + Pi * eta_t

    State vector:

        s_t =
        [
            y_hat_t,          # 0: output deviation
            pi_hat_t,         # 1: inflation deviation
            R_hat_t,          # 2: nominal interest-rate deviation
            epsR_t,           # 3: monetary policy shock state
            g_hat_t,          # 4: government-spending shock state
            z_hat_t,          # 5: technology-growth shock state
            E_t y_hat_{t+1},  # 6: expected future output
            E_t pi_hat_{t+1}  # 7: expected future inflation
        ]'

    Structural shock vector:

        eps_t = [eps_R,t, eps_g,t, eps_z,t]'

    Rational-expectations error vector:

        eta_t = [eta_y,t, eta_pi,t]'

    where

        eta_y,t  = y_hat_t  - E_{t-1}[y_hat_t]
        eta_pi,t = pi_hat_t - E_{t-1}[pi_hat_t]
    """

    n_state = 8
    n_shock = 3
    n_eta = 2

    Gamma0 = np.zeros((n_state, n_state))
    Gamma1 = np.zeros((n_state, n_state))
    Psi = np.zeros((n_state, n_shock))
    Pi = np.zeros((n_state, n_eta))

    # ========================================================
    # Equation 1: Consumption Euler equation
    # ========================================================
    #
    # y_hat_t
    # =
    # E_t[y_hat_{t+1}]
    # - (1/tau) * (R_hat_t - E_t[pi_hat_{t+1}] - E_t[z_hat_{t+1}])
    # + g_hat_t - E_t[g_hat_{t+1}]
    #
    # Since:
    #
    # E_t[g_hat_{t+1}] = rho_g * g_hat_t
    # E_t[z_hat_{t+1}] = rho_z * z_hat_t
    #
    # we get:
    #
    # y_hat_t
    # - E_t[y_hat_{t+1}]
    # + (1/tau) * R_hat_t
    # - (1/tau) * E_t[pi_hat_{t+1}]
    # - (rho_z/tau) * z_hat_t
    # - (1-rho_g) * g_hat_t
    # = 0
    #
    # ========================================================

    Gamma0[0, 0] = 1.0                         # y_hat_t
    Gamma0[0, 2] = 1.0 / p.tau                  # R_hat_t
    Gamma0[0, 4] = -(1.0 - p.rho_g)             # g_hat_t
    Gamma0[0, 5] = -p.rho_z / p.tau             # z_hat_t
    Gamma0[0, 6] = -1.0                         # E_t y_hat_{t+1}
    Gamma0[0, 7] = -1.0 / p.tau                 # E_t pi_hat_{t+1}

    # ========================================================
    # Equation 2: New Keynesian Phillips curve
    # ========================================================
    #
    # pi_hat_t = beta * E_t[pi_hat_{t+1}]
    #            + kappa * (y_hat_t - g_hat_t)
    #
    # Rearranged:
    #
    # pi_hat_t
    # - kappa * y_hat_t
    # + kappa * g_hat_t
    # - beta * E_t[pi_hat_{t+1}]
    # = 0
    #
    # beta is implied by r_A.
    #
    # ========================================================

    Gamma0[1, 0] = -p.kappa                     # y_hat_t
    Gamma0[1, 1] = 1.0                          # pi_hat_t
    Gamma0[1, 4] = p.kappa                      # g_hat_t
    Gamma0[1, 7] = -p.beta                      # E_t pi_hat_{t+1}

    # ========================================================
    # Equation 3: Monetary policy rule
    # ========================================================
    #
    # R_hat_t
    # =
    # rho_R * R_hat_{t-1}
    # + (1-rho_R) * psi1 * pi_hat_t
    # + (1-rho_R) * psi2 * (y_hat_t - g_hat_t)
    # + epsR_t
    #
    # Rearranged:
    #
    # R_hat_t
    # - (1-rho_R) * psi1 * pi_hat_t
    # - (1-rho_R) * psi2 * y_hat_t
    # + (1-rho_R) * psi2 * g_hat_t
    # - epsR_t
    # =
    # rho_R * R_hat_{t-1}
    #
    # ========================================================

    Gamma0[2, 0] = -(1.0 - p.rho_R) * p.psi2    # y_hat_t
    Gamma0[2, 1] = -(1.0 - p.rho_R) * p.psi1    # pi_hat_t
    Gamma0[2, 2] = 1.0                          # R_hat_t
    Gamma0[2, 3] = -1.0                         # epsR_t
    Gamma0[2, 4] = (1.0 - p.rho_R) * p.psi2     # g_hat_t
    Gamma1[2, 2] = p.rho_R                      # R_hat_{t-1}

    # ========================================================
    # Equation 4: Monetary policy shock state
    # ========================================================
    #
    # epsR_t = eps_R,t
    #
    # This shock is i.i.d., so it has no lagged state.
    #
    # ========================================================

    Gamma0[3, 3] = 1.0
    Psi[3, 0] = 1.0

    # ========================================================
    # Equation 5: Government-spending shock process
    # ========================================================
    #
    # g_hat_t = rho_g * g_hat_{t-1} + eps_g,t
    #
    # ========================================================

    Gamma0[4, 4] = 1.0
    Gamma1[4, 4] = p.rho_g
    Psi[4, 1] = 1.0

    # ========================================================
    # Equation 6: Technology-growth shock process
    # ========================================================
    #
    # z_hat_t = rho_z * z_hat_{t-1} + eps_z,t
    #
    # ========================================================

    Gamma0[5, 5] = 1.0
    Gamma1[5, 5] = p.rho_z
    Psi[5, 2] = 1.0

    # ========================================================
    # Equation 7: Expectational error for output
    # ========================================================
    #
    # eta_y,t = y_hat_t - E_{t-1}[y_hat_t]
    #
    # Since E_{t-1}[y_hat_t] is stored as the previous-period
    # state variable E_{t-1}[y_hat_t], which corresponds to
    # element 6 of s_{t-1}, we write:
    #
    # y_hat_t = E_{t-1}[y_hat_t] + eta_y,t
    #
    # ========================================================

    Gamma0[6, 0] = 1.0
    Gamma1[6, 6] = 1.0
    Pi[6, 0] = 1.0

    # ========================================================
    # Equation 8: Expectational error for inflation
    # ========================================================
    #
    # eta_pi,t = pi_hat_t - E_{t-1}[pi_hat_t]
    #
    # pi_hat_t = E_{t-1}[pi_hat_t] + eta_pi,t
    #
    # ========================================================

    Gamma0[7, 1] = 1.0
    Gamma1[7, 7] = 1.0
    Pi[7, 1] = 1.0

    state_names = [
        "y_hat",
        "pi_hat",
        "R_hat",
        "epsR",
        "g_hat",
        "z_hat",
        "E_y_next",
        "E_pi_next",
    ]

    shock_names = ["eps_R", "eps_g", "eps_z"]

    return Gamma0, Gamma1, Psi, Pi, state_names, shock_names


# ============================================================
# 3. Solve the Sims (2002) system using QZ decomposition
# ============================================================

def solve_sims_qz(Gamma0, Gamma1, Psi, Pi, div=1.000001):
    """
    Solve:

        Gamma0 * s_t = Gamma1 * s_{t-1} + Psi * eps_t + Pi * eta_t

    using the Sims (2002) QZ logic.

    The solution is returned as:

        s_t = G1 * s_{t-1} + impact * eps_t

    Key idea:
    - QZ decomposition separates stable and unstable generalized roots.
    - Non-explosive solutions require the unstable block to be killed off.
    - Rational-expectations errors eta_t are chosen to offset the effect
      of fundamental shocks on the unstable block.
    """
    # div 是误差容忍度，大概就是为了保留那些因为float精度问题，导致alpha/beta绝对值略大于1的根。
    n = Gamma0.shape[0]
    k_eta = Pi.shape[1]

    def stable_selector(alpha, beta):
        """
        scipy's QZ returns generalized roots alpha / beta
        for the pencil Gamma0 - lambda Gamma1.

        For the dynamic system:

            Gamma0 * s_t = Gamma1 * s_{t-1}

        the dynamic eigenvalue is beta / alpha.

        Stability requires:

            |beta / alpha| < 1  # 这里不是generalized roots, 而是动态系统的特征值（两者互为倒数）。
        """
        alpha_abs = np.abs(alpha)
        beta_abs = np.abs(beta)

        dyn_abs = np.where(alpha_abs > 1e-12, beta_abs / alpha_abs, np.inf) #如果alpha无限小，那就直接按照无穷大的特征根处理，因为alpha is the denominator

        return dyn_abs < div

    # Ordered QZ decomposition.
    #
    # With output="real", scipy returns real generalized Schur form
    # when possible. Stable roots are ordered first.
    S, T, alpha, beta, Q, Z = ordqz(
        Gamma0,
        Gamma1,
        sort=stable_selector,
        output="real"
    )
    # S和T就是QZ分解后的上三角矩阵和下三角矩阵， 通过选择合适的Q和Z来达成这一点；
    # S和T其实是Bayesian Estimation of DSGE那本书里的Lambda和Omega。

    dyn_eigs = np.where(np.abs(alpha) > 1e-12, beta / alpha, np.inf)

    n_stable = int(np.sum(np.abs(dyn_eigs) < div))
    n_unstable = n - n_stable

    # Blanchard-Kahn uniqueness condition:
    # number of unstable roots = number of expectational errors.
    if n_unstable != k_eta:
        raise RuntimeError(
            f"Blanchard-Kahn condition failed. "
            f"Unstable roots = {n_unstable}, "
            f"expectational errors = {k_eta}. "
            f"For a unique stable solution, these must be equal."
        )

    # Transform shock and expectation-error matrices.
    QTPsi = Q.T @ Psi # 这个是把Psi也给旋转了
    QTPi = Q.T @ Pi # 这个是把Pi也给旋转了（因为需要左乘Q才能让Gamma变成S（或者T））

    # Partition stable and unstable blocks.
    S11 = S[:n_stable, :n_stable] # 这里没有S22和T22是因为直接默认T22和S22是0了。
    T11 = T[:n_stable, :n_stable]

    Q1Psi = QTPsi[:n_stable, :] # 这俩是相当于把原来的QTPsi纵向拆分了
    Q1Pi = QTPi[:n_stable, :]

    Q2Psi = QTPsi[n_stable:, :]
    Q2Pi = QTPi[n_stable:, :]

    # The unstable block must be zero:
    #
    #     Q2Psi * eps_t + Q2Pi * eta_t = 0 这就是最核心的那个思想：冲击进来一定被offset
    #
    # Therefore:
    #
    #     eta_t = - inv(Q2Pi) Q2Psi eps_t
    #
    if np.linalg.matrix_rank(Q2Pi) < n_unstable:
        raise RuntimeError(
            "No unique stable solution: Q2Pi is rank deficient."
        )
    # 如果Q2Pi不是full rank, it menas the eta_t can not span the eps_t, it means no solution.
    M_eta_eps = -solve(Q2Pi, Q2Psi) # this is the solution for M_eta_eps, this means how the eta_t offset the eps_t.

    # 所以接下来nta可以被从系统里消除掉，因为eta can be expressed by epsilon_t
    # Stable block:
    #
    #     S11 * w1_t
    #     =
    #     T11 * w1_{t-1}
    #     + (Q1Psi + Q1Pi * M_eta_eps) * eps_t
    #
    # Therefore:
    #
    #     w1_t = A_w * w1_{t-1} + B_w * eps_t
    #
    A_w = solve(S11, T11)
    B_w = solve(S11, Q1Psi + Q1Pi @ M_eta_eps)

    # Transform back to original state vector.
    #
    # w_t = Z' s_t.
    # The unstable block w2_t is set to zero.
    # Therefore:
    #
    #     s_t = Z_stable * w1_t
    #
    Z_stable = Z[:, :n_stable]

    G1 = Z_stable @ A_w @ Z_stable.T
    impact = Z_stable @ B_w

    # Remove tiny imaginary parts and numerical noise.
    G1 = np.real_if_close(G1, tol=1000)
    impact = np.real_if_close(impact, tol=1000)
    dyn_eigs = np.real_if_close(dyn_eigs, tol=1000)

    return {
        "G1": G1,
        "impact": impact,
        "dyn_eigs": dyn_eigs,
        "n_stable": n_stable,
        "n_unstable": n_unstable,
        "M_eta_eps": M_eta_eps,
    }


# ============================================================
# 4. Build state-space representation
# ============================================================

def build_state_space(p: NKParams, measurement_error_std=None):
    """
    Build the final state-space model needed for the Kalman filter.

    Raw Sims solution:

        s_t = G1 * s_{t-1} + impact * eps_t

    But the measurement equation for output growth uses y_hat_t - y_hat_{t-1}.
    Therefore, we augment the state vector with lagged output:

        a_t = [
            s_t,
            y_hat_{t-1}
        ]

    Augmented transition equation:

        a_t = T_aug * a_{t-1} + R_aug * eps_t

    Measurement equation:

        obs_t = d_obs + Z_obs * a_t + u_t

    Observables:

        obs_t = [YGR_t, INFL_t, INT_t]'

    Measurement equations from the book:

        YGR_t  = gamma_Q + 100 * (y_hat_t - y_hat_{t-1} + z_hat_t)

        INFL_t = pi_A + 400 * pi_hat_t

        INT_t  = pi_A + r_A + 4 * gamma_Q + 400 * R_hat_t

    Note:
    - r_A is directly in the parameter vector.
    - beta is implied by r_A and used inside the NK Phillips curve.
    """

    Gamma0, Gamma1, Psi, Pi, state_names, shock_names = build_sims_matrices(p)
    sol = solve_sims_qz(Gamma0, Gamma1, Psi, Pi)

    G1 = sol["G1"]
    impact = sol["impact"]

    n_raw = G1.shape[0]
    n_aug = n_raw + 1
    n_shock = impact.shape[1]

    # --------------------------------------------------------
    # Augmented transition matrix
    # --------------------------------------------------------
    #
    # a_t = [s_t, y_hat_{t-1}]'
    #
    # s_t = G1 * s_{t-1} + impact * eps_t
    #
    # y_hat_{t-1} in a_t is equal to previous period's y_hat_t,
    # which is element 0 of s_{t-1}.
    #
    # --------------------------------------------------------

    T_aug = np.zeros((n_aug, n_aug))
    R_aug = np.zeros((n_aug, n_shock))

    T_aug[:n_raw, :n_raw] = G1
    R_aug[:n_raw, :] = impact

    # Last augmented state = lagged output deviation.
    T_aug[n_raw, 0] = 1.0

    aug_state_names = state_names + ["y_hat_lag"]

    # --------------------------------------------------------
    # Structural shock covariance matrix
    # --------------------------------------------------------
    #
    # Shock order:
    #
    #     eps_t = [eps_R,t, eps_g,t, eps_z,t]'
    #
    # --------------------------------------------------------

    Q_eps = np.diag([
        p.sigma_R**2,
        p.sigma_g**2,
        p.sigma_z**2,
    ])

    # --------------------------------------------------------
    # Measurement equation
    # --------------------------------------------------------
    #
    # obs_t = d_obs + Z_obs * a_t + u_t
    #
    # --------------------------------------------------------

    Z_obs = np.zeros((3, n_aug))
    d_obs = np.zeros(3)

    # YGR_t = gamma_Q + 100*(y_hat_t - y_hat_{t-1} + z_hat_t)
    d_obs[0] = p.gamma_Q
    Z_obs[0, 0] = 100.0       # y_hat_t
    Z_obs[0, 5] = 100.0       # z_hat_t
    Z_obs[0, n_raw] = -100.0  # y_hat_{t-1}

    # INFL_t = pi_A + 400*pi_hat_t
    d_obs[1] = p.pi_A
    Z_obs[1, 1] = 400.0       # pi_hat_t

    # INT_t = pi_A + r_A + 4*gamma_Q + 400*R_hat_t
    d_obs[2] = p.pi_A + p.r_A + 4.0 * p.gamma_Q
    Z_obs[2, 2] = 400.0       # R_hat_t

    # Optional measurement error covariance.
    #
    # If you do not want measurement errors, H_obs is zero.
    # If the Kalman filter later has numerical issues because F_t is singular,
    # you may add tiny measurement errors, for example:
    #
    #     measurement_error_std=[1e-6, 1e-6, 1e-6]
    #
    if measurement_error_std is None:
        H_obs = np.zeros((3, 3))
    else:
        measurement_error_std = np.asarray(measurement_error_std)
        H_obs = np.diag(measurement_error_std**2)

    obs_names = ["YGR", "INFL", "INT"]

    return {
        "T": T_aug,
        "R": R_aug,
        "Q": Q_eps,
        "d": d_obs,
        "Z": Z_obs,
        "H": H_obs,
        "state_names": aug_state_names,
        "obs_names": obs_names,
        "shock_names": shock_names,
        "raw_solution": sol,
        "primitive_parameters": p,
        "beta_implied_by_r_A": p.beta,
    }


# ============================================================
# 5. Pretty-print equations
# ============================================================

def _format_linear_equation(
    lhs,
    coeffs,
    names,
    const=None,
    shock_coeffs=None,
    shock_names=None,
    tol=1e-10
):
    """
    Helper function to print equations in readable linear form.
    """

    terms = []

    if const is not None and abs(const) > tol:
        terms.append(f"{const:.8g}")

    for c, name in zip(coeffs, names):
        if abs(c) > tol:
            terms.append(f"({c:.8g})*{name}")

    if shock_coeffs is not None and shock_names is not None:
        for c, name in zip(shock_coeffs, shock_names):
            if abs(c) > tol:
                terms.append(f"({c:.8g})*{name}")

    rhs = " + ".join(terms) if terms else "0"
    print(f"{lhs} = {rhs}")


def print_state_space(ss, tol=1e-8):
    """
    Print the final transition and measurement equations.

    Transition:

        a_t = T a_{t-1} + R eps_t

    Measurement:

        obs_t = d + Z a_t + u_t
    """

    T = ss["T"]
    R = ss["R"]
    d = ss["d"]
    Z = ss["Z"]

    state_names = ss["state_names"]
    lag_state_names = [name + "(-1)" for name in state_names]
    shock_names = ss["shock_names"]
    obs_names = ss["obs_names"]

    print("\n===================================================")
    print("State transition equation")
    print("===================================================")
    print("a_t = T * a_{t-1} + R * eps_t\n")

    for i, name in enumerate(state_names):
        _format_linear_equation(
            lhs=f"{name}_t",
            coeffs=T[i, :],
            names=lag_state_names,
            shock_coeffs=R[i, :],
            shock_names=shock_names,
            tol=tol
        )

    print("\n===================================================")
    print("Measurement equation")
    print("===================================================")
    print("obs_t = d + Z * a_t + u_t\n")

    for i, name in enumerate(obs_names):
        _format_linear_equation(
            lhs=f"{name}_t",
            coeffs=Z[i, :],
            names=state_names,
            const=d[i],
            tol=tol
        )

    print("\n===================================================")
    print("Parameter mapping")
    print("===================================================")
    print(f"r_A is primitive: r_A = {ss['primitive_parameters'].r_A:.8g}")
    print(f"beta is implied by r_A: beta = 1 / (1 + r_A / 400) = {ss['beta_implied_by_r_A']:.8g}")


# ============================================================
# 6. Convenience wrapper
# ============================================================

def solve_nk_model(p: NKParams, measurement_error_std=None, print_equations=True):
    """
    Main function you will call during estimation.

    Input:
        p: NKParams object

    Output:
        state-space dictionary containing:

            T: state transition matrix
            R: shock loading matrix
            Q: structural shock covariance matrix
            d: measurement intercept
            Z: measurement loading matrix
            H: measurement error covariance matrix

    These are exactly the objects needed for the Kalman filter.
    """

    ss = build_state_space(p, measurement_error_std=measurement_error_std)

    if print_equations:
        print_state_space(ss)

        print("\n===================================================")
        print("Generalized dynamic eigenvalues beta/alpha")
        print("===================================================")
        print(ss["raw_solution"]["dyn_eigs"])

        print("\nNumber of stable roots:", ss["raw_solution"]["n_stable"])
        print("Number of unstable roots:", ss["raw_solution"]["n_unstable"])

    return ss


# ============================================================
# 7. Example usage
# ============================================================

if __name__ == "__main__":

    # Example parameter values.
    # Replace these with prior mean, posterior mode, or trial parameters.
    params = NKParams(
        tau=2.0,
        kappa=0.30,
        psi1=1.50,
        psi2=0.125,
        rho_R=0.50,
        rho_g=0.80,
        rho_z=0.30,

        # Empirical steady-state parameters
        r_A=0.50,       # annualized steady-state real-rate component, percent
        pi_A=4.00,      # annualized steady-state inflation, percent
        gamma_Q=0.50,   # quarterly steady-state growth, percent

        # Shock standard deviations (quarterly model units; priors in Table 2.2 are on 100*sigma)
        sigma_R=0.40 / 100.0,
        sigma_g=1.00 / 100.0,
        sigma_z=0.50 / 100.0,
    )

    ss = solve_nk_model(params, print_equations=True)

    # Matrices for Kalman filter likelihood:
    T = ss["T"]
    R = ss["R"]
    Q = ss["Q"]
    d = ss["d"]
    Z = ss["Z"]
    H = ss["H"]

    print("\n===================================================")
    print("State-space matrices for Kalman filter")
    print("===================================================")
    print("T shape:", T.shape)
    print("R shape:", R.shape)
    print("Q shape:", Q.shape)
    print("d shape:", d.shape)
    print("Z shape:", Z.shape)
    print("H shape:", H.shape)

# ============================================================
# 8. Extra imports for likelihood, priors, and MCMC
# ============================================================

from dataclasses import asdict
from scipy.linalg import solve_discrete_lyapunov, cho_factor, cho_solve
from scipy.special import gammaln
from scipy.stats import norm, gamma as gamma_dist, beta as beta_dist


# ============================================================
# 9. Parameter transformation
# ============================================================

THETA_NAMES = [
    "tau",
    "kappa",
    "psi1",
    "psi2",
    "rho_R",
    "rho_g",
    "rho_z",
    "r_A",
    "pi_A",
    "gamma_Q",
    "sigma_R",
    "sigma_g",
    "sigma_z",
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.asarray(p)
    return np.log(p / (1.0 - p))


def params_to_unconstrained(p: NKParams):
    """
    Map structural parameters to unconstrained vector.

    Positive parameters use log transform.
    Persistence parameters in (0,1) use logit transform.
    gamma_Q is allowed to be real, so it is left untransformed.
    """

    return np.array([
        np.log(p.tau),
        np.log(p.kappa),
        np.log(p.psi1),
        np.log(p.psi2),

        logit(p.rho_R),
        logit(p.rho_g),
        logit(p.rho_z),

        np.log(p.r_A),
        np.log(p.pi_A),
        p.gamma_Q,

        np.log(p.sigma_R),
        np.log(p.sigma_g),
        np.log(p.sigma_z),
    ], dtype=float)


def unconstrained_to_params(x):
    """
    Map unconstrained vector to NKParams.

    This is the parameterization used for transformed-space MCMC.
    """

    x = np.asarray(x, dtype=float)

    return NKParams(
        tau=np.exp(x[0]),
        kappa=np.exp(x[1]),
        psi1=np.exp(x[2]),
        psi2=np.exp(x[3]),

        rho_R=sigmoid(x[4]),
        rho_g=sigmoid(x[5]),
        rho_z=sigmoid(x[6]),

        r_A=np.exp(x[7]),
        pi_A=np.exp(x[8]),
        gamma_Q=x[9],

        sigma_R=np.exp(x[10]),
        sigma_g=np.exp(x[11]),
        sigma_z=np.exp(x[12]),
    )


# ============================================================
# 10. Prior helper functions
# ============================================================

def gamma_shape_scale_from_mean_sd(mean, sd):
    """
    For Gamma(shape=a, scale=b):

        mean = a*b
        var  = a*b^2

    Therefore:

        a = (mean/sd)^2
        b = sd^2 / mean
    """

    shape = (mean / sd) ** 2
    scale = (sd ** 2) / mean
    return shape, scale


def beta_ab_from_mean_sd(mean, sd):
    """
    For Beta(a,b):

        mean = a/(a+b)
        var = mean*(1-mean)/(a+b+1)

    This function converts mean and sd into a,b.
    """

    var = sd ** 2
    tmp = mean * (1.0 - mean) / var - 1.0

    if tmp <= 0:
        raise ValueError("Invalid mean/sd for Beta prior.")

    a = mean * tmp
    b = (1.0 - mean) * tmp
    return a, b


def log_inv_gamma_sigma_pdf(sigma, nu, s):
    """
    Log density for the inverse-gamma prior on a positive scale sigma:

        p(sigma | nu, s) ∝ sigma^(-nu-1) * exp[-nu*s^2/(2*sigma^2)]

    This is the common DSGE prior form used for shock standard deviations.
    For Table 2.2, pass sigma = 100 * sigma_model (and add log(100) if mapping
    back to the model's quarterly shock std).

    The normalizing constant is included, although for posterior-kernel evaluation it is not
    essential.
    """

    if sigma <= 0:
        return -np.inf

    alpha = nu / 2.0
    beta_scale = nu * s**2 / 2.0

    # If x = sigma^2 follows InvGamma(alpha, beta_scale),
    # then density for sigma includes Jacobian 2*sigma.
    #
    # log p(sigma)
    # = log 2 + log beta^alpha - log Gamma(alpha)
    #   - (2*alpha + 1) log sigma - beta/sigma^2

    return (
        np.log(2.0)
        + alpha * np.log(beta_scale)
        - gammaln(alpha)
        - (2.0 * alpha + 1.0) * np.log(sigma)
        - beta_scale / sigma**2
    )


# ============================================================
# 11. Prior specification
# ============================================================

def log_prior_nk(p: NKParams):
    """
    Log prior for the small-scale NK model.

    This prior is calibrated to be close to the standard small-scale
    New Keynesian priors used in Herbst and Schorfheide-style examples.

    Shock standard deviations follow Table 2.2: inverse-gamma priors on
    100*sigma_R, 100*sigma_g, 100*sigma_z (with log-Jacobian to map to sigma).

    Parameter conventions:
    - r_A: annualized steady-state real-rate component, percent.
    - pi_A: annualized steady-state inflation, percent.
    - gamma_Q: quarterly steady-state growth, percent.
    """

    lp = 0.0

    # -----------------------------
    # Positive structural parameters
    # -----------------------------

    # tau ~ Gamma(mean=2.0, sd=0.5)
    a, scale = gamma_shape_scale_from_mean_sd(mean=2.0, sd=0.5)
    lp += gamma_dist.logpdf(p.tau, a=a, scale=scale)

    # kappa ~ Gamma(mean=0.30, sd=0.15)
    a, scale = gamma_shape_scale_from_mean_sd(mean=0.30, sd=0.15)
    lp += gamma_dist.logpdf(p.kappa, a=a, scale=scale)

    # psi1 ~ Gamma(mean=1.50, sd=0.25)
    a, scale = gamma_shape_scale_from_mean_sd(mean=1.50, sd=0.25)
    lp += gamma_dist.logpdf(p.psi1, a=a, scale=scale)

    # psi2 ~ Gamma(mean=0.125, sd=0.10)
    a, scale = gamma_shape_scale_from_mean_sd(mean=0.125, sd=0.10)
    lp += gamma_dist.logpdf(p.psi2, a=a, scale=scale)

    # -----------------------------
    # Persistence parameters
    # -----------------------------

    # rho_R ~ Beta(mean=0.50, sd=0.20)
    a, b = beta_ab_from_mean_sd(mean=0.50, sd=0.20)
    lp += beta_dist.logpdf(p.rho_R, a=a, b=b)

    # rho_g ~ Beta(mean=0.80, sd=0.10)
    a, b = beta_ab_from_mean_sd(mean=0.80, sd=0.10)
    lp += beta_dist.logpdf(p.rho_g, a=a, b=b)

    # rho_z ~ Beta(mean=0.30, sd=0.10)
    a, b = beta_ab_from_mean_sd(mean=0.30, sd=0.10)
    lp += beta_dist.logpdf(p.rho_z, a=a, b=b)

    # -----------------------------
    # Steady-state empirical parameters
    # -----------------------------

    # r_A ~ Gamma(mean=0.50, sd=0.25)
    # Annualized steady-state real-rate component in percent.
    a, scale = gamma_shape_scale_from_mean_sd(mean=0.50, sd=0.25)
    lp += gamma_dist.logpdf(p.r_A, a=a, scale=scale)

    # pi_A ~ Gamma(mean=4.00, sd=2.00)
    # Annualized steady-state inflation in percent.
    a, scale = gamma_shape_scale_from_mean_sd(mean=4.00, sd=2.00)
    lp += gamma_dist.logpdf(p.pi_A, a=a, scale=scale)

    # gamma_Q ~ Normal(mean=0.50, sd=0.25)
    # Quarterly steady-state growth in percent.
    lp += norm.logpdf(p.gamma_Q, loc=0.50, scale=0.25)

    # -----------------------------
    # Shock standard deviations
    # -----------------------------

    # Inverse-gamma priors from Table 2.2 (Herbst & Schorfheide).
    # The prior is placed on 100 * sigma (quarterly shock std in model units),
    # not directly on sigma:
    #   100*sigma_R ~ InvGamma(0.40, 4.00), etc.
    # If x = 100*sigma, log p_sigma(sigma) = log p_x(x) + log|dx/dsigma| = log p_x(100*sigma) + log(100).
    # The +log(100) terms are constants for posterior-kernel maximization but matter for normalized posteriors / marginal likelihood.
    log_jac_100 = np.log(100.0)
    lp += log_inv_gamma_sigma_pdf(100.0 * p.sigma_R, nu=4.0, s=0.40) + log_jac_100
    lp += log_inv_gamma_sigma_pdf(100.0 * p.sigma_g, nu=4.0, s=1.00) + log_jac_100
    lp += log_inv_gamma_sigma_pdf(100.0 * p.sigma_z, nu=4.0, s=0.50) + log_jac_100

    if not np.isfinite(lp):
        return -np.inf

    return lp


def prior_mean_params():
    """
    Convenient prior-mean parameter vector.

    This is useful as the starting point for posterior evaluation.
    """

    return NKParams(
        tau=2.0,
        kappa=0.30,
        psi1=1.50,
        psi2=0.125,
        rho_R=0.50,
        rho_g=0.80,
        rho_z=0.30,
        r_A=0.50,
        pi_A=4.00,
        gamma_Q=0.50,
        # Prior means for sigma match E[100*sigma]/100 under Table 2.2 InvGamma(nu=4) hyperparameters.
        sigma_R=0.40 / 100.0,
        sigma_g=1.00 / 100.0,
        sigma_z=0.50 / 100.0,
    )


# ============================================================
# 12. Kalman filter likelihood
# ============================================================

def initialize_state_covariance(ss, jitter=1e-10):
    """
    Initialize the Kalman filter at the unconditional distribution:

        a_0|0 = 0
        P_0|0 = P_inf

    where:

        P_inf = T P_inf T' + R Q R'

    This is appropriate because the model is written in stationary
    transformed variables.
    """

    T = ss["T"]
    R = ss["R"]
    Q = ss["Q"]

    state_innovation_cov = R @ Q @ R.T
    state_innovation_cov = 0.5 * (state_innovation_cov + state_innovation_cov.T)

    # Stability check.
    eigvals = np.linalg.eigvals(T)
    if np.max(np.abs(eigvals)) >= 1.0:
        raise RuntimeError("State transition matrix is not stable.")

    P0 = solve_discrete_lyapunov(T, state_innovation_cov)
    P0 = 0.5 * (P0 + P0.T)

    # Small jitter for numerical safety.
    P0 += jitter * np.eye(P0.shape[0])

    a0 = np.zeros(T.shape[0])

    return a0, P0


def kalman_loglikelihood(Y, ss, jitter=1e-8, return_details=False):
    """
    Kalman filter log likelihood for the linear Gaussian state-space model:

        a_t = T a_{t-1} + R eps_t,       eps_t ~ N(0,Q)

        y_t = d + Z a_t + u_t,           u_t ~ N(0,H)

    Inputs:
        Y: array of shape (T_obs, n_obs)
           Observed data ordered as [YGR, INFL, INT].
        ss: state-space dictionary returned by build_state_space().
        jitter: small diagonal addition to forecast covariance F_t.

    Missing values:
        If some entries of y_t are np.nan, the function uses only observed
        entries in that period.

    Output:
        scalar log likelihood.
    """

    Y = np.asarray(Y, dtype=float)

    if Y.ndim != 2:
        raise ValueError("Y must be a 2D array with shape (T_obs, n_obs).")

    Tmat = ss["T"]
    R = ss["R"]
    Q = ss["Q"]
    d = ss["d"]
    Z = ss["Z"]
    H = ss["H"]

    n_obs = Y.shape[1]

    if n_obs != Z.shape[0]:
        raise ValueError(
            f"Y has {n_obs} columns, but measurement equation expects {Z.shape[0]}."
        )

    a_filt, P_filt = initialize_state_covariance(ss)

    loglik = 0.0
    loglik_terms = []

    for t in range(Y.shape[0]):

        # -----------------------------
        # Prediction step
        # -----------------------------

        a_pred = Tmat @ a_filt
        P_pred = Tmat @ P_filt @ Tmat.T + R @ Q @ R.T
        P_pred = 0.5 * (P_pred + P_pred.T)

        y_t = Y[t, :]
        observed = np.isfinite(y_t)

        # If the whole row is missing, skip measurement update.
        if not np.any(observed):
            a_filt = a_pred
            P_filt = P_pred
            loglik_terms.append(0.0)
            continue

        y_obs = y_t[observed]
        d_obs = d[observed]
        Z_obs = Z[observed, :]
        H_obs = H[np.ix_(observed, observed)]

        # -----------------------------
        # Forecast error
        # -----------------------------

        v = y_obs - d_obs - Z_obs @ a_pred
        F = Z_obs @ P_pred @ Z_obs.T + H_obs
        F = 0.5 * (F + F.T)

        # Numerical jitter.
        F += jitter * np.eye(F.shape[0])

        try:
            cF, lower = cho_factor(F, lower=True, check_finite=False)
            F_inv_v = cho_solve((cF, lower), v, check_finite=False)
        except Exception as e:
            raise RuntimeError(f"Kalman forecast covariance not positive definite at t={t}.") from e

        logdet_F = 2.0 * np.sum(np.log(np.diag(cF)))
        quad = float(v.T @ F_inv_v)

        ll_t = (
            -0.5 * len(y_obs) * np.log(2.0 * np.pi)
            -0.5 * logdet_F
            -0.5 * quad
        )

        loglik += ll_t
        loglik_terms.append(ll_t)

        # -----------------------------
        # Update step
        # -----------------------------

        K = P_pred @ Z_obs.T
        K = cho_solve((cF, lower), K.T, check_finite=False).T

        a_filt = a_pred + K @ v

        # Joseph-style covariance update for numerical stability:
        I = np.eye(P_pred.shape[0])
        P_filt = (I - K @ Z_obs) @ P_pred @ (I - K @ Z_obs).T + K @ H_obs @ K.T
        P_filt = 0.5 * (P_filt + P_filt.T)

    if return_details:
        return {
            "loglik": loglik,
            "loglik_terms": np.asarray(loglik_terms),
            "a_last": a_filt,
            "P_last": P_filt,
        }

    return loglik


# ============================================================
# 13. Log posterior
# ============================================================

def check_basic_parameter_bounds(p: NKParams):
    """
    Basic parameter admissibility checks before solving the model.
    """

    positive_params = [
        p.tau,
        p.kappa,
        p.psi1,
        p.psi2,
        p.r_A,
        p.pi_A,
        p.sigma_R,
        p.sigma_g,
        p.sigma_z,
    ]

    if any(x <= 0 for x in positive_params):
        return False

    persistence_params = [p.rho_R, p.rho_g, p.rho_z]

    if any((x <= 0.0) or (x >= 1.0) for x in persistence_params):
        return False

    # Loose but useful guardrails.
    if not (-5.0 <= p.gamma_Q <= 5.0):
        return False

    return True


def log_posterior_original_params(
    p: NKParams,
    Y,
    measurement_error_std=None,
    jitter=1e-8,
    verbose=False
):
    """
    Log posterior kernel:

        log p(theta | Y)
        =
        log p(Y | theta) + log p(theta) + constant

    This function evaluates the posterior in the original structural
    parameter space.
    """

    if not check_basic_parameter_bounds(p):
        return -np.inf

    lp = log_prior_nk(p)

    if not np.isfinite(lp):
        return -np.inf

    try:
        ss = build_state_space(p, measurement_error_std=measurement_error_std)
        ll = kalman_loglikelihood(Y, ss, jitter=jitter)
    except Exception as e:
        if verbose:
            print("Model solution / Kalman likelihood failed:", repr(e))
        return -np.inf

    out = ll + lp

    if not np.isfinite(out):
        return -np.inf

    return out


# ============================================================
# 14. MCMC in transformed/unconstrained parameter space
# ============================================================


def log_jacobian_unconstrained(x):
    """
    Log absolute Jacobian for the transformation from unconstrained
    parameters x to original structural parameters theta.

    The MCMC sampler below samples x, but the prior is defined on theta.
    Therefore the correct target density in x-space is

        log p(Y | theta(x)) + log p(theta(x)) + log |d theta / d x|.

    Transformations used here:
    - positive parameters: theta = exp(x), so log Jacobian contribution is x;
    - persistence parameters: rho = sigmoid(x), so log contribution is
      log(rho) + log(1-rho);
    - gamma_Q is left unchanged, so its contribution is zero.
    """

    x = np.asarray(x, dtype=float)
    if x.shape[0] != len(THETA_NAMES):
        raise ValueError(f"Expected {len(THETA_NAMES)} parameters, got {x.shape[0]}.")

    logj = 0.0

    # Positive parameters under the log transform.
    # Indices: tau, kappa, psi1, psi2, r_A, pi_A, sigma_R, sigma_g, sigma_z.
    log_positive_indices = [0, 1, 2, 3, 7, 8, 10, 11, 12]
    logj += np.sum(x[log_positive_indices])

    # Persistence parameters under the logit transform.
    # Use numerically stable formulas:
    # log(sigmoid(z))     = -log(1 + exp(-z))
    # log(1 - sigmoid(z)) = -log(1 + exp(z))
    logit_indices = [4, 5, 6]
    z = x[logit_indices]
    log_sigmoid = -np.logaddexp(0.0, -z)
    log_one_minus_sigmoid = -np.logaddexp(0.0, z)
    logj += np.sum(log_sigmoid + log_one_minus_sigmoid)

    return float(logj)


def log_posterior_unconstrained_mcmc(
    x,
    Y,
    measurement_error_std=None,
    jitter=1e-8,
    verbose=False,
):
    """
    Log posterior target for MCMC in the unconstrained parameter space.

    This function differs from the posterior-mode objective in one crucial way:
    it adds the log-Jacobian correction. Without the Jacobian term, the sampler
    would target the wrong posterior distribution.
    """

    x = np.asarray(x, dtype=float)

    if x.shape[0] != len(THETA_NAMES):
        return -np.inf

    if not np.all(np.isfinite(x)):
        return -np.inf

    try:
        p = unconstrained_to_params(x)
    except Exception:
        return -np.inf

    logpost_theta = log_posterior_original_params(
        p,
        Y,
        measurement_error_std=measurement_error_std,
        jitter=jitter,
        verbose=verbose,
    )

    if not np.isfinite(logpost_theta):
        return -np.inf

    logj = log_jacobian_unconstrained(x)
    out = logpost_theta + logj

    if not np.isfinite(out):
        return -np.inf

    return float(out)


def params_to_vector(p: NKParams):
    """
    Convert NKParams to a numeric vector ordered by THETA_NAMES.
    """

    return np.array([getattr(p, name) for name in THETA_NAMES], dtype=float)


def samples_x_to_theta(samples_x):
    """
    Convert unconstrained MCMC draws into original structural parameters.

    Parameters
    ----------
    samples_x : ndarray
        Either shape (n_draws, n_params) or (n_chains, n_draws, n_params).

    Returns
    -------
    samples_theta : ndarray
        Same leading dimensions as samples_x, with the last dimension ordered
        by THETA_NAMES.
    """

    samples_x = np.asarray(samples_x, dtype=float)
    original_shape = samples_x.shape

    if samples_x.ndim == 2:
        flat_x = samples_x
        leading_shape = original_shape[:-1]
    elif samples_x.ndim == 3:
        flat_x = samples_x.reshape(-1, original_shape[-1])
        leading_shape = original_shape[:-1]
    else:
        raise ValueError("samples_x must have shape (draws, params) or (chains, draws, params).")

    flat_theta = np.empty_like(flat_x)
    for i, x in enumerate(flat_x):
        flat_theta[i] = params_to_vector(unconstrained_to_params(x))

    return flat_theta.reshape(*leading_shape, original_shape[-1])


def default_proposal_sd():
    """
    Default proposal standard deviations in unconstrained parameter space.

    These are deliberately conservative. If the acceptance rate is too high,
    increase them. If the acceptance rate is too low, decrease them.
    """

    return np.array([
        0.05,  # log tau
        0.07,  # log kappa
        0.05,  # log psi1
        0.20,  # log psi2
        0.10,  # logit rho_R
        0.15,  # logit rho_g
        0.10,  # logit rho_z
        0.20,  # log r_A
        0.05,  # log pi_A
        0.03,  # gamma_Q
        0.05,  # log sigma_R
        0.05,  # log sigma_g
        0.05,  # log sigma_z
    ], dtype=float)


def _maybe_tqdm(iterator, use_progress=True, description=None):
    """
    Use tqdm progress bars if tqdm is installed; otherwise fall back silently.
    """

    if not use_progress:
        return iterator

    try:
        from tqdm.auto import tqdm
        return tqdm(iterator, desc=description)
    except Exception:
        return iterator


def _find_finite_initial_x(
    base_x,
    Y,
    rng,
    measurement_error_std=None,
    jitter=1e-8,
    init_scale=0.05,
    max_tries=200,
):
    """
    Find a finite initial point for a chain.

    The model solution may fail in some parameter regions. This helper starts
    from the prior mean and then tries small random perturbations until the
    transformed-space posterior is finite.
    """

    base_x = np.asarray(base_x, dtype=float)

    base_lp = log_posterior_unconstrained_mcmc(
        base_x,
        Y,
        measurement_error_std=measurement_error_std,
        jitter=jitter,
    )
    if np.isfinite(base_lp):
        # Try a perturbed start first, but keep the base point as a backup.
        for _ in range(max_tries):
            candidate = base_x + init_scale * rng.standard_normal(len(base_x))
            lp = log_posterior_unconstrained_mcmc(
                candidate,
                Y,
                measurement_error_std=measurement_error_std,
                jitter=jitter,
            )
            if np.isfinite(lp):
                return candidate, lp
        return base_x.copy(), base_lp

    # If even the prior mean fails, try random perturbations around it.
    for _ in range(max_tries):
        candidate = base_x + init_scale * rng.standard_normal(len(base_x))
        lp = log_posterior_unconstrained_mcmc(
            candidate,
            Y,
            measurement_error_std=measurement_error_std,
            jitter=jitter,
        )
        if np.isfinite(lp):
            return candidate, lp

    raise RuntimeError(
        "Could not find a finite initial point. "
        "Check the data matrix Y, priors, model solution, and Kalman likelihood."
    )


def run_random_walk_mh_single_chain(
    Y,
    start_x,
    n_draws=20000,
    proposal_cov=None,
    measurement_error_std=None,
    jitter=1e-8,
    rng=None,
    progress=True,
    description="MH chain",
):
    """
    Run one Random-Walk Metropolis-Hastings chain in unconstrained space.

    Parameters
    ----------
    Y : ndarray
        Observed data matrix used by the Kalman likelihood.
    start_x : ndarray
        Initial point in unconstrained parameter space.
    n_draws : int
        Total number of MCMC draws, including burn-in.
    proposal_cov : ndarray
        Proposal covariance matrix in unconstrained space.
    measurement_error_std : None, scalar, or ndarray
        Passed through to build_state_space().
    jitter : float
        Numerical jitter for the Kalman filter.
    rng : numpy.random.Generator
        Random number generator.
    progress : bool
        Whether to show a progress bar if tqdm is available.

    Returns
    -------
    result : dict
        Contains chain_x, chain_logpost, and acceptance_rate.
    """

    if rng is None:
        rng = np.random.default_rng()

    start_x = np.asarray(start_x, dtype=float)
    ndim = len(THETA_NAMES)

    if start_x.shape[0] != ndim:
        raise ValueError(f"start_x must have length {ndim}.")

    if proposal_cov is None:
        sd = default_proposal_sd()
        proposal_cov = np.diag(sd ** 2)
    else:
        proposal_cov = np.asarray(proposal_cov, dtype=float)

    if proposal_cov.shape != (ndim, ndim):
        raise ValueError(f"proposal_cov must have shape {(ndim, ndim)}.")

    # Cholesky factor for multivariate normal proposal innovations.
    proposal_chol = np.linalg.cholesky(proposal_cov)

    current_x = start_x.copy()
    current_lp = log_posterior_unconstrained_mcmc(
        current_x,
        Y,
        measurement_error_std=measurement_error_std,
        jitter=jitter,
    )

    if not np.isfinite(current_lp):
        raise ValueError("The initial point has non-finite log posterior.")

    chain_x = np.empty((n_draws, ndim), dtype=float)
    chain_logpost = np.empty(n_draws, dtype=float)
    accepted = 0

    iterator = _maybe_tqdm(range(n_draws), use_progress=progress, description=description)

    for draw in iterator:
        proposal_x = current_x + proposal_chol @ rng.standard_normal(ndim)
        proposal_lp = log_posterior_unconstrained_mcmc(
            proposal_x,
            Y,
            measurement_error_std=measurement_error_std,
            jitter=jitter,
        )

        # If proposal_lp is -inf, it is automatically rejected.
        log_alpha = proposal_lp - current_lp
        if np.isfinite(log_alpha) and (np.log(rng.uniform()) < log_alpha):
            current_x = proposal_x
            current_lp = proposal_lp
            accepted += 1

        chain_x[draw] = current_x
        chain_logpost[draw] = current_lp

    return {
        "chain_x": chain_x,
        "chain_logpost": chain_logpost,
        "acceptance_rate": accepted / n_draws,
    }


def run_random_walk_mh(
    Y,
    start_params=None,
    n_chains=4,
    n_draws=20000,
    burnin=5000,
    thin=5,
    proposal_sd=None,
    proposal_cov=None,
    measurement_error_std=None,
    jitter=1e-8,
    init_scale=0.05,
    seed=12345,
    progress=True,
):
    """
    Run multiple Random-Walk Metropolis-Hastings chains.

    The sampler works in transformed/unconstrained parameter space and includes
    the correct log-Jacobian correction in the target density.

    Recommended workflow:
    1. Start with conservative proposal_sd.
    2. Check acceptance rates. A rough target is 20%--35%.
    3. If acceptance is too low, shrink proposal_sd.
    4. If acceptance is too high, increase proposal_sd.
    5. Inspect trace plots and posterior summaries before using the draws.
    """

    Y = np.asarray(Y, dtype=float)
    ndim = len(THETA_NAMES)

    if start_params is None:
        start_params = prior_mean_params()

    base_x = params_to_unconstrained(start_params)

    if proposal_cov is not None and proposal_sd is not None:
        raise ValueError("Provide either proposal_sd or proposal_cov, not both.")

    if proposal_cov is None:
        if proposal_sd is None:
            proposal_sd = default_proposal_sd()
        proposal_sd = np.asarray(proposal_sd, dtype=float)
        if proposal_sd.shape[0] != ndim:
            raise ValueError(f"proposal_sd must have length {ndim}.")
        proposal_cov = np.diag(proposal_sd ** 2)
    else:
        proposal_cov = np.asarray(proposal_cov, dtype=float)

    rng = np.random.default_rng(seed)

    chains_x = np.empty((n_chains, n_draws, ndim), dtype=float)
    chains_logpost = np.empty((n_chains, n_draws), dtype=float)
    acceptance_rates = np.empty(n_chains, dtype=float)
    initial_x = np.empty((n_chains, ndim), dtype=float)
    initial_logpost = np.empty(n_chains, dtype=float)

    for chain_id in range(n_chains):
        chain_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))

        x0, lp0 = _find_finite_initial_x(
            base_x,
            Y,
            rng=chain_rng,
            measurement_error_std=measurement_error_std,
            jitter=jitter,
            init_scale=init_scale,
        )
        initial_x[chain_id] = x0
        initial_logpost[chain_id] = lp0

        result = run_random_walk_mh_single_chain(
            Y=Y,
            start_x=x0,
            n_draws=n_draws,
            proposal_cov=proposal_cov,
            measurement_error_std=measurement_error_std,
            jitter=jitter,
            rng=chain_rng,
            progress=progress,
            description=f"MH chain {chain_id + 1}/{n_chains}",
        )

        chains_x[chain_id] = result["chain_x"]
        chains_logpost[chain_id] = result["chain_logpost"]
        acceptance_rates[chain_id] = result["acceptance_rate"]

        print(
            f"Chain {chain_id + 1}: acceptance rate = "
            f"{acceptance_rates[chain_id]:.3f}"
        )

    if burnin < 0 or burnin >= n_draws:
        raise ValueError("burnin must satisfy 0 <= burnin < n_draws.")
    if thin < 1:
        raise ValueError("thin must be at least 1.")

    posterior_x = chains_x[:, burnin::thin, :]
    posterior_logpost = chains_logpost[:, burnin::thin]
    posterior_theta = samples_x_to_theta(posterior_x)
    summary = summarize_posterior(posterior_theta)

    return {
        "chains_x": chains_x,
        "chains_logpost": chains_logpost,
        "posterior_x": posterior_x,
        "posterior_theta": posterior_theta,
        "posterior_logpost": posterior_logpost,
        "acceptance_rates": acceptance_rates,
        "initial_x": initial_x,
        "initial_logpost": initial_logpost,
        "proposal_cov": proposal_cov,
        "burnin": burnin,
        "thin": thin,
        "theta_names": THETA_NAMES,
        "summary": summary,
    }


def summarize_posterior(samples_theta, names=None):
    """
    Create a posterior summary table from original-parameter MCMC draws.

    Parameters
    ----------
    samples_theta : ndarray
        Shape (n_chains, n_kept_draws, n_params) or (n_draws, n_params).
    names : list[str]
        Parameter names. Defaults to THETA_NAMES.
    """

    if names is None:
        names = THETA_NAMES

    samples_theta = np.asarray(samples_theta, dtype=float)
    ndim = samples_theta.shape[-1]
    flat = samples_theta.reshape(-1, ndim)

    rows = []
    for j, name in enumerate(names):
        draws = flat[:, j]
        rows.append({
            "parameter": name,
            "mean": np.mean(draws),
            "sd": np.std(draws, ddof=1),
            "median": np.quantile(draws, 0.50),
            "q05": np.quantile(draws, 0.05),
            "q10": np.quantile(draws, 0.10),
            "q90": np.quantile(draws, 0.90),
            "q95": np.quantile(draws, 0.95),
        })

    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except Exception:
        return rows


def save_mcmc_results(results, output_dir="mcmc_output"):
    """
    Save MCMC arrays and posterior summary to disk.
    """

    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "chains_x.npy", results["chains_x"])
    np.save(output_dir / "chains_logpost.npy", results["chains_logpost"])
    np.save(output_dir / "posterior_x.npy", results["posterior_x"])
    np.save(output_dir / "posterior_theta.npy", results["posterior_theta"])
    np.save(output_dir / "posterior_logpost.npy", results["posterior_logpost"])
    np.save(output_dir / "acceptance_rates.npy", results["acceptance_rates"])
    np.save(output_dir / "proposal_cov.npy", results["proposal_cov"])

    # Save posterior summary.
    summary = results["summary"]
    if hasattr(summary, "to_csv"):
        summary.to_csv(output_dir / "posterior_summary.csv", index=False)
    else:
        import csv
        with open(output_dir / "posterior_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    # Save acceptance rates in a simple CSV format.
    with open(output_dir / "acceptance_rates.csv", "w") as f:
        f.write("chain,acceptance_rate\n")
        for i, rate in enumerate(results["acceptance_rates"]):
            f.write(f"{i + 1},{rate}\n")

    return output_dir


def plot_trace(results, parameter_names=None, max_params=None):
    """
    Basic trace plots for original structural parameters.

    This function is optional and requires matplotlib.
    """

    import matplotlib.pyplot as plt

    samples_theta = results["posterior_theta"]
    names = results.get("theta_names", THETA_NAMES)

    if parameter_names is None:
        parameter_names = names
    if max_params is not None:
        parameter_names = parameter_names[:max_params]

    name_to_idx = {name: i for i, name in enumerate(names)}

    for name in parameter_names:
        j = name_to_idx[name]
        plt.figure(figsize=(10, 4))
        for c in range(samples_theta.shape[0]):
            plt.plot(samples_theta[c, :, j], alpha=0.8, label=f"chain {c + 1}")
        plt.title(f"Trace plot: {name}")
        plt.xlabel("Kept draw index")
        plt.ylabel(name)
        plt.legend()
        plt.tight_layout()
        plt.show()


def arviz_summary(results):
    """
    Optional ArviZ summary with R-hat and effective sample sizes.

    Install ArviZ if needed:
        pip install arviz
    """

    import arviz as az

    posterior_theta = results["posterior_theta"]
    names = list(results.get("theta_names", THETA_NAMES))
    n_chains, n_draws, _ = posterior_theta.shape

    posterior = {
        name: posterior_theta[:, :, j]
        for j, name in enumerate(names)
    }
    coords = {"chain": np.arange(n_chains), "draw": np.arange(n_draws)}
    dims = {name: ["chain", "draw"] for name in names}

    # ArviZ >= 1.0: from_dict(data={"posterior": ...}); older: posterior=...
    try:
        idata = az.from_dict(
            {"posterior": posterior}, coords=coords, dims=dims
        )
    except TypeError:
        idata = az.from_dict(posterior=posterior, coords=coords, dims=dims)

    return az.summary(idata)


# ============================================================
# 15. Example usage
# ============================================================

if __name__ == "__main__":
    """
    Example usage. Replace the placeholder Y with your actual data matrix.

    Typical workflow:

        import pandas as pd
        from Solution_estimation_mcmc import run_random_walk_mh, save_mcmc_results

        df = pd.read_csv("your_observed_data.csv")
        Y = df[["YGR", "INFL", "INT"]].to_numpy(dtype=float)

        results = run_random_walk_mh(
            Y,
            n_chains=4,
            n_draws=20000,
            burnin=5000,
            thin=5,
            seed=2026,
            progress=True,
        )

        print(results["summary"])
        save_mcmc_results(results, output_dir="mcmc_output")

    Tuning advice:
    - If acceptance rates are below roughly 0.15, reduce proposal_sd.
    - If acceptance rates are above roughly 0.45, increase proposal_sd.
    - For a first test run, use n_draws=2000 and n_chains=2.
    - For final posterior results, use longer chains and inspect trace plots.
    """

    print(
        "This file defines the DSGE posterior and a Random-Walk MH sampler.\n"
        "Import it and call run_random_walk_mh(Y) with your observed data matrix."
    )
