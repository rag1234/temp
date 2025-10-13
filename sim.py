import numpy as np
import pandas as pd
from arch import arch_model

def fit_arma_garch(pnl_list,
                   arma_order=(1, 0),
                   garch_pq=(1, 1),
                   dist="t",
                   vol="GARCH"):
    y = -np.asarray(pnl_list, dtype=float)
    y = pd.Series(y - np.mean(y))
    mean_type = 'ARX' if arma_order[0] > 0 else 'Constant'
    am = arch_model(y, mean=mean_type,
                    lags=arma_order[0] if arma_order[0] > 0 else 0,
                    vol=vol, p=garch_pq[0], q=garch_pq[1],
                    dist=dist, rescale=True)
    res = am.fit(disp="off")
    return am, res


def simulate_aggregated_losses(fitted_res, horizon_days, n_paths=100_000, seed=42):
    rng = np.random.default_rng(seed)
    am = fitted_res.model
    params = fitted_res.params
    sim = am.simulate(params, nobs=horizon_days, repetitions=n_paths, random_state=rng)
    return sim['data'].sum(axis=0)


def var_quantile_and_ci(samples, q=0.999, ci=0.9, n_boot=500, seed=1):
    rng = np.random.default_rng(seed)
    point = np.quantile(samples, q)
    idx = np.arange(samples.size)
    boots = [np.quantile(samples[rng.choice(idx, size=samples.size, replace=True)], q)
             for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return point, (lo, hi)


def economic_capital_projection(pnl_list,
                                horizons=[63, 126],
                                arma_order=(1, 0),
                                garch_pq=(1, 1),
                                vol="GARCH",
                                dist="t",
                                q=0.999,
                                n_paths=200_000):
    am, res = fit_arma_garch(pnl_list, arma_order, garch_pq, dist, vol)
    results = {}

    # --- 1-day VaR from GARCH (model-implied)
    agg_1d = simulate_aggregated_losses(res, 1, n_paths=n_paths, seed=99)
    VaR_1d, _ = var_quantile_and_ci(agg_1d, q=q, ci=0.9, n_boot=100)

    for i, H in enumerate(horizons):
        agg_losses = simulate_aggregated_losses(res, H, n_paths=n_paths, seed=100 + i)
        var_point, ci_mc = var_quantile_and_ci(agg_losses, q=q, ci=0.9, n_boot=500, seed=200 + i)

        # Scaled VaR via sqrt(h)
        var_scaled = VaR_1d * np.sqrt(H)
        ratio = var_point / var_scaled

        results[f"{H}_days"] = {
            "VaR_model": var_point,
            "VaR_scaled": var_scaled,
            "Ratio_model_to_scaled": ratio,
            "CI_90%": ci_mc
        }

    return results, VaR_1d, res.summary().as_text()


# -----------------------
# EXAMPLE USAGE
# -----------------------
if __name__ == "__main__":
    np.random.seed(0)
    pnl_list = np.random.normal(0, 1, 250)
    horizons = [21, 63, 126, 252]

    results, VaR_1d, summary = economic_capital_projection(
        pnl_list,
        horizons=horizons,
        arma_order=(1, 0),
        garch_pq=(1, 1),
        dist="t",
        q=0.999,
        n_paths=100_000
    )

    print(summary)
    print(f"\nOne-day 99.9% VaR: {VaR_1d:.2f}\n")
    print("--- Multi-horizon Comparison ---")
    print(f"{'Horizon':>10} | {'Model VaR':>12} | {'Scaled VaR':>12} | {'Ratio (Model/Scaled)':>20} | {'CI_90%':>15}")
    print("-"*80)
    for h, vals in results.items():
        print(f"{h:>10} | {vals['VaR_model']:12.2f} | {vals['VaR_scaled']:12.2f} | {vals['Ratio_model_to_scaled']:20.3f} | {vals['CI_90%']}")
