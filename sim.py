import numpy as np
import pandas as pd
import math

def taylor_pnl(
    moves_df: pd.DataFrame,
    risk_dict: dict
) -> pd.DataFrame:
    """
    Compute P&L per date using Taylor expansion up to Nth order.

    Parameters
    ----------
    moves_df : pd.DataFrame
        Index = dates, columns = risk factors.
        Values = actual factor moves Δx_i for that date.
        Column order defines factor index order.

    risk_dict : dict
        Keys like "first_order", "second_order", ..., "n_order".
        Values are sensitivities for that order:
            order 1: shape (F,)
            order 2: shape (F,F)  or (F,) for diagonal-only
            order k: shape (F,)*k or (F,) for diagonal-only kth-order term.
        Assumed to be w.r.t. the SAME factor ordering as moves_df.columns.

    Returns
    -------
    pd.DataFrame
        Columns:
          - 'order_1', 'order_2', ..., 'total'
        Index = same as moves_df.index
    """

    # --- prep
    factors = list(moves_df.columns)
    dX = moves_df.to_numpy()              # shape (T, F)
    T, F = dX.shape

    # We'll accumulate per-order pnl in a dict of arrays shape (T,)
    pnl_terms = {}
    total_pnl = np.zeros(T)

    # helper: safe factorial coeff
    def coeff(k: int) -> float:
        return 1.0 / math.factorial(k)

    # iterate orders present in risk_dict
    # We'll parse keys like "first_order","second_order","third_order",...
    for key, sens in risk_dict.items():
        # infer order k from key
        # we'll strip "_order" and map 'first','second','third','fourth',...
        # fallback: try to parse leading int if user uses "order_3" style
        order_name = key.lower().strip()

        # try common english ordinals first
        ordinal_map = {
            "first": 1, "second": 2, "third": 3, "fourth": 4,
            "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
            "ninth": 9, "tenth": 10
        }
        k = None
        for word, val in ordinal_map.items():
            if order_name.startswith(word):
                k = val
                break
        if k is None:
            # fallback patterns like "order_3", "3rd_order", "thirdorder"
            import re
            m = re.search(r'(\d+)', order_name)
            if m:
                k = int(m.group(1))
        if k is None:
            raise ValueError(f"Cannot infer order from key '{key}'")

        S = np.array(sens, dtype=float)

        # compute kth order pnl term for all T days ->
        # term_t = (1/k!) * sum_{i1..ik} S[i1..ik] * dX[t,i1]*...*dX[t,ik]

        if S.ndim == 1:
            # interpret as diagonal-only for that order:
            # term_t = (1/k!) * sum_i S[i] * (dX[t,i])^k
            # So you gave only the pure self term per factor.
            if S.shape[0] != F:
                raise ValueError(f"{key}: diagonal vector length {S.shape[0]} "
                                 f"!= num factors {F}")
            term = coeff(k) * np.sum(
                (S * (dX ** k)), axis=1
            )  # shape (T,)
        else:
            # full tensor case
            # Example:
            # k=2 -> S shape (F,F)
            # k=3 -> S shape (F,F,F)
            # etc.
            # We'll use einsum:
            # For k=2: term_t = 1/2 * Σ_ij S_ij dX_ti dX_tj
            # -> einsum('t i, i j, t j -> t', dX, S, dX)
            #
            # For k=3: term_t = 1/6 * Σ_ijk S_ijk dX_ti dX_tj dX_tk
            # -> einsum('t i, t j, t k, i j k -> t', dX, dX, dX, S)
            #
            # General k: einsum over: k copies of dX plus S.

            if S.shape != (F,) * k:
                raise ValueError(
                    f"{key}: tensor shape {S.shape} does not match "
                    f"(num_factors,)*{k} = {(F,)*k}"
                )

            # build einsum subscripts programmatically
            # We'll create something like:
            # inputs:  ['t i', 't j', 't k', 'i j k']
            # output:  't'
            idx_letters = list("ijklmnopqrstuvwxyzabcdefgh")  # enough distinct indices
            idx = idx_letters[:k]  # e.g. ['i','j','k'] for k=3

            # dX term subscripts: 't i', 't j', ...
            dX_terms = [f"t {ix}" for ix in idx]
            # S term subscripts: 'i j k'
            S_term = " ".join(idx)

            einsum_input = ",".join(dX_terms + [S_term])
            einsum_output = "t"
            subscript = f"{einsum_input}->{einsum_output}"

            # Prepare args: k copies of dX plus S
            args = [dX] * k + [S]

            term = coeff(k) * np.einsum(subscript, *args)  # shape (T,)

        pnl_terms[f"order_{k}"] = term
        total_pnl += term

    # final output dataframe
    out = pd.DataFrame(index=moves_df.index, data=pnl_terms)
    out["total"] = total_pnl
    return out
