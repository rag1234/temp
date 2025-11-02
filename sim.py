import numpy as np
import pandas as pd
import math

def _kth_order_term(dX: np.ndarray, tensor: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the k-th order Taylor contribution for all time rows in dX.

    dX     : shape (T, F)  factor moves for each date
    tensor : sensitivities for this order k
             - shape (F,)           => diagonal-only for this order
             - shape (F,)*k         => full cross tensor for this order
    k      : integer order

    Returns
    -------
    term : shape (T,) P&L contribution from this order
    """
    T, F = dX.shape
    coeff = 1.0 / math.factorial(k)
    tensor = np.asarray(tensor, dtype=float)

    # Case A: diagonal-only vector
    if tensor.ndim == 1:
        if tensor.shape[0] != F:
            raise ValueError(
                f"Order {k}: diagonal vector length {tensor.shape[0]} "
                f"!= num factors {F}"
            )
        # P_n(t) = 1/k! * sum_i tensor[i] * (dX[t,i])^k
        return coeff * np.einsum("tf,f->t", (dX ** k), tensor)

    # Case B: full tensor of rank k
    expected_shape = (F,) * k
    if tensor.shape != expected_shape:
        raise ValueError(
            f"Order {k}: tensor must have shape {expected_shape}, got {tensor.shape}"
        )

    # Build einsum string dynamically.
    #
    # For k=2:
    #   einsum('ti,tj,ij->t', dX, dX, tensor)
    #
    # For k=3:
    #   einsum('ti,tj,tk,ijk->t', dX, dX, dX, tensor)
    #
    # General:
    #   'ti,tj,tk,...,ijk...->t'
    idx_letters_pool = list("ijklmnopqrstuvwxyzabcdefgh")  # enough distinct indices
    idx_letters = idx_letters_pool[:k]  # e.g. ['i','j','k'] for k=3

    dX_terms = [f"t{ix}" for ix in idx_letters]     # ['ti','tj','tk',...]
    tens_term = "".join(idx_letters)                # 'ijk...'

    eins_lhs = ",".join(dX_terms + [tens_term])     # 'ti,tj,tk,ijk'
    eins_rhs = "t"
    eins_sub = f"{eins_lhs}->{eins_rhs}"

    args = [dX] * k + [tensor]

    return coeff * np.einsum(eins_sub, *args)


def taylor_pnl_general(
    moves_df: pd.DataFrame,
    risk_dict: dict,
    order_map: dict
) -> pd.DataFrame:
    """
    General n-th order Taylor P&L expansion with explicit order mapping.

    Parameters
    ----------
    moves_df : pd.DataFrame
        Index = dates, columns = risk factors.
        Values = Δx for each factor on each date.
        Shape (T, F).

    risk_dict : dict
        Dict of sensitivities. Each key is a label for a derivative block.
        Each value is either:
          - 1D array length F      (diagonal terms only for that order)
          - full tensor of shape (F, F, ..., F) with k repeats (full cross terms)
        IMPORTANT: same factor ordering as moves_df.columns.

        Example:
        {
            "first_derivative":  [delta_RF1, delta_RF2, ...],
            "second_derivative": [[gamma_11, gamma_12, ...],
                                   ...                     ],
            "third_derivative":  [skew_RF1, skew_RF2, ...]
        }

    order_map : dict
        Maps each key in risk_dict -> integer order k.
        Example:
        {
            "first_derivative": 1,
            "second_derivative": 2,
            "third_derivative": 3,
        }

        This tells us how to scale with 1/k!, and what power
        of dX to use.

    Returns
    -------
    pd.DataFrame
        Columns: one per order ("order_1", "order_2", ...), plus "total".
        Index  : same as moves_df.index

        Each column order_k is the P&L contribution from exactly that order.
        "total" is the sum across all provided orders.
    """

    # factor moves matrix
    dX = moves_df.to_numpy()  # shape (T, F)
    T, F = dX.shape

    # Sanity: all keys in risk_dict must exist in order_map
    missing = [k for k in risk_dict.keys() if k not in order_map]
    if missing:
        raise ValueError(f"order_map missing entries for keys: {missing}")

    # We'll accumulate PnL per *order k*
    pnl_by_order = {}  # k -> np.array shape (T,)

    for risk_key, sens in risk_dict.items():
        k = int(order_map[risk_key])

        term_vec = _kth_order_term(dX, np.asarray(sens, dtype=float), k)

        # If multiple blocks map to same k, sum them (rare but allowed)
        if k in pnl_by_order:
            pnl_by_order[k] = pnl_by_order[k] + term_vec
        else:
            pnl_by_order[k] = term_vec

    # Build output dataframe
    out_cols = {}
    total = np.zeros(T)

    for k in sorted(pnl_by_order.keys()):
        col_name = f"order_{k}"
        out_cols[col_name] = pnl_by_order[k]
        total += pnl_by_order[k]

    out_cols["total"] = total

    return pd.DataFrame(out_cols, index=moves_df.index)
