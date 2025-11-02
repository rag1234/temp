import numpy as np
from scipy.interpolate import CubicSpline

def interpolate_points(
    x_known: np.ndarray,
    y_known: np.ndarray,
    x_query: np.ndarray,
    interp_method: str = "linear",    # "linear" or "cubic"
    extrap_method: str = "linear",    # "flat", "linear", or "cubic"
):
    """
    Interpolate and extrapolate y for given x_query with independently chosen
    interpolation method (inside domain) and extrapolation method (outside domain).

    Parameters
    ----------
    x_known : np.ndarray
        Known x points (1D, strictly increasing).
    y_known : np.ndarray
        Known y values at x_known.
    x_query : np.ndarray
        Target x points to evaluate.
    interp_method : str
        "linear" -> piecewise linear between knots
        "cubic"  -> natural cubic spline between knots
    extrap_method : str
        "flat"   -> clamp to endpoint y
        "linear" -> extend endpoint slope linearly
        "cubic"  -> use cubic spline's natural extrapolation

    Returns
    -------
    y_out : np.ndarray
        Interpolated/extrapolated values at x_query.
    extrap_mask : np.ndarray
        Boolean array marking which x_query points were extrapolated.
    """

    x_known = np.asarray(x_known, dtype=float)
    y_known = np.asarray(y_known, dtype=float)
    x_query = np.asarray(x_query, dtype=float)

    # basic validation
    if x_known.ndim != 1 or y_known.ndim != 1:
        raise ValueError("x_known and y_known must be 1D.")
    if x_known.shape[0] != y_known.shape[0]:
        raise ValueError("x_known and y_known must have same length.")
    if not np.all(np.diff(x_known) > 0):
        raise ValueError("x_known must be strictly increasing.")

    # masks
    left_mask = x_query < x_known[0]
    right_mask = x_query > x_known[-1]
    inside_mask = ~(left_mask | right_mask)
    extrap_mask = ~inside_mask

    # prepare storage
    y_out = np.empty_like(x_query, dtype=float)

    # --- build spline once if we'll need cubic anywhere ---
    use_cubic = (interp_method == "cubic") or (extrap_method == "cubic")
    cs = None
    if use_cubic:
        cs = CubicSpline(
            x_known,
            y_known,
            bc_type="natural",
            extrapolate=True  # allow cubic to speak outside too
        )

    # -----------------------
    # 1. INTERPOLATION inside
    # -----------------------
    if np.any(inside_mask):
        xin = x_query[inside_mask]

        if interp_method == "linear":
            # np.interp will only use linear between bracketing knots
            y_in = np.interp(xin, x_known, y_known)

        elif interp_method == "cubic":
            # cubic spline inside range
            y_in = cs(xin)

        else:
            raise ValueError("interp_method must be 'linear' or 'cubic'.")

        y_out[inside_mask] = y_in

    # -----------------------
    # 2. EXTRAPOLATION outside
    # -----------------------
    # LEFT side
    if np.any(left_mask):
        xl = x_query[left_mask]

        if extrap_method == "flat":
            y_left = np.full_like(xl, y_known[0], dtype=float)

        elif extrap_method == "linear":
            slope_left = (y_known[1] - y_known[0]) / (x_known[1] - x_known[0])
            y_left = y_known[0] + slope_left * (xl - x_known[0])

        elif extrap_method == "cubic":
            # natural spline extrapolation to the left
            y_left = cs(xl)

        else:
            raise ValueError("extrap_method must be 'flat', 'linear', or 'cubic'.")

        y_out[left_mask] = y_left

    # RIGHT side
    if np.any(right_mask):
        xr = x_query[right_mask]

        if extrap_method == "flat":
            y_right = np.full_like(xr, y_known[-1], dtype=float)

        elif extrap_method == "linear":
            slope_right = ((y_known[-1] - y_known[-2]) /
                           (x_known[-1] - x_known[-2]))
            y_right = y_known[-1] + slope_right * (xr - x_known[-1])

        elif extrap_method == "cubic":
            y_right = cs(xr)

        else:
            raise ValueError("extrap_method must be 'flat', 'linear', or 'cubic'.")

        y_out[right_mask] = y_right

    return y_out, extrap_mask
