# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from scipy import stats
from scipy.ndimage import zoom
from scipy.interpolate import griddata
from matplotlib import cm

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------------------------
# Helpers
# ---------------------------
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64

# ---------------------------
# Models + transforms
# ---------------------------
def generate_system(model, params, N, xmin, xmax):
    if model in ("power", "logarithmic"):
        xmin = max(xmin, 1e-8)
    x = np.linspace(xmin, xmax, N)
    if model == "linear":
        A, B = params
        y = A * x + B
    elif model == "exponential":
        lam, mu = params
        y = lam * np.exp(mu * x)
    elif model == "power":
        a, b = params
        y = a * (x ** b)
    elif model == "logarithmic":
        a, b = params
        y = a * np.log(x) + b
    else:
        raise ValueError("Unknown model")
    return x, y

def apply_transform(model, x, y):
    if model == "linear":
        return x.copy(), y.copy(), np.ones_like(x, dtype=bool)
    elif model == "exponential":
        mask = y > 0
        return x[mask].copy(), np.log(y[mask]), mask
    elif model == "power":
        mask = (x > 0) & (y > 0)
        return np.log(x[mask]), np.log(y[mask]), mask
    elif model == "logarithmic":
        mask = x > 0
        return np.log(x[mask]), y[mask].copy(), mask
    else:
        raise ValueError("Unknown model")

def inverse_params(model, b0, b1):
    if model == "linear":
        return {"A": b1, "B": b0}
    elif model == "exponential":
        return {"lambda": float(np.exp(b0)), "mu": float(b1)}
    elif model == "power":
        return {"a": float(np.exp(b0)), "b": float(b1)}
    elif model == "logarithmic":
        return {"a": float(b1), "b": float(b0)}
    else:
        raise ValueError("Unknown model")

def linear_regression(x, y):
    if len(x) < 2:
        return 0.0, 0.0, 0.0
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return intercept, slope, float(r_value**2)

def compute_errors(true_params, est_params):
    abs_err = {}
    rel_err = {}
    for k in est_params:
        t = true_params.get(k, None)
        e = est_params[k]
        if t is None:
            continue
        abs_e = float(np.abs(t - e))
        abs_err[k] = abs_e
        rel_err[k] = float(100.0 * abs_e / (np.abs(t) + 1e-12))
    return abs_err, rel_err

# ---------------------------
# Window selection utilities
# ---------------------------
def evaluate_windows(xt, yt, mode="transformed"):
    # xt, yt are arrays after transform if mode == "transformed"
    # if mode == "original", xt, yt are original x and transformed y (or original y depending)
    n = len(xt)
    windows = [1.0, 0.8, 0.6, 0.4]
    results = {}
    best = {"ratio": None, "b0": 0.0, "b1": 0.0, "r2": -1.0}
    for w in windows:
        m = max(2, int(np.round(n * w)))
        # pick last m points
        xs = xt[-m:]
        ys = yt[-m:]
        b0, b1, r2 = linear_regression(xs, ys)
        results[f"{int(w*100)}%"] = float(r2)
        if r2 > best["r2"]:
            best = {"ratio": f"{int(w*100)}%", "b0": b0, "b1": b1, "r2": r2}
    return best, results

# ---------------------------
# plotting helpers
# ---------------------------
def plot_system(x, y):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(x, y, lw=2)
    ax.set_title("True System")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2)
    return fig

def plot_hist(noise, bins):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(noise, bins=bins, edgecolor="k", alpha=0.7)
    ax.set_title("Noise Histogram")
    ax.grid(alpha=0.2)
    return fig

def plot_system_noise(x, y_true, y_noisy):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(x, y_true, color="black", lw=1.5, label="True")
    ax.scatter(x, y_noisy, s=14, alpha=0.8, label="Noisy")
    ax.legend()
    ax.set_title("System + Noise")
    ax.grid(alpha=0.2)
    return fig

def plot_fit(x, y_noisy, y_fit):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(x, y_noisy, s=12, color="gray", label="Noisy")
    ax.plot(x, y_fit, lw=2, label="Fit")
    ax.legend()
    ax.set_title("Regression Fit")
    ax.grid(alpha=0.2)
    return fig

def plot_heatmap(matrix, Ns_display, sigmas_display):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    # display matrix with axis ticks based on display arrays
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="jet",
                   extent=[sigmas_display[0], sigmas_display[-1], Ns_display[0], Ns_display[-1]])
    ax.set_xlabel("σ")
    ax.set_ylabel("N")
    ax.set_title("Relative error heatmap (%)")
    fig.colorbar(im, ax=ax, label="Rel error (%)")
    return fig

# ---------------------------
# Flask routes
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run", methods=["POST"])
def run_one():
    data = request.json
    model = data.get("model", "linear")
    a = float(data.get("a", 1.0))
    b = float(data.get("b", 0.0))
    xmin = float(data.get("xmin", 0.0))
    xmax = float(data.get("xmax", 10.0))
    N = int(data.get("N", 50))
    sigma = float(data.get("sigma", 0.1))
    bins = int(data.get("bins", 20))
    noise_type = data.get("noise_type", "gaussian")
    do_fit = bool(data.get("fit", True))
    window_method = data.get("window_method", "transformed")  # 'original' or 'transformed'

    # map params
    if model == "linear":
        true_params = {"A": a, "B": b}
    elif model == "exponential":
        true_params = {"lambda": a, "mu": b}
    else:
        true_params = {"a": a, "b": b}

    # generate system + noise
    x, y_true = generate_system(model, (a,b), N, xmin, xmax)
    if noise_type == "gaussian":
        noise = np.random.normal(0, sigma, size=N)
    else:
        noise = np.random.uniform(-sigma, sigma, size=N)
    y_noisy = y_true + noise

    # compute SNR (dB)
    snr = 20.0 * np.log10((np.linalg.norm(y_true) + 1e-12) / (np.linalg.norm(noise) + 1e-12))

    # Figures
    fig_sys = plot_system(x, y_true)
    fig_hist = plot_hist(noise, bins)
    fig_sys_noise = plot_system_noise(x, y_true, y_noisy)

    # Regression
    est_params = {k: 0.0 for k in true_params.keys()}
    r2 = 0.0
    best_window_info = {}
    r2_results = {}
    y_fit = np.zeros_like(x)

    if do_fit:
        # Prepare transformed arrays depending on model
        xt_all, yt_all, mask_all = apply_transform(model, x, y_noisy)

        # If user chose original-window method, use original X spacing:
        if window_method == "original":
            # we must transform y as per model's Y-transform, but window is chosen on original x
            # obtain x_for_window (original) and y_for_reg (transformed y)
            if model == "linear":
                x_for_window = x
                y_for_reg = y_noisy
                xt = x_for_window.copy()
                yt = y_for_reg.copy()
            else:
                # for non-linear models, window selection is on original x but regression uses transformed data
                x_for_window = x
                # compute transformed y with noisy observations but keep x_for_window for slicing
                xt_tmp, yt_tmp, mask_tmp = apply_transform(model, x, y_noisy)
                # Since transformed arrays exclude invalid points, map slices by indices in mask
                indices = np.nonzero(mask_tmp)[0]
                # We'll build arrays aligned to indices for sliding windows based on original x position
                # Use indices for last-m selection
                xt = xt_tmp.copy()
                yt = yt_tmp.copy()
            # Evaluate windows on xt,yt (which are transformed y values)
            best, r2_results = evaluate_windows(xt, yt, mode="original")
            b0, b1, r2 = best["b0"], best["b1"], best["r2"]
            best_window_info = best
        else:
            # transformed window selection (recommended)
            xt, yt, mask = apply_transform(model, x, y_noisy)
            if len(xt) >= 2:
                best, r2_results = evaluate_windows(xt, yt, mode="transformed")
                b0, b1, r2 = best["b0"], best["b1"], best["r2"]
                best_window_info = best
            else:
                b0, b1, r2 = 0.0, 0.0, 0.0
                best_window_info = {"ratio": "N/A", "r2": r2}
        # inverse transform params
        est_params = inverse_params(model, b0, b1)
        # reconstruct fitted curve in original domain
        if model == "linear":
            y_fit = est_params["A"] * x + est_params["B"]
        elif model == "exponential":
            y_fit = est_params["lambda"] * np.exp(est_params["mu"] * x)
        elif model == "power":
            x_pos = np.maximum(x, 1e-8)
            y_fit = est_params["a"] * (x_pos ** est_params["b"])
        elif model == "logarithmic":
            x_pos = np.maximum(x, 1e-8)
            y_fit = est_params["a"] * np.log(x_pos) + est_params["b"]

    fig_fit = plot_fit(x, y_noisy, y_fit)

    # errors
    abs_err, rel_err = compute_errors(true_params, est_params)

    return jsonify({
        "system": fig_to_base64(fig_sys),
        "hist": fig_to_base64(fig_hist),
        "system_noise": fig_to_base64(fig_sys_noise),
        "fit": fig_to_base64(fig_fit),
        "estimated": est_params,
        "r2": r2,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "snr": float(snr),
        "best_window": best_window_info,
        "r2_values": r2_results
    })


@app.route("/grid", methods=["POST"])
def run_grid():
    data = request.json
    model = data.get("model", "linear")
    a = float(data.get("a", 1.0))
    b = float(data.get("b", 0.0))
    xmin = float(data.get("xmin", 0.0))
    xmax = float(data.get("xmax", 10.0))
    Ns = data.get("Ns", [10, 50, 100])
    sigmas = data.get("sigmas", [0.1, 0.5, 1.0])
    noise_type = data.get("noise_type", "gaussian")
    trials = int(data.get("trials", 10))
    zoom_factor = int(data.get("zoom", 1))
    window_method = data.get("window_method", "transformed")

    Ns = [int(n) for n in Ns]
    sigmas = [float(s) for s in sigmas]

    matrix = np.zeros((len(Ns), len(sigmas)))

    # compute coarse matrix
    for iN, N in enumerate(Ns):
        for jS, s in enumerate(sigmas):
            errs = []
            for t in range(trials):
                x, y_true = generate_system(model, (a,b), N, xmin, xmax)
                if noise_type == "gaussian":
                    noise = np.random.normal(0, s, size=N)
                else:
                    noise = np.random.uniform(-s, s, size=N)
                y_noisy = y_true + noise
                # transform
                xt, yt, mask = apply_transform(model, x, y_noisy)
                if len(xt) >= 2:
                    # window selection
                    if window_method == "original":
                        # for simplicity choose window on original x by using transformed arrays
                        best, _ = evaluate_windows(xt, yt, mode="original")
                    else:
                        best, _ = evaluate_windows(xt, yt, mode="transformed")
                    est = inverse_params(model, best["b0"], best["b1"])
                    # choose first parameter for matrix (first key)
                    true_params = {}
                    if model == "linear":
                        true_params = {"A": a, "B": b}
                    elif model == "exponential":
                        true_params = {"lambda": a, "mu": b}
                    else:
                        true_params = {"a": a, "b": b}
                    first_key = list(true_params.keys())[0]
                    rel = float(100.0 * np.abs(true_params[first_key] - est[first_key]) / (np.abs(true_params[first_key]) + 1e-12))
                    errs.append(rel)
                else:
                    errs.append(100.0)
            matrix[iN, jS] = float(np.mean(errs))

    # prepare fine grid for interpolation (CUBIC)
    # we will build a scattered set of points and interpolate with cubic griddata
    NN, SS = np.meshgrid(Ns, sigmas, indexing='ij')
    points = np.vstack([NN.ravel(), SS.ravel()]).T
    values = matrix.ravel()

    # fine grid coordinates
    n_fine = (len(Ns)-1) * zoom_factor + 1
    s_fine = (len(sigmas)-1) * zoom_factor + 1
    Ns_display = np.linspace(Ns[0], Ns[-1], n_fine)
    sigmas_display = np.linspace(sigmas[0], sigmas[-1], s_fine)
    NNf, SSf = np.meshgrid(Ns_display, sigmas_display, indexing='ij')
    grid_points = np.vstack([NNf.ravel(), SSf.ravel()]).T

    # attempt cubic interpolation; fallback to linear if cubic fails
    try:
        interp_vals = griddata(points, values, grid_points, method='cubic')
        # griddata may return NaN where cubic is not defined; fill those with linear
        mask_nan = np.isnan(interp_vals)
        if mask_nan.any():
            interp_linear = griddata(points, values, grid_points, method='linear')
            interp_vals[mask_nan] = interp_linear[mask_nan]
        matrix_fine = interp_vals.reshape(NNf.shape)
        interp_method_used = "cubic"
    except Exception:
        interp_vals = griddata(points, values, grid_points, method='linear')
        matrix_fine = interp_vals.reshape(NNf.shape)
        interp_method_used = "linear"

    # Evaluate interpolation error by computing "ground truth" at a subset of fine grid
    # We'll compute actual matrix at a small random subset of fine grid points (to save time)
    # choose up to 20 validation points
    rng = np.random.default_rng(12345)
    num_validate = min(20, NNf.size)
    idx = rng.choice(np.arange(NNf.size), size=num_validate, replace=False)
    real_vals = []
    interp_vals_at_points = []
    for k in idx:
        N_val = int(round(NNf.ravel()[k]))
        s_val = float(SSf.ravel()[k])
        # compute true by running few trials (3) to estimate
        errs = []
        for tr in range(3):
            x, y_true = generate_system(model, (a,b), N_val, xmin, xmax)
            if noise_type == "gaussian":
                noise = np.random.normal(0, s_val, size=N_val)
            else:
                noise = np.random.uniform(-s_val, s_val, size=N_val)
            y_noisy = y_true + noise
            xt, yt, mask = apply_transform(model, x, y_noisy)
            if len(xt) >= 2:
                best, _ = evaluate_windows(xt, yt, mode=window_method)
                est = inverse_params(model, best["b0"], best["b1"])
                # first parameter error
                if model == "linear":
                    true_first = a
                elif model == "exponential":
                    true_first = a
                else:
                    true_first = a
                rel = float(100.0 * np.abs(true_first - est[list(est.keys())[0]]) / (np.abs(true_first) + 1e-12))
                errs.append(rel)
            else:
                errs.append(100.0)
        real_vals.append(np.mean(errs))
        interp_vals_at_points.append(matrix_fine.ravel()[k])

    real_vals = np.array(real_vals)
    interp_vals_at_points = np.array(interp_vals_at_points)
    interp_error_rmse = float(np.sqrt(np.mean((real_vals - interp_vals_at_points)**2)))
    interp_error_mean = float(np.mean(np.abs(real_vals - interp_vals_at_points)))

    # produce heatmap image
    fig_heat = plot_heatmap(matrix_fine, Ns_display, sigmas_display)
    heat_b64 = fig_to_base64(fig_heat)

    return jsonify({
        "heatmap": heat_b64,
        "matrix": matrix.tolist(),
        "Ns": Ns,
        "sigmas": sigmas,
        "interp_method": interp_method_used,
        "interp_rmse": interp_error_rmse,
        "interp_mean_abs": interp_error_mean
    })


if __name__ == "__main__":
    app.run(debug=True)
