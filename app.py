from flask import Flask, render_template, request, jsonify, send_file, make_response
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import base64
from scipy import stats
from flask import session
from scipy.interpolate import griddata

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "param-estimator-secret"

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

def round_sig(x, sig=3):
    try:
        return float(f"{x:.{sig}g}")
    except Exception:
        return x

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
    n = len(xt)
    windows = [1.0, 0.8, 0.6, 0.4]
    results = {}
    best = {"ratio": None, "b0": 0.0, "b1": 0.0, "r2": -1.0}
    for w in windows:
        m = max(2, int(np.round(n * w)))
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

# @app.route("/run", methods=["POST"])
# def run_one():
#     data = request.json
#     model = data.get("model", "linear")
#     a = float(data.get("a", 1.0))
#     b = float(data.get("b", 0.0))
#     xmin = float(data.get("xmin", 0.0))
#     xmax = float(data.get("xmax", 10.0))
#     N = int(data.get("N", 50))
#     sigma = float(data.get("sigma", 0.1))
#     bins = int(data.get("bins", 20))
#     noise_type = data.get("noise_type", "gaussian")
#     do_fit = bool(data.get("fit", True))
#     window_method = data.get("window_method", "transformed")

#     # map params
#     if model == "linear":
#         true_params = {"A": a, "B": b}
#     elif model == "exponential":
#         true_params = {"lambda": a, "mu": b}
#     else:
#         true_params = {"a": a, "b": b}

#     # generate system + single noise vector (for this run)
#     x, y_true = generate_system(model, (a, b), N, xmin, xmax)
#     if noise_type == "gaussian":
#         noise = np.random.normal(0, sigma, size=N)
#     else:
#         noise = np.random.uniform(-sigma, sigma, size=N)
#     y_noisy = y_true + noise

#     # compute SNR (dB)
#     snr = 20.0 * np.log10((np.linalg.norm(y_true) + 1e-12) / (np.linalg.norm(noise) + 1e-12))

#     # Figures
#     fig_sys = plot_system(x, y_true)
#     fig_hist = plot_hist(noise, bins)
#     fig_sys_noise = plot_system_noise(x, y_true, y_noisy)

#     # Regression
#     est_params = {k: 0.0 for k in true_params.keys()}
#     r2 = 0.0
#     best_window_info = {}
#     r2_results = {}
#     y_fit = np.zeros_like(x)

#     if do_fit:
#         xt_all, yt_all, mask_all = apply_transform(model, x, y_noisy)

#         if window_method == "original":
#             if model == "linear":
#                 xt = x.copy()
#                 yt = y_noisy.copy()
#             else:
#                 xt_tmp, yt_tmp, mask_tmp = apply_transform(model, x, y_noisy)
#                 xt = xt_tmp.copy()
#                 yt = yt_tmp.copy()
#             best, r2_results = evaluate_windows(xt, yt, mode="original")
#             b0, b1, r2 = best["b0"], best["b1"], best["r2"]
#             best_window_info = best
#         else:
#             xt, yt, mask = apply_transform(model, x, y_noisy)
#             if len(xt) >= 2:
#                 best, r2_results = evaluate_windows(xt, yt, mode="transformed")
#                 b0, b1, r2 = best["b0"], best["b1"], best["r2"]
#                 best_window_info = best
#             else:
#                 b0, b1, r2 = 0.0, 0.0, 0.0
#                 best_window_info = {"ratio": "N/A", "r2": r2}

#         est_params = inverse_params(model, b0, b1)

#         # reconstruct fitted curve
#         if model == "linear":
#             y_fit = est_params["A"] * x + est_params["B"]
#         elif model == "exponential":
#             y_fit = est_params["lambda"] * np.exp(est_params["mu"] * x)
#         elif model == "power":
#             x_pos = np.maximum(x, 1e-8)
#             y_fit = est_params["a"] * (x_pos ** est_params["b"])
#         elif model == "logarithmic":
#             x_pos = np.maximum(x, 1e-8)
#             y_fit = est_params["a"] * np.log(x_pos) + est_params["b"]

#     fig_fit = plot_fit(x, y_noisy, y_fit)

#     # errors and rounding to 3 significant digits
#     abs_err, rel_err = compute_errors(true_params, est_params)
#     est_params_rounded = {k: round_sig(v, 3) for k, v in est_params.items()}
#     abs_err_rounded = {k: round_sig(v, 3) for k, v in abs_err.items()}
#     rel_err_rounded = {k: round_sig(v, 3) for k, v in rel_err.items()}

#     return jsonify({
#         "system": fig_to_base64(fig_sys),
#         "hist": fig_to_base64(fig_hist),
#         "system_noise": fig_to_base64(fig_sys_noise),
#         "fit": fig_to_base64(fig_fit),
#         "estimated": est_params_rounded,
#         "r2": r2,
#         "abs_err": abs_err_rounded,
#         "rel_err": rel_err_rounded,
#         "snr": float(snr),
#         "best_window": best_window_info,
#         "r2_values": r2_results
#     })

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
    window_method = data.get("window_method", "transformed")

    # true params
    if model == "linear":
        true_params = {"A": a, "B": b}
    elif model == "exponential":
        true_params = {"lambda": a, "mu": b}
    else:
        true_params = {"a": a, "b": b}

    # system
    x, y_true = generate_system(model, (a, b), N, xmin, xmax)

    # ---- NOISE (GENERATE ONCE) ----
    if noise_type == "gaussian":
        noise = np.random.normal(0, sigma, size=N)
    else:
        noise = np.random.uniform(-sigma, sigma, size=N)

    # ✅ STORE NOISE IN SESSION
    session["last_noise"] = noise.tolist()
    session["last_noise_meta"] = {
        "N": N,
        "sigma": sigma,
        "noise_type": noise_type
    }

    y_noisy = y_true + noise

    # SNR
    snr = 20.0 * np.log10(
        (np.linalg.norm(y_true) + 1e-12) /
        (np.linalg.norm(noise) + 1e-12)
    )

    # plots
    fig_sys = plot_system(x, y_true)
    fig_hist = plot_hist(noise, bins)
    fig_sys_noise = plot_system_noise(x, y_true, y_noisy)

    est_params = {}
    abs_err = {}
    rel_err = {}
    r2 = 0.0
    best_window_info = {}
    r2_results = {}
    y_fit = np.zeros_like(x)

    if do_fit:
        xt, yt, _ = apply_transform(model, x, y_noisy)
        if len(xt) >= 2:
            best, r2_results = evaluate_windows(xt, yt, window_method)
            b0, b1 = best["b0"], best["b1"]
            r2 = best["r2"]
            best_window_info = best
            est_params = inverse_params(model, b0, b1)

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

    abs_err, rel_err = compute_errors(true_params, est_params)

    return jsonify({
        "system": fig_to_base64(fig_sys),
        "hist": fig_to_base64(fig_hist),
        "system_noise": fig_to_base64(fig_sys_noise),
        "fit": fig_to_base64(fig_fit),
        "estimated": {k: round_sig(v, 3) for k, v in est_params.items()},
        "abs_err": {k: round_sig(v, 3) for k, v in abs_err.items()},
        "rel_err": {k: round_sig(v, 3) for k, v in rel_err.items()},
        "r2": r2,
        "snr": float(snr),
        "best_window": best_window_info,
        "r2_values": r2_results
    })


# @app.route("/export_pdf", methods=["POST"])
# def export_pdf():
#     # Accept the same payload as /run to create a small PDF report for the current parameters.
#     data = request.json or {}
#     model = data.get("model", "linear")
#     a = float(data.get("a", 1.0))
#     b = float(data.get("b", 0.0))
#     xmin = float(data.get("xmin", 0.0))
#     xmax = float(data.get("xmax", 10.0))
#     N = int(data.get("N", 50))
#     sigma = float(data.get("sigma", 0.1))
#     bins = int(data.get("bins", 20))
#     noise_type = data.get("noise_type", "gaussian")
#     window_method = data.get("window_method", "transformed")
#     do_fit = bool(data.get("fit", True))

#     # reuse run_one logic to compute everything and figures
#     x, y_true = generate_system(model, (a,b), N, xmin, xmax)
#     if noise_type == "gaussian":
#         noise = np.random.normal(0, sigma, size=N)
#     else:
#         noise = np.random.uniform(-sigma, sigma, size=N)
#     y_noisy = y_true + noise

#     fig_sys = plot_system(x, y_true)
#     fig_hist = plot_hist(noise, bins)
#     fig_sys_noise = plot_system_noise(x, y_true, y_noisy)

#     est_params = {}
#     abs_err = {}
#     rel_err = {}
#     r2 = 0.0
#     y_fit = np.zeros_like(x)
#     best_window_info = {}
#     r2_results = {}

#     if do_fit:
#         xt, yt, mask = apply_transform(model, x, y_noisy)
#         if window_method == "original":
#             if model == "linear":
#                 xt = x.copy(); yt = y_noisy.copy()
#             else:
#                 xt_tmp, yt_tmp, mask_tmp = apply_transform(model, x, y_noisy)
#                 xt = xt_tmp.copy(); yt = yt_tmp.copy()
#             best, r2_results = evaluate_windows(xt, yt, mode="original")
#         else:
#             best, r2_results = evaluate_windows(xt, yt, mode="transformed")
#         b0, b1 = best["b0"], best["b1"]
#         r2 = best.get("r2", 0.0)
#         est_params = inverse_params(model, b0, b1)
#         if model == "linear":
#             y_fit = est_params["A"] * x + est_params["B"]
#         elif model == "exponential":
#             y_fit = est_params["lambda"] * np.exp(est_params["mu"] * x)
#         elif model == "power":
#             x_pos = np.maximum(x, 1e-8); y_fit = est_params["a"] * (x_pos ** est_params["b"])
#         elif model == "logarithmic":
#             x_pos = np.maximum(x, 1e-8); y_fit = est_params["a"] * np.log(x_pos) + est_params["b"]

#     fig_fit = plot_fit(x, y_noisy, y_fit)
#     true_params = {}
#     if model == "linear":
#         true_params = {"A": a, "B": b}
#     elif model == "exponential":
#         true_params = {"lambda": a, "mu": b}
#     else:
#         true_params = {"a": a, "b": b}

#     abs_err, rel_err = compute_errors(true_params, est_params)
#     est_params_rounded = {k: round_sig(v, 3) for k, v in est_params.items()}
#     abs_err_rounded = {k: round_sig(v, 3) for k, v in abs_err.items()}
#     rel_err_rounded = {k: round_sig(v, 3) for k, v in rel_err.items()}

#     # Build PDF
#     buf = BytesIO()
#     with PdfPages(buf) as pdf:
#         # page 1: plots grid
#         fig_page, axs = plt.subplots(2, 2, figsize=(8.27, 11.69))  # A4-ish
#         # True system
#         axs[0,0].plot(x, y_true, lw=2); axs[0,0].set_title("True System"); axs[0,0].grid(alpha=0.2)
#         # Noise histogram
#         axs[0,1].hist(noise, bins=bins, edgecolor="k", alpha=0.7); axs[0,1].set_title("Noise Histogram")
#         # System + Noise
#         axs[1,0].plot(x, y_true, color="black", lw=1.5); axs[1,0].scatter(x, y_noisy, s=8, alpha=0.8)
#         axs[1,0].set_title("System + Noise")
#         # Fit
#         axs[1,1].scatter(x, y_noisy, s=8, alpha=0.6); axs[1,1].plot(x, y_fit, lw=2); axs[1,1].set_title("Regression Fit")
#         plt.tight_layout()
#         pdf.savefig(fig_page)
#         plt.close(fig_page)

#         # page 2: parameter table
#         fig_table = plt.figure(figsize=(8.27, 11.69))
#         plt.axis('off')
#         txt = f"Model: {model}\nN: {N}\nx range: [{xmin}, {xmax}]\nNoise: {noise_type} (σ={sigma})\n\n"
#         txt += "Parameters (True | Estimated | Abs err | Rel err %)\n\n"
#         for k in true_params.keys():
#             t = true_params.get(k, "--")
#             e = est_params_rounded.get(k, "--")
#             ae = abs_err_rounded.get(k, "--")
#             re = rel_err_rounded.get(k, "--")
#             txt += f"{k} : {t}  |  {e}  |  {ae}  |  {re}\n"
#         plt.text(0.01, 0.99, txt, va='top', ha='left', fontfamily='monospace', fontsize=10)
#         pdf.savefig(fig_table)
#         plt.close(fig_table)

#     buf.seek(0)
#     return send_file(buf, as_attachment=True, download_name="ParamEstimator_report.pdf", mimetype="application/pdf")
@app.route("/export_pdf", methods=["POST"])
def export_pdf():
    data = request.json or {}
    model = data.get("model", "linear")
    a = float(data.get("a", 1.0))
    b = float(data.get("b", 0.0))
    xmin = float(data.get("xmin", 0.0))
    xmax = float(data.get("xmax", 10.0))
    N = int(data.get("N", 50))
    sigma = float(data.get("sigma", 0.1))
    bins = int(data.get("bins", 20))
    noise_type = data.get("noise_type", "gaussian")
    window_method = data.get("window_method", "transformed")

    # ---- REUSE STORED NOISE ----
    noise = session.get("last_noise")
    meta = session.get("last_noise_meta", {})

    if noise is None:
        return make_response("No stored noise. Run experiment first.", 400)

    if meta.get("N") != N or meta.get("sigma") != sigma:
        return make_response("Parameters changed. Re-run experiment.", 400)

    noise = np.array(noise)

    # system
    x, y_true = generate_system(model, (a, b), N, xmin, xmax)
    y_noisy = y_true + noise

    # regression
    xt, yt, _ = apply_transform(model, x, y_noisy)
    best, r2_results = evaluate_windows(xt, yt, window_method)
    b0, b1 = best["b0"], best["b1"]
    r2 = best["r2"]
    est_params = inverse_params(model, b0, b1)

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

    # true params
    if model == "linear":
        true_params = {"A": a, "B": b}
    elif model == "exponential":
        true_params = {"lambda": a, "mu": b}
    else:
        true_params = {"a": a, "b": b}

    abs_err, rel_err = compute_errors(true_params, est_params)

    # ---- PDF ----
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        fig, axs = plt.subplots(2, 2, figsize=(8.27, 11.69))
        axs[0,0].plot(x, y_true); axs[0,0].set_title("True System")
        axs[0,1].hist(noise, bins=bins); axs[0,1].set_title("Noise")
        axs[1,0].scatter(x, y_noisy, s=8); axs[1,0].set_title("System + Noise")
        axs[1,1].scatter(x, y_noisy, s=8); axs[1,1].plot(x, y_fit); axs[1,1].set_title("Fit")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig2 = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        txt = f"Model: {model}\nN: {N}\nσ: {sigma}\n\n"
        txt += "Parameters (True | Estimated | Abs err | Rel err %)\n\n"
        for k in true_params:
            txt += f"{k}: {true_params[k]} | {round_sig(est_params[k],3)} | "
            txt += f"{round_sig(abs_err[k],3)} | {round_sig(rel_err[k],3)}\n"
        plt.text(0.01, 0.99, txt, va="top", family="monospace")
        pdf.savefig(fig2)
        plt.close(fig2)

    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name="ParamEstimator_report.pdf",
                     mimetype="application/pdf")


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

    # We'll use the teacher's suggested approach:
    # For each trial we generate a full noise vector with length N_max and then take subsets for each N.
    N_max = max(Ns)

    # pre-generate system for N_max
    x_full, y_full_true = generate_system(model, (a, b), N_max, xmin, xmax)

    for jS, s in enumerate(sigmas):
        # For each sigma, run 'trials' times; in each trial generate noise_full and then reuse subsets for different N
        rels_per_N = {N: [] for N in Ns}
        for t in range(trials):
            if noise_type == "gaussian":
                noise_full = np.random.normal(0, s, size=N_max)
            else:
                noise_full = np.random.uniform(-s, s, size=N_max)
            y_full_noisy = y_full_true + noise_full

            for iN, N in enumerate(Ns):
                x = x_full[:N]
                y_true = y_full_true[:N]
                y_noisy = y_full_noisy[:N]

                xt, yt, mask = apply_transform(model, x, y_noisy)
                if len(xt) >= 2:
                    if window_method == "original":
                        best, _ = evaluate_windows(xt, yt, mode="original")
                    else:
                        best, _ = evaluate_windows(xt, yt, mode="transformed")
                    est = inverse_params(model, best["b0"], best["b1"])
                    # choose first parameter for matrix (first key)
                    if model == "linear":
                        true_first = a
                        est_first = est.get("A", np.nan)
                    elif model == "exponential":
                        true_first = a
                        est_first = est.get("lambda", np.nan)
                    else:
                        true_first = a
                        est_first = est.get(list(est.keys())[0], np.nan)
                    rel = float(100.0 * np.abs(true_first - est_first) / (np.abs(true_first) + 1e-12))
                    rels_per_N[N].append(rel)
                else:
                    rels_per_N[N].append(100.0)

        # after trials average
        for iN, N in enumerate(Ns):
            matrix[iN, jS] = float(np.mean(rels_per_N[N]))

    # prepare fine grid for interpolation (CUBIC)
    NN, SS = np.meshgrid(Ns, sigmas, indexing='ij')
    points = np.vstack([NN.ravel(), SS.ravel()]).T
    values = matrix.ravel()

    n_fine = (len(Ns)-1) * zoom_factor + 1
    s_fine = (len(sigmas)-1) * zoom_factor + 1
    Ns_display = np.linspace(Ns[0], Ns[-1], n_fine)
    sigmas_display = np.linspace(sigmas[0], sigmas[-1], s_fine)
    NNf, SSf = np.meshgrid(Ns_display, sigmas_display, indexing='ij')
    grid_points = np.vstack([NNf.ravel(), SSf.ravel()]).T

    try:
        interp_vals = griddata(points, values, grid_points, method='cubic')
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

    # Validate interpolation at subset of fine grid points
    rng = np.random.default_rng(12345)
    num_validate = min(20, NNf.size)
    idx = rng.choice(np.arange(NNf.size), size=num_validate, replace=False)
    real_vals = []
    interp_vals_at_points = []
    for k in idx:
        N_val = int(round(NNf.ravel()[k]))
        s_val = float(SSf.ravel()[k])
        errs = []
        # run a few trials to estimate ground truth
        for tr in range(3):
            x_tmp, y_tmp_true = generate_system(model, (a, b), N_val, xmin, xmax)
            if noise_type == "gaussian":
                noise_tmp = np.random.normal(0, s_val, size=N_val)
            else:
                noise_tmp = np.random.uniform(-s_val, s_val, size=N_val)
            y_tmp_noisy = y_tmp_true + noise_tmp
            xt, yt, mask = apply_transform(model, x_tmp, y_tmp_noisy)
            if len(xt) >= 2:
                best, _ = evaluate_windows(xt, yt, mode=window_method)
                est = inverse_params(model, best["b0"], best["b1"])
                if model == "linear":
                    true_first = a
                    est_first = est.get("A", np.nan)
                elif model == "exponential":
                    true_first = a
                    est_first = est.get("lambda", np.nan)
                else:
                    true_first = a
                    est_first = est.get(list(est.keys())[0], np.nan)
                rel = float(100.0 * np.abs(true_first - est_first) / (np.abs(true_first) + 1e-12))
                errs.append(rel)
            else:
                errs.append(100.0)
        real_vals.append(np.mean(errs))
        interp_vals_at_points.append(matrix_fine.ravel()[k])

    real_vals = np.array(real_vals)
    interp_vals_at_points = np.array(interp_vals_at_points)
    interp_error_rmse = float(np.sqrt(np.mean((real_vals - interp_vals_at_points)**2)))
    interp_error_mean = float(np.mean(np.abs(real_vals - interp_vals_at_points)))

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
