// static/js/main.js
function qs(id) { return document.getElementById(id); }

function updateDisplay() {
    qs("val-a").innerText = parseFloat(qs("a").value).toFixed(2);
    qs("val-b").innerText = parseFloat(qs("b").value).toFixed(2);
    qs("val-N").innerText = parseInt(qs("N").value);
    qs("val-sigma").innerText = parseFloat(qs("sigma").value).toFixed(2);
    qs("val-bins").innerText = parseInt(qs("bins").value);
}

function updateLabels() {
    const model = qs("model").value;
    if (model === "linear") {
        qs("label-a").innerText = "A";
        qs("label-b").innerText = "B";
    } else if (model === "exponential") {
        qs("label-a").innerText = "λ (lambda)";
        qs("label-b").innerText = "μ (mu)";
    } else if (model === "power") {
        qs("label-a").innerText = "a";
        qs("label-b").innerText = "b (exponent)";
    } else if (model === "logarithmic") {
        qs("label-a").innerText = "a";
        qs("label-b").innerText = "b";
    }
}

function collectBasePayload() {
    return {
        model: qs("model").value,
        a: parseFloat(qs("a").value),
        b: parseFloat(qs("b").value),
        xmin: parseFloat(qs("xmin").value),
        xmax: parseFloat(qs("xmax").value),
        N: parseInt(qs("N").value),
        sigma: parseFloat(qs("sigma").value),
        bins: parseInt(qs("bins").value),
        noise_type: qs("noise_type").value,
        fit: qs("doFit").checked,
        window_method: qs("window_method").value
    };
}

function runExperiment() {
    qs("runBtn").disabled = true;
    qs("runBtn").innerText = "Running...";
    const payload = collectBasePayload();

    fetch("/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    }).then(r => r.json())
    .then(data => {
        qs("plot_system").src = "data:image/png;base64," + data.system;
        qs("plot_hist").src = "data:image/png;base64," + data.hist;
        qs("plot_system_noise").src = "data:image/png;base64," + data.system_noise;
        qs("plot_fit").src = "data:image/png;base64," + data.fit;

        qs("params_est").innerText = JSON.stringify(data.estimated, null, 2);
        qs("r2").innerText = (data.r2 !== undefined) ? data.r2.toFixed(4) : "--";
        qs("snr").innerText = (data.snr !== undefined) ? data.snr.toFixed(2) : "--";
        qs("best_window").innerText = (data.best_window && data.best_window.get) ? JSON.stringify(data.best_window) : JSON.stringify(data.best_window);
        qs("r2_values").innerText = JSON.stringify(data.r2_values, null, 2);
        qs("errs").innerText = "abs: " + JSON.stringify(data.abs_err) + "\nrel%: " + JSON.stringify(data.rel_err);
    })
    .catch(err => alert("Error: " + err))
    .finally(() => {
        qs("runBtn").disabled = false;
        qs("runBtn").innerText = "Run Experiment";
    });
}

function runGrid() {
    qs("runGridBtn").disabled = true;
    qs("runGridBtn").innerText = "Running grid...";
    const payload = collectBasePayload();

    const Ns_raw = qs("gridNs").value.split(",").map(s => s.trim()).filter(s => s.length);
    const sig_raw = qs("gridSigmas").value.split(",").map(s => s.trim()).filter(s => s.length);
    payload["Ns"] = Ns_raw.map(x => parseInt(x));
    payload["sigmas"] = sig_raw.map(x => parseFloat(x));
    payload["trials"] = parseInt(qs("gridTrials").value) || 5;
    payload["zoom"] = parseInt(qs("gridZoom").value) || 1;

    fetch("/grid", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    }).then(r => r.json())
    .then(data => {
        qs("plot_heat").src = "data:image/png;base64," + data.heatmap;
        qs("interp_method").innerText = data.interp_method;
        qs("interp_rmse").innerText = data.interp_rmse.toFixed(3);
        qs("interp_mean").innerText = data.interp_mean_abs.toFixed(3);
    })
    .catch(err => alert("Grid error: " + err))
    .finally(() => {
        qs("runGridBtn").disabled = false;
        qs("runGridBtn").innerText = "Run Grid & Show Heatmap";
    });
}

document.addEventListener("DOMContentLoaded", function() {
    updateLabels();
    updateDisplay();

    qs("model").addEventListener("change", updateLabels);
    ["a","b","N","sigma","bins"].forEach(id => {
        qs(id).addEventListener("input", updateDisplay);
    });

    qs("runBtn").addEventListener("click", runExperiment);
    qs("runGridBtn").addEventListener("click", runGrid);

    setTimeout(runExperiment, 200);
});
