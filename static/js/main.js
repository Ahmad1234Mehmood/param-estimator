function qs(id) { return document.getElementById(id); }

function updateDisplay() {
    const displayMap = {
        "a": "val-a",
        "b": "val-b",
        "N": "val-N",
        "sigma": "val-sigma",
        "bins": "val-bins"
    };
    Object.keys(displayMap).forEach(id => {
        const el = qs(id);
        const valEl = qs(displayMap[id]);
        if(el && valEl) {
            if(id === "N" || id === "bins") valEl.innerText = parseInt(el.value);
            else valEl.innerText = parseFloat(el.value).toFixed(2);
        }
    });
}

function updateLabels() {
    const model = qs("model")?.value;
    if(!model) return;

    const labelAMap = {
        linear: "A",
        exponential: "λ (lambda)",
        power: "a",
        logarithmic: "a"
    };
    const labelBMap = {
        linear: "B",
        exponential: "μ (mu)",
        power: "b (exponent)",
        logarithmic: "b"
    };

    if(qs("label-a")) qs("label-a").innerText = labelAMap[model];
    if(qs("label-b")) qs("label-b").innerText = labelBMap[model];
}

function collectBasePayload() {
    return {
        model: qs("model")?.value || "linear",
        a: parseFloat(qs("a")?.value || 1),
        b: parseFloat(qs("b")?.value || 0),
        xmin: parseFloat(qs("xmin")?.value || 0),
        xmax: parseFloat(qs("xmax")?.value || 10),
        N: parseInt(qs("N")?.value || 50),
        sigma: parseFloat(qs("sigma")?.value || 0.5),
        bins: parseInt(qs("bins")?.value || 20),
        noise_type: qs("noise_type")?.value || "gaussian",
        fit: qs("doFit")?.checked || false,
        window_method: qs("window_method")?.value || "transformed"
    };
}

function runExperiment() {
    const runBtn = qs("runBtn");
    if(runBtn) {
        runBtn.disabled = true;
        runBtn.innerText = "Running...";
    }

    const payload = collectBasePayload();

    fetch("/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    }).then(r => r.json())
    .then(data => {
        ["plot_system","plot_hist","plot_system_noise","plot_fit"].forEach(id => {
            if(qs(id) && data[id.split("_")[1]]) qs(id).src = "data:image/png;base64," + data[id.split("_")[1]];
        });

        // Build parameter table
        const params = data.estimated || {};
        const abs_err = data.abs_err || {};
        const rel_err = data.rel_err || {};
        let table = "Parameter | True | Estimated | Abs err | Rel err (%)\n";

        const model = qs("model")?.value || "linear";
        const trueA = parseFloat(qs("a")?.value || 1);
        const trueB = parseFloat(qs("b")?.value || 0);
        let trueParams = {};
        if(model === "linear") trueParams = {"A": trueA, "B": trueB};
        else if(model === "exponential") trueParams = {"lambda": trueA, "mu": trueB};
        else trueParams = {"a": trueA, "b": trueB};

        Object.keys(params).forEach(k => {
            const t = trueParams[k] !== undefined ? trueParams[k] : "--";
            const e = params[k];
            const ae = abs_err[k] !== undefined ? abs_err[k] : "--";
            const re = rel_err[k] !== undefined ? rel_err[k] : "--";
            table += `${k} | ${t} | ${e} | ${ae} | ${re}\n`;
        });

const tbody = qs("params_est_table");
if (tbody) {
    tbody.innerHTML = "";

    const modelName = qs("model")?.value || "--";
    const N_val = qs("N")?.value || "--";
    const sigma_val = qs("sigma")?.value || "--";

    Object.keys(params).forEach(k => {
        const row = document.createElement("tr");

        const trueVal = trueParams[k] !== undefined ? trueParams[k] : "--";
        const estVal = params[k] ?? "--";
        const absErr = abs_err[k] ?? "--";

        // Proper relative error handling
        let relErr = "--";
        if (trueVal !== 0) {
            relErr = (rel_err[k] !== undefined) ? rel_err[k] : "--";
        }

        row.innerHTML = `
            <td>${modelName}</td>
            <td>${N_val}</td>
            <td>${sigma_val}</td>
            <td>${k}</td>
            <td>${estVal}</td>
            <td>${absErr}</td>
            <td>${relErr}</td>
        `;

        tbody.appendChild(row);
    });
}


        if(qs("r2")) qs("r2").innerText = (data.r2 !== undefined) ? data.r2.toFixed(4) : "--";
        if(qs("snr")) qs("snr").innerText = (data.snr !== undefined) ? data.snr.toFixed(2) : "--";
        if(qs("best_window")) qs("best_window").innerText = JSON.stringify(data.best_window || {});
        if(qs("r2_values")) qs("r2_values").innerText = JSON.stringify(data.r2_values || {}, null, 2);
        if(qs("errs")) qs("errs").innerText = "";
    })
    .catch(err => alert("Error: " + err))
    .finally(() => {
        if(runBtn) {
            runBtn.disabled = false;
            runBtn.innerText = "Run Experiment";
        }
    });
}

function runGrid() {
    const btn = qs("runGridBtn");
    if(btn) {
        btn.disabled = true;
        btn.innerText = "Running grid...";
    }

    const payload = collectBasePayload();

    const Ns_raw = (qs("gridNs")?.value || "").split(",").map(s => s.trim()).filter(s => s.length);
    const sig_raw = (qs("gridSigmas")?.value || "").split(",").map(s => s.trim()).filter(s => s.length);
    payload["Ns"] = Ns_raw.map(x => parseInt(x));
    payload["sigmas"] = sig_raw.map(x => parseFloat(x));
    payload["trials"] = parseInt(qs("gridTrials")?.value || 5);
    payload["zoom"] = parseInt(qs("gridZoom")?.value || 1);

    fetch("/grid", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    }).then(r => r.json())
    .then(data => {
        if(qs("plot_heat")) qs("plot_heat").src = "data:image/png;base64," + (data.heatmap || "");
        if(qs("interp_method")) qs("interp_method").innerText = data.interp_method || "--";
        if(qs("interp_rmse")) qs("interp_rmse").innerText = (data.interp_rmse !== undefined) ? data.interp_rmse.toFixed(3) : "--";
        if(qs("interp_mean")) qs("interp_mean").innerText = (data.interp_mean_abs !== undefined) ? data.interp_mean_abs.toFixed(3) : "--";
    })
    .catch(err => alert("Grid error: " + err))
    .finally(() => {
        if(btn) {
            btn.disabled = false;
            btn.innerText = "Run Grid & Show Heatmap";
        }
    });
}

// --- Make parameter table clickable for CSV download ---
function downloadParamTable() {
    const pre = qs("params_est");
    if(!pre) return;

    const text = pre.innerText;
    const csvContent = "data:text/csv;charset=utf-8," + text.split("\n").join("\r\n");
    const a = document.createElement("a");
    a.href = encodeURI(csvContent);
    a.download = "params_table.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
}

document.addEventListener("DOMContentLoaded", function() {
    updateLabels();
    updateDisplay();

    // Update model & sliders
    qs("model")?.addEventListener("change", () => {
        updateLabels();
        runExperiment();
    });

    ["a","b","N","sigma","bins","xmin","xmax"].forEach(id => {
        const el = qs(id);
        if(!el) return;
        el.addEventListener("input", () => updateDisplay());
        el.addEventListener("change", () => runExperiment());
    });

    qs("runBtn")?.addEventListener("click", runExperiment);
    qs("runGridBtn")?.addEventListener("click", runGrid);

    // PDF export
    qs("exportPDF")?.addEventListener("click", () => {
        const payload = collectBasePayload();
        const btn = qs("exportPDF");
        if(btn) {
            btn.disabled = true;
            btn.innerText = "Generating PDF...";
        }

        fetch("/export_pdf", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        }).then(r => {
            if(!r.ok) throw new Error("PDF generation failed");
            return r.blob();
        }).then(blob => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "ParamEstimator_report.pdf";
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
        }).catch(err => alert("PDF error: " + err))
        .finally(() => {
            if(btn) {
                btn.disabled = false;
                btn.innerText = "Export PDF";
            }
        });
    });

    // Make parameter table clickable to download CSV
    const paramsPre = qs("params_est");
    if(paramsPre) {
        paramsPre.style.cursor = "pointer";
        paramsPre.title = "Click to download CSV";
        paramsPre.addEventListener("click", downloadParamTable);
    }

    setTimeout(runExperiment, 200);
});
