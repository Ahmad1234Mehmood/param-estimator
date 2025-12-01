function runExperiment() {
    let payload = {
        a: Number(document.getElementById("a").value),
        b: Number(document.getElementById("b").value),
        xmin: Number(document.getElementById("xmin").value),
        xmax: Number(document.getElementById("xmax").value),
        N: Number(document.getElementById("N").value),
        sigma: Number(document.getElementById("sigma").value),
        bins: Number(document.getElementById("bins").value)
    };

    fetch("/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(data => {
        document.getElementById("plot_system").src = "data:image/png;base64," + data.system;
        document.getElementById("plot_hist").src = "data:image/png;base64," + data.hist;
        document.getElementById("plot_system_noise").src = "data:image/png;base64," + data.system_noise;
        document.getElementById("plot_fit").src = "data:image/png;base64," + data.fit;

        document.getElementById("est_a").innerText = data.estimated_a.toFixed(4);
        document.getElementById("est_b").innerText = data.estimated_b.toFixed(4);
        document.getElementById("error").innerText = data.error.toFixed(2);
    });
}
