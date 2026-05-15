/**
 * XY KSampler Plot — ComfyUI frontend extension
 *
 * Adds two things to the XYKSamplerPlot node:
 *   1.  "Pick LoRAs" buttons next to x_values / y_values that open a
 *       searchable checkbox picker modal and fill the text field.
 *   2.  Live axis-type awareness: changes the placeholder/hint text on
 *       the values field to match the selected axis type.
 */

import { app } from "../../scripts/app.js";

// ── Utility: fetch lora list from the node's own combo ─────────────────────

async function fetchLoraList() {
    try {
        const resp = await fetch("/object_info/XYKSamplerPlot");
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        // base_lora combo is at input.required.base_lora[0]
        const loras = data?.XYKSamplerPlot?.input?.required?.base_lora?.[0] ?? [];
        return loras.filter(l => l !== "None");
    } catch (e) {
        console.warn("[XYKSamplerPlot] Could not fetch LoRA list:", e);
        return [];
    }
}

// ── Placeholder hints per axis type ────────────────────────────────────────

const AXIS_HINTS = {
    "Seed":        "e.g.  0, 42, 1337, 99999",
    "LoRA":        "filenames — use Pick LoRAs button →",
    "LoRA Weight": "e.g.  0.5, 0.75, 1.0, 1.25",
    "CFG":         "e.g.  4.0, 6.0, 7.5, 9.0",
    "Steps":       "e.g.  10, 15, 20, 30",
};

// ── Modal ──────────────────────────────────────────────────────────────────

function buildModal(loras, currentValues, onConfirm) {
    // Overlay
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        position:        "fixed",
        inset:           "0",
        background:      "rgba(0,0,0,0.75)",
        zIndex:          "99999",
        display:         "flex",
        alignItems:      "center",
        justifyContent:  "center",
        fontFamily:      "system-ui, sans-serif",
    });

    // Dialog box
    const dialog = document.createElement("div");
    Object.assign(dialog.style, {
        background:    "#1c1c2a",
        border:        "1px solid #3a3a55",
        borderRadius:  "10px",
        padding:       "20px",
        width:         "min(580px, 92vw)",
        maxHeight:     "75vh",
        display:       "flex",
        flexDirection: "column",
        gap:           "12px",
        color:         "#ddd",
        boxShadow:     "0 8px 40px rgba(0,0,0,0.7)",
    });

    // ── Title ──
    const title = document.createElement("h3");
    title.textContent = "🎛️  Pick LoRAs";
    Object.assign(title.style, { margin: "0", color: "#b0b0e0", fontSize: "15px", letterSpacing: "0.04em" });
    dialog.appendChild(title);

    // ── Search ──
    const search = document.createElement("input");
    search.type = "text";
    search.placeholder = "Search LoRAs…";
    Object.assign(search.style, {
        padding:      "8px 12px",
        border:       "1px solid #444",
        borderRadius: "6px",
        background:   "#282838",
        color:        "#eee",
        fontSize:     "13px",
        outline:      "none",
    });
    dialog.appendChild(search);

    // ── Count label ──
    const countLabel = document.createElement("div");
    Object.assign(countLabel.style, { fontSize: "11px", color: "#666", marginTop: "-6px" });
    dialog.appendChild(countLabel);

    // ── List container ──
    const listWrap = document.createElement("div");
    Object.assign(listWrap.style, {
        overflowY:   "auto",
        flex:        "1 1 auto",
        border:      "1px solid #2a2a40",
        borderRadius:"6px",
        padding:     "4px",
        maxHeight:   "45vh",
    });
    dialog.appendChild(listWrap);

    const selected = new Set(currentValues);

    function renderList(filter = "") {
        listWrap.innerHTML = "";
        const fl = filter.toLowerCase();
        const visible = loras.filter(l => l.toLowerCase().includes(fl));
        countLabel.textContent = `${visible.length} of ${loras.length} LoRAs`;

        visible.forEach(lora => {
            const row = document.createElement("label");
            Object.assign(row.style, {
                display:       "flex",
                alignItems:    "center",
                gap:           "10px",
                padding:       "5px 10px",
                borderRadius:  "4px",
                cursor:        "pointer",
                userSelect:    "none",
                transition:    "background 0.1s",
            });
            row.onmouseenter = () => row.style.background = "#272740";
            row.onmouseleave = () => row.style.background = "transparent";

            const cb = document.createElement("input");
            cb.type    = "checkbox";
            cb.value   = lora;
            cb.checked = selected.has(lora);
            Object.assign(cb.style, { accentColor: "#7070cc", flexShrink: "0" });
            cb.addEventListener("change", () => {
                if (cb.checked) selected.add(lora);
                else            selected.delete(lora);
            });

            // Display name = strip path separators + extension
            const displayName = lora
                .replace(/\\/g, "/")
                .split("/")
                .pop()
                .replace(/\.(safetensors|ckpt|pt|bin)$/i, "");

            const nameSpan = document.createElement("span");
            nameSpan.textContent = displayName;
            Object.assign(nameSpan.style, { fontSize: "13px", color: "#ccd" });

            const fullSpan = document.createElement("span");
            fullSpan.textContent = lora;
            Object.assign(fullSpan.style, { fontSize: "10px", color: "#555", marginLeft: "auto", maxWidth: "180px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" });

            row.appendChild(cb);
            row.appendChild(nameSpan);
            row.appendChild(fullSpan);
            listWrap.appendChild(row);
        });
    }

    renderList();
    search.addEventListener("input", e => renderList(e.target.value));

    // ── Buttons ──
    const btnRow = document.createElement("div");
    Object.assign(btnRow.style, { display: "flex", gap: "10px", justifyContent: "flex-end" });

    const makeBtn = (text, bg, onClick) => {
        const b = document.createElement("button");
        b.textContent = text;
        Object.assign(b.style, {
            padding:      "7px 18px",
            border:       "none",
            borderRadius: "5px",
            cursor:       "pointer",
            fontWeight:   "600",
            fontSize:     "13px",
            background:   bg,
            color:        "#fff",
        });
        b.onclick = onClick;
        return b;
    };

    btnRow.appendChild(makeBtn("Cancel", "#3a3a4a", () => overlay.remove()));
    btnRow.appendChild(makeBtn("✓  Confirm", "#4a7a5a", () => {
        onConfirm([...selected]);
        overlay.remove();
    }));
    dialog.appendChild(btnRow);

    overlay.appendChild(dialog);
    overlay.addEventListener("click", e => { if (e.target === overlay) overlay.remove(); });

    // Close on Escape
    const escHandler = e => { if (e.key === "Escape") { overlay.remove(); document.removeEventListener("keydown", escHandler); } };
    document.addEventListener("keydown", escHandler);

    return overlay;
}

// ── Register extension ──────────────────────────────────────────────────────

app.registerExtension({
    name: "XYKSamplerPlot.LoRAPicker",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "XYKSamplerPlot") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const node = this;

            // ── Find widgets by name ──────────────────
            const find = name => node.widgets?.find(w => w.name === name);

            const xAxisW  = find("x_axis");
            const xValsW  = find("x_values");
            const yAxisW  = find("y_axis");
            const yValsW  = find("y_values");

            if (!xAxisW || !xValsW || !yAxisW || !yValsW) return;

            // ── Hint updater ──────────────────────────
            const updateHint = (axisWidget, valsWidget) => {
                const hint = AXIS_HINTS[axisWidget.value] ?? "";
                if (valsWidget.inputEl) {
                    valsWidget.inputEl.placeholder = hint;
                }
            };

            // Poll for inputEl (ComfyUI creates it lazily)
            const pollHints = setInterval(() => {
                if (xValsW.inputEl && yValsW.inputEl) {
                    clearInterval(pollHints);
                    updateHint(xAxisW, xValsW);
                    updateHint(yAxisW, yValsW);

                    // Live update when axis type changes
                    const origXCallback = xAxisW.callback;
                    xAxisW.callback = (...args) => {
                        origXCallback?.(...args);
                        updateHint(xAxisW, xValsW);
                    };
                    const origYCallback = yAxisW.callback;
                    yAxisW.callback = (...args) => {
                        origYCallback?.(...args);
                        updateHint(yAxisW, yValsW);
                    };
                }
            }, 200);

            // ── Pick LoRAs buttons ────────────────────
            let loraCache = null;
            const getLoraList = async () => {
                if (!loraCache) loraCache = await fetchLoraList();
                return loraCache;
            };

            const addPickerButton = (label, axisWidget, valsWidget) => {
                node.addWidget("button", `📂 ${label}`, null, async () => {
                    if (axisWidget.value !== "LoRA") {
                        // Quick toast-style warning instead of blocking alert
                        const msg = document.createElement("div");
                        msg.textContent = `Set "${axisWidget.name.replace("_", " ")}" to "LoRA" first`;
                        Object.assign(msg.style, {
                            position: "fixed", bottom: "24px", left: "50%",
                            transform: "translateX(-50%)",
                            background: "#2a2a1a", border: "1px solid #887722",
                            color: "#ddc", padding: "10px 20px", borderRadius: "6px",
                            zIndex: "99999", fontSize: "13px",
                        });
                        document.body.appendChild(msg);
                        setTimeout(() => msg.remove(), 2800);
                        return;
                    }

                    const loras = await getLoraList();
                    if (!loras.length) {
                        alert("No LoRAs found. Make sure your models/loras folder is populated.");
                        return;
                    }

                    const currentVals = (valsWidget.value || "")
                        .split(",")
                        .map(s => s.trim())
                        .filter(Boolean);

                    const modal = buildModal(loras, currentVals, selectedLoras => {
                        valsWidget.value = selectedLoras.join(", ");
                        // Trigger any downstream callbacks
                        if (typeof valsWidget.callback === "function") {
                            valsWidget.callback(valsWidget.value);
                        }
                        app.graph.setDirtyCanvas(true);
                    });

                    document.body.appendChild(modal);
                    // Auto-focus search
                    setTimeout(() => modal.querySelector("input[type=text]")?.focus(), 50);
                });
            };

            addPickerButton("Pick X LoRAs", xAxisW, xValsW);
            addPickerButton("Pick Y LoRAs", yAxisW, yValsW);
        };
    },
});
