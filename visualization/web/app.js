// Initialize chart
const chart = klinecharts.init("chart");

// Get symbol from query string (e.g., index.html?symbol=TSLA)
const urlParams = new URLSearchParams(window.location.search);
const SYMBOL = (urlParams.get("symbol") || "AAPL").toUpperCase();

// Candidate paths depending on where you run `http.server`
const candidates = [
    `./output/${SYMBOL}_chart.json`,   // if you serve from visualization/web
    `../output/${SYMBOL}_chart.json`,  // if you serve from visualization/
    `./${SYMBOL}_chart.json`           // fallback if you copy next to index.html
];

async function loadFirstAvailable(paths) {
    for (const p of paths) {
        try {
            const res = await fetch(p, { cache: "no-cache" });
            console.debug('[KLine] fetch', p, 'status', res.status);
            if (!res.ok) continue;
            const raw = await res.json();
            return { raw, path: p };
        } catch (err) {
            console.debug('[KLine] fetch error for', p, err);
            // try next path
        }
    }
    throw new Error("No chart JSON found in known paths.");
}

function normalizeAndApply(raw) {
    // Debug raw payload
    console.debug('[KLine] Raw payload type:', typeof raw);
    console.debug('[KLine] Raw payload preview:', raw);

    // Determine data array from multiple possible exporter formats
    let data = null;
    if (Array.isArray(raw)) {
        data = raw;
        console.debug('[KLine] Using top-level array, length=', data.length);
    } else if (raw && Array.isArray(raw.candles)) {
        data = raw.candles;
        console.debug('[KLine] Using raw.candles, length=', data.length);
    } else if (raw && Array.isArray(raw.chart)) {
        data = raw.chart;
        console.debug('[KLine] Using raw.chart, length=', data.length);
    } else {
        console.error('[KLine] Unexpected JSON format - expected array or { candles: [...] }', raw);
        const el = document.getElementById("chart");
        el.innerHTML = `<div style="padding:12px;font-family:system-ui, sans-serif;color:#900">
      Invalid JSON format for <b>${SYMBOL}</b>. Check console for details.
    </div>`;
        return;
    }

    // Normalize and validate each data point
    for (let i = 0; i < data.length; i++) {
        const d = data[i];
        if (!d) {
            console.warn('[KLine] skipping null data at index', i);
            continue;
        }

        // Some exporters use "date" or ISO strings; prefer numeric timestamp
        if (d.timestamp === undefined && d.date) {
            // try parse date string
            const parsed = new Date(d.date);
            d.timestamp = Number(parsed.getTime());
        }

        if (typeof d.timestamp === "string") {
            d.timestamp = Number(d.timestamp);
        }

        // If timestamp looks like seconds (10 digits), convert to ms
        if (Number.isFinite(d.timestamp) && d.timestamp < 1e12) {
            d.timestamp = d.timestamp * 1000;
        }

        // Attach cleaned timestamp back
        data[i].timestamp = d.timestamp;
    }

    // Apply the cleaned candle array to the chart
    try {
        chart.applyNewData(data);
        chart.setTimezone('Asia/Kuala_Lumpur');  // optional
        chart.setPriceVolumePrecision(2, 0);
        chart.setOffsetRightDistance(20); // adds some breathing room
        chart.setZoomEnabled(true);       // allow user zoom/pan
        // chart.setVisibleRange({ from: 0, to: data.length }); // show all candles

    } catch (err) {
        console.error('[KLine] chart.applyNewData failed:', err, { dataPreview: data.slice(0, 3) });
        const el = document.getElementById("chart");
        el.innerHTML = `<div style="padding:12px;font-family:system-ui, sans-serif;color:#900">
      Chart rendering error - check console for details.
    </div>`;
        return;
    }

    // Build marker points from signals (preferred) or from inline signals inside candles
    let markers = [];

    if (raw && Array.isArray(raw.signals)) {
        for (const sig of raw.signals) {
            let ts = sig.timestamp;
            if (typeof ts === "string") {
                const dt = new Date(ts);
                ts = dt.getTime();
            }
            if (Number.isFinite(ts) && ts < 2e11) ts = ts * 1000;
            markers.push({
                id: String(ts),
                timestamp: ts,
                position: "aboveBar",
                color: sig.type === "BUY" ? "green" : "red",
                text: sig.type
            });
        }
    } else {
        // fallback: look for .signal field on candles
        markers = data
            .filter(d => d && d.signal)
            .map(d => ({
                id: String(d.timestamp),
                timestamp: d.timestamp,
                position: "aboveBar",
                color: d.signal === "BUY" ? "green" : "red",
                text: d.signal
            }));
    }

    if (markers.length) {
        try {
            markers.forEach(m => {
                chart.createOverlay({
                    name: 'text',
                    points: [{ timestamp: m.timestamp, value: m.price || data.find(c => c.timestamp === m.timestamp)?.close }],
                    styles: {
                        text: {
                            color: m.color,
                            text: m.text,
                            size: 12,
                            family: 'system-ui, sans-serif'
                        }
                    }
                });
            });
            console.debug('[KLine] Created overlays count=', markers.length);
        } catch (err) {
            console.error('[KLine] createOverlay failed:', err, { markersPreview: markers.slice(0, 5) });
        }
    } else {
        console.debug('[KLine] No markers to create');
    }
}

(async function main() {
    try {
        const { raw, path } = await loadFirstAvailable(candidates);
        console.log("Loaded raw JSON:", raw);
        normalizeAndApply(raw);
        // Optional: show which file was loaded in console
        console.info(`[KLine] Loaded ${SYMBOL} from ${path}`);
    } catch (err) {
        console.error("Error loading chart data:", err);
        const el = document.getElementById("chart");
        el.innerHTML = `<div style="padding:12px;font-family:system-ui, sans-serif">
      Failed to load chart data for <b>${SYMBOL}</b>.<br/>
      Tried:<br/>
      ${candidates.map(c => `&bull; ${c}`).join("<br/>")}
    </div>`;
    }
})();