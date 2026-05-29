/**
 * Skytrace Phase 6 — minimal annotate workflow.
 */

const API = window.location.origin;

const CITIES = [
  "new york", "london", "paris", "tokyo", "sydney",
  "los angeles", "chicago", "toronto", "berlin", "mumbai","new_delhi",
  "singapore", "denver", "seattle", "boston", "san francisco",
];

const form = document.getElementById("form");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const outputImg = document.getElementById("output");
const summaryEl = document.getElementById("summary");
const submitBtn = document.getElementById("submit");

function init() {
  const list = document.getElementById("cities");
  CITIES.forEach((city) => {
    const opt = document.createElement("option");
    opt.value = city;
    list.appendChild(opt);
  });

  const now = new Date();
  const utc = new Date(now.getTime() - now.getTimezoneOffset() * 60000);
  document.getElementById("timestamp").value = utc.toISOString().slice(0, 16);
}

function toUtcIso(localValue) {
  if (!localValue) return null;
  const d = new Date(localValue);
  if (Number.isNaN(d.getTime())) return null;
  return d.toISOString().replace(/\.\d{3}Z$/, "Z");
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.className = isError ? "status error" : "status";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const fileInput = document.getElementById("file");
  const file = fileInput.files[0];
  if (!file) {
    setStatus("Choose an image.", true);
    return;
  }

  const timestamp = toUtcIso(document.getElementById("timestamp").value);
  if (!timestamp) {
    setStatus("Enter a valid date and time.", true);
    return;
  }

  const body = new FormData();
  body.append("file", file);
  body.append("city", document.getElementById("city").value.trim());
  body.append("timestamp", timestamp);
  body.append("center_azimuth_deg", document.getElementById("direction").value);
  body.append("center_altitude_deg", document.getElementById("altitude").value);
  body.append("show_planets", "true");
  body.append("show_constellations", "true");

  submitBtn.disabled = true;
  resultEl.hidden = true;
  setStatus("Processing…");

  try {
    const response = await fetch(`${API}/annotate`, { method: "POST", body });
    const data = await response.json();

    if (!response.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map((d) => d.msg || d).join(", ")
        : data.detail;
      throw new Error(data.error || detail || response.statusText);
    }

    outputImg.src = `${API}${data.annotated_image}?t=${Date.now()}`;
    summaryEl.textContent = [
      `${data.label_count ?? 0} labels`,
      `${data.match_count ?? 0} stars matched`,
      (data.warnings || []).join(" "),
    ]
      .filter(Boolean)
      .join(" · ");

    resultEl.hidden = false;
    setStatus("Done.");
  } catch (err) {
    setStatus(err.message || "Request failed.", true);
  } finally {
    submitBtn.disabled = false;
  }
});

init();
