// The alerts page: upload an image or a video, follow its check, and see the alerts it raised.

import { currentHealth, el, formatTime, getJSON, initPage, number, onLanguageChange, percent, t } from "/common.js";

let uploads = [];
let alerts = [];
let uploadState = null; // { key, vars } of the last upload message

function localNow() {
  const now = new Date();
  now.setMilliseconds(0);
  return new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 19);
}

function setUploadStatus(key, vars = {}) {
  uploadState = key ? { key, vars } : null;
  renderUploadStatus();
}

function renderUploadStatus() {
  document.getElementById("upload-status").textContent = uploadState ? t(uploadState.key, uploadState.vars) : "";
  document.getElementById("upload-hint").textContent = t("upload.hint", { fps: currentHealth()?.sample_fps ?? 2 });
}

function statusText(upload) {
  if (upload.status === "processing") return t("status.processing", { pct: Math.round(upload.progress * 100) });
  if (upload.status === "failed") return t("status.failed", { error: upload.error ?? "" });
  return t(`status.${upload.status}`);
}

function renderUploads() {
  const body = document.getElementById("uploads-body");
  if (!uploads.length) {
    body.replaceChildren(el("tr", {}, el("td", { colspan: 6, class: "muted", text: t("uploads.empty") })));
    return;
  }
  body.replaceChildren(
    ...uploads.map((u) =>
      el(
        "tr",
        {},
        el("td", { dir: "auto", text: u.filename }),
        el("td", { text: t(`kind.${u.kind}`) }),
        el("td", { dir: "ltr", text: u.camera }),
        el("td", { text: formatTime(u.start_time) }),
        el("td", { class: `state-${u.status}`, text: statusText(u) }),
        el(
          "td",
          { class: u.status === "done" && !u.alerts ? "none-found" : u.alerts ? "has-alerts" : "" },
          u.status === "done" && !u.alerts ? t("uploads.none") : String(u.alerts),
        ),
      ),
    ),
  );
}

function alertCard(a) {
  const src = `/media/alerts/${encodeURIComponent(a.image)}?v=${a.confidence}`;
  const facts = el(
    "dl",
    { class: "facts" },
    el("dt", { text: t("alert.confidence") }),
    el("dd", { dir: "ltr", text: percent(a.confidence) }),
    el("dt", { text: t("alert.camera") }),
    el("dd", { dir: "ltr", text: a.camera }),
    el("dt", { text: t("alert.time") }),
    el("dd", { text: formatTime(a.at_time) }),
    el("dt", { text: t("alert.source") }),
    el(
      "dd",
      {},
      el("span", { dir: "auto", text: a.filename }),
      a.kind === "video" ? el("span", { class: "muted", text: ` · ${t("alert.offset", { s: number(a.offset_seconds, 1) })}` }) : null,
      a.kind === "video" && a.frames > 1 ? el("span", { class: "muted", text: ` · ${t("alert.frames", { n: a.frames })}` }) : null,
    ),
  );
  return el(
    "article",
    { class: "card alert", "data-label": a.label },
    el("a", { href: src, target: "_blank", rel: "noopener", title: t("alert.open") }, el("img", { src, alt: t(`label.${a.label}`), loading: "lazy" })),
    el("div", { class: "card-body" }, el("strong", { class: `label-badge label-${a.label}`, text: t(`label.${a.label}`) }), facts),
  );
}

function renderAlerts() {
  document.getElementById("alerts-count").textContent = alerts.length ? t("alerts.count", { n: alerts.length }) : "";
  document.getElementById("alerts-empty").textContent = alerts.length ? "" : t("alerts.empty");
  document.getElementById("alerts").replaceChildren(...alerts.map(alertCard));
}

function render() {
  renderUploadStatus();
  renderUploads();
  renderAlerts();
}

let refreshTimer = null;

async function refresh() {
  clearTimeout(refreshTimer);
  try {
    [uploads, alerts] = await Promise.all([getJSON("/api/uploads"), getJSON("/api/alerts")]);
    renderUploads();
    renderAlerts();
  } catch {
    // The health badge already says the server is unreachable; try again shortly.
  }
  const busy = uploads.some((u) => u.status === "queued" || u.status === "processing");
  if (!busy && uploadState?.key === "upload.done") setUploadStatus(null);
  refreshTimer = setTimeout(refresh, busy ? 1000 : 10000);
}

function upload(form) {
  const progress = document.getElementById("upload-progress");
  const button = form.querySelector("button[type=submit]");
  const request = new XMLHttpRequest();
  request.open("POST", "/api/uploads");
  request.upload.addEventListener("progress", (event) => {
    if (!event.lengthComputable) return;
    progress.value = event.loaded / event.total;
    setUploadStatus("upload.sending", { pct: Math.round((100 * event.loaded) / event.total) });
  });
  request.addEventListener("loadend", () => {
    progress.hidden = true;
    button.disabled = false;
    if (request.status === 201) {
      setUploadStatus("upload.done");
      form.elements.file.value = "";
    } else {
      let detail = request.statusText || "network error";
      try {
        detail = JSON.parse(request.responseText).detail ?? detail;
      } catch {
        // Keep the status text.
      }
      setUploadStatus("upload.failed", { error: typeof detail === "string" ? detail : JSON.stringify(detail) });
    }
    refresh();
  });
  progress.value = 0;
  progress.hidden = false;
  button.disabled = true;
  request.send(new FormData(form));
}

async function clearAll() {
  if (!confirm(t("alerts.clearConfirm"))) return;
  const response = await fetch("/api/alerts", { method: "DELETE" });
  if (response.status === 409) alert(t("alerts.busy"));
  setUploadStatus(null);
  refresh();
}

initPage();
const form = document.getElementById("upload-form");
form.elements.start_time.value = localNow();
form.addEventListener("submit", (event) => {
  event.preventDefault();
  upload(form);
});
document.getElementById("alerts-clear").addEventListener("click", clearAll);
onLanguageChange(render);
render();
refresh();
