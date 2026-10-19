// The test results page: scores of both detectors, the false-alarm check, and the 20 test images with boxes.

import { el, getJSON, initPage, number, onLanguageChange, t } from "/common.js";

let data = null;
let shown = null; // the detector whose boxes are drawn on the test images

function ratio(value) {
  return number(value, 2);
}

function scoreRows(name, result) {
  const test = result.test;
  const rows = [];
  const classes = Object.keys(test.per_class);
  classes.forEach((cls, index) => {
    const s = test.per_class[cls];
    rows.push(
      el(
        "tr",
        { class: index === 0 ? "group-start" : "" },
        index === 0 ? el("th", { rowspan: classes.length + 1, scope: "rowgroup", text: t(`detector.${name}`) }) : null,
        el("td", { text: t(`label.${cls}`) }),
        el("td", { dir: "ltr", text: ratio(s.precision) }),
        el("td", { dir: "ltr", text: ratio(s.recall) }),
        el("td", { dir: "ltr", text: ratio(s.ap50) }),
        el("td", { dir: "ltr", text: `${s.tp} / ${s.fp} / ${s.fn}` }),
        index === 0 ? el("td", { rowspan: classes.length + 1, dir: "ltr", text: ratio(result.threshold) }) : null,
        index === 0 ? el("td", { rowspan: classes.length + 1, dir: "ltr", text: number(result.seconds_per_image, 2) }) : null,
      ),
    );
  });
  const o = test.overall;
  rows.push(
    el(
      "tr",
      { class: "overall" },
      el("td", { text: t("class.overall") }),
      el("td", { dir: "ltr", text: ratio(o.precision) }),
      el("td", { dir: "ltr", text: ratio(o.recall) }),
      el("td", { dir: "ltr", text: ratio(o.map50) }),
      el("td", { dir: "ltr", text: `${o.tp} / ${o.fp} / ${o.fn}` }),
    ),
  );
  return rows;
}

function renderScores() {
  const detectors = data.evaluation.detectors;
  document.getElementById("scores-body").replaceChildren(...Object.entries(detectors).flatMap(([name, r]) => scoreRows(name, r)));
}

function renderFalseAlarms() {
  const fa = data.false_alarms;
  const note = document.getElementById("fa-note");
  const table = document.getElementById("fa-table");
  if (!fa) {
    note.textContent = t("results.fa.missing");
    table.hidden = true;
    return;
  }
  table.hidden = false;
  note.textContent = t("results.fa.note", { detector: t(`detector.${fa.detector}`), threshold: ratio(fa.threshold), fps: fa.sample_fps });
  const rows = fa.clips.map((c) =>
    el(
      "tr",
      {},
      el("td", { dir: "ltr", text: c.clip }),
      el("td", { dir: "ltr", text: number(c.seconds, 0) }),
      el("td", { dir: "ltr", text: c.frames_checked }),
      el("td", { dir: "ltr", text: c.alerting_frames }),
      el("td", { dir: "ltr", class: c.false_alarms ? "has-alerts" : "none-found", text: c.false_alarms }),
    ),
  );
  const total = fa.total;
  rows.push(
    el(
      "tr",
      { class: "overall" },
      el("td", { text: t("results.fa.total") }),
      el("td", { dir: "ltr", text: number(total.seconds, 0) }),
      el("td", { dir: "ltr", text: total.frames_checked }),
      el("td", { dir: "ltr", text: total.alerting_frames }),
      el("td", { dir: "ltr", class: total.false_alarms ? "has-alerts" : "none-found", text: total.false_alarms }),
    ),
  );
  document.getElementById("fa-body").replaceChildren(...rows);
}

function boxElement(box, width, height, className, caption) {
  const [x1, y1, x2, y2] = box;
  const style = `left:${(100 * x1) / width}%;top:${(100 * y1) / height}%;width:${(100 * (x2 - x1)) / width}%;height:${(100 * (y2 - y1)) / height}%`;
  return el("div", { class: `box ${className}`, style }, caption ? el("span", { text: caption }) : null);
}

function renderImages() {
  const evaluation = data.evaluation;
  const result = evaluation.detectors[shown];
  document.getElementById("detector-buttons").replaceChildren(
    ...Object.keys(evaluation.detectors).map((name) =>
      el("button", { type: "button", class: "chip", "aria-pressed": String(name === shown), text: t(`detector.${name}`), onclick: () => { shown = name; renderImages(); } }),
    ),
  );
  const figures = evaluation.test_images.map((img) => {
    const preds = result.predictions[img.file].filter((p) => p.confidence >= result.threshold);
    const counts = result.per_image[img.file];
    const truthBoxes = evaluation.truth[img.file].map((b) => boxElement(b.box, img.width, img.height, "truth", null));
    const predBoxes = preds.map((p) =>
      boxElement(p.box, img.width, img.height, `pred label-${p.label}`, `${t(`label.${p.label}`)} ${number(p.confidence, 2)}`),
    );
    return el(
      "figure",
      { class: "test-image" },
      el("div", { class: "frame-image" }, el("img", { src: `/dataset/${img.file}`, alt: img.file, loading: "lazy" }), ...truthBoxes, ...predBoxes),
      el(
        "figcaption",
        {},
        el("strong", { class: `label-badge label-${img.label}`, text: t(`label.${img.label}`) }),
        el("span", { dir: "ltr", class: "file", text: img.file.split("/").pop() }),
        el("span", { class: counts.fp || counts.fn ? "has-alerts" : "none-found", text: t("results.image.counts", counts) }),
      ),
    );
  });
  document.getElementById("test-images").replaceChildren(...figures);
}

function render() {
  if (!data?.evaluation) return;
  document.getElementById("choice").textContent = t(`results.choice.${data.demo_detector}`);
  renderScores();
  renderFalseAlarms();
  renderImages();
}

initPage();
onLanguageChange(render);
data = await getJSON("/api/results");
if (data.evaluation) {
  shown = data.demo_detector in data.evaluation.detectors ? data.demo_detector : Object.keys(data.evaluation.detectors)[0];
  document.getElementById("results").hidden = false;
  render();
} else {
  document.getElementById("results-missing").hidden = false;
}
