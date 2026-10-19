// Shared by both pages: the language switch, translation, formatting and the model status badge.

import { STRINGS } from "/i18n.js";

const LANG_KEY = "wd.lang";
const listeners = [];

function storedLang() {
  try {
    const lang = localStorage.getItem(LANG_KEY);
    return lang in STRINGS ? lang : "ar";
  } catch {
    return "ar";
  }
}

export let lang = storedLang();

export function t(key, vars = {}) {
  const text = STRINGS[lang][key] ?? STRINGS.ar[key] ?? key;
  return text.replace(/\{(\w+)\}/g, (_, name) => (name in vars ? String(vars[name]) : `{${name}}`));
}

export function onLanguageChange(fn) {
  listeners.push(fn);
}

function applyStatic() {
  document.documentElement.lang = lang;
  document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
  for (const el of document.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n);
  for (const el of document.querySelectorAll("[data-i18n-title]")) el.title = t(el.dataset.i18nTitle);
  document.title = `${t(document.body.dataset.pageTitle)} · ${t("app.title")}`;
}

export function setLanguage(next) {
  lang = next;
  try {
    localStorage.setItem(LANG_KEY, lang);
  } catch {
    // The choice just is not remembered.
  }
  applyStatic();
  for (const fn of listeners) fn();
}

// Numbers stay in Latin digits in both languages, as on CCTV equipment.
export function locale() {
  return lang === "ar" ? "ar-u-nu-latn" : "en-GB";
}

export function formatTime(iso) {
  // Camera-local time without a zone: shown exactly as recorded.
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return new Intl.DateTimeFormat(locale(), { dateStyle: "medium", timeStyle: "medium" }).format(date);
}

export function percent(value) {
  return new Intl.NumberFormat(locale(), { style: "percent", maximumFractionDigits: 0 }).format(value);
}

export function number(value, digits = 2) {
  return new Intl.NumberFormat(locale(), { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
}

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (name === "class") node.className = value;
    else if (name === "text") node.textContent = value;
    else if (name.startsWith("on")) node.addEventListener(name.slice(2), value);
    else node.setAttribute(name, value === true ? "" : value);
  }
  node.append(...children.filter((c) => c !== null && c !== undefined));
  return node;
}

export async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

let health = null;

function renderHealth() {
  const badge = document.getElementById("health");
  if (!badge) return;
  if (!health) {
    badge.textContent = t("health.down");
    badge.dataset.state = "error";
    return;
  }
  badge.dataset.state = health.status;
  if (health.status === "ready") badge.textContent = t("health.ready", { detector: t(`detector.${health.detector}`) });
  else if (health.status === "error") badge.textContent = t("health.error", { error: health.error });
  else badge.textContent = t("health.loading");
}

export function currentHealth() {
  return health;
}

async function pollHealth() {
  try {
    health = await getJSON("/api/health");
  } catch {
    health = null;
  }
  renderHealth();
  setTimeout(pollHealth, health?.status === "ready" ? 15000 : 2000);
}

export function initPage() {
  applyStatic();
  document.getElementById("lang-switch").addEventListener("click", () => setLanguage(lang === "ar" ? "en" : "ar"));
  onLanguageChange(renderHealth);
  pollHealth();
}
