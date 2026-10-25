const updateBtn = document.getElementById("updateBtn");
const statusText = document.getElementById("statusText");
const langEnBtn = document.getElementById("langEnBtn");
const langFrBtn = document.getElementById("langFrBtn");

let activeLanguage = "en";
let hasRenderedSuccessfulMatch = false;
let lastRenderedPayload = null;
let imageRefreshToken = "";

const EXPO_PAYLOAD_STORAGE_KEY = "mtgExpo.latestMatchPayload";
const EXPO_IMAGE_TOKEN_STORAGE_KEY = "mtgExpo.imageRefreshToken";

const I18N = {
  en: {
    updateBtn: "Update Example",
    kicker: "Magic: The Gathering Detection Stages",
    heroTitle: "Detection & Localization",
    heroSub:
      "Visual checkpoints for YOLO detection, contrast masks, axis + side guides, and the final card crop used for identification.",
    summaryTitle: "Latest Detection Snapshot",
    summaryHint: "Press Update Example to sync these localization visuals with the latest successful card match.",
    locATitle: "A) YOLO Guess",
    locASub: "Full YOLO box with confidence details when available.",
    locBTitle: "B) Contrast Masked Boxes",
    locBSub: "Top and bottom contrast masks with no extra guide lines.",
    locCTitle: "C) Axis + Side Guesses",
    locCSub: "Pre-crop axis and side bounds forming the exact crop box.",
    locDTitle: "D) Final Cropped Image",
    locDSub: "Final cropped card image sent to identification.",
    mainPageLink: "Main Dashboard",
    localizationPageLink: "Detection & Localization",
    yoloConfidenceLabel: "YOLO Confidence",
    yoloConfidenceUnavailable: "YOLO Confidence: -",
    statusWaitingMatch: "Waiting for a matched card...",
    statusLoading: "Loading latest successful match...",
    statusNoMatches: "No successful matches available yet.",
    statusSynced: "Detection page synced with latest successful match.",
    statusFailedPrefix: "Failed to load data:",
  },
  fr: {
    updateBtn: "Mettre a jour",
    kicker: "Etapes de detection Magic: The Gathering",
    heroTitle: "Detection et localisation",
    heroSub:
      "Points de controle visuels pour la detection YOLO, les masques de contraste, les guides d'axe et de cotes, puis le recadrage final.",
    summaryTitle: "Derniere capture de detection",
    summaryHint: "Appuyez sur Mettre a jour pour synchroniser ces visuels avec la derniere correspondance reussie.",
    locATitle: "A) Estimation YOLO",
    locASub: "Boite YOLO complete avec details de confiance si disponibles.",
    locBTitle: "B) Boites masquees par contraste",
    locBSub: "Masques de contraste haut et bas sans lignes supplementaires.",
    locCTitle: "C) Axe et estimations laterales",
    locCSub: "Apercu avant recadrage montrant la boite finale.",
    locDTitle: "D) Image finale recadree",
    locDSub: "Image finale recadree envoyee a l'identification.",
    mainPageLink: "Tableau principal",
    localizationPageLink: "Detection et localisation",
    yoloConfidenceLabel: "Confiance YOLO",
    yoloConfidenceUnavailable: "Confiance YOLO : -",
    statusWaitingMatch: "En attente d'une carte correspondante...",
    statusLoading: "Chargement de la derniere correspondance reussie...",
    statusNoMatches: "Aucune correspondance reussie disponible pour le moment.",
    statusSynced: "Page de detection synchronisee avec la derniere correspondance reussie.",
    statusFailedPrefix: "Echec du chargement :",
  },
};

function t(key) {
  return (I18N[activeLanguage] || I18N.en)[key] || key;
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = value ?? "-";
}

function withCacheBust(url) {
  if (!url || typeof url !== "string") return "";
  if (!imageRefreshToken) return url;
  const token = `t=${encodeURIComponent(imageRefreshToken)}`;
  return url.includes("?") ? `${url}&${token}` : `${url}?${token}`;
}

function setImage(id, src) {
  const el = document.getElementById(id);
  if (!el) return;
  if (!src) {
    el.removeAttribute("src");
    return;
  }
  el.src = withCacheBust(src);
}

function updateStatus(message, isError = false) {
  if (!statusText) return;
  statusText.textContent = message;
  statusText.style.color = isError ? "#ffd6dd" : "";
}

function normalizeLabel(label) {
  return String(label || "").trim().toLowerCase();
}

function loadStoredPayload() {
  try {
    const raw = localStorage.getItem(EXPO_PAYLOAD_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch (_error) {
    return null;
  }
}

function loadStoredImageToken() {
  try {
    return localStorage.getItem(EXPO_IMAGE_TOKEN_STORAGE_KEY) || "";
  } catch (_error) {
    return "";
  }
}

function persistLatestPayload(payload) {
  try {
    localStorage.setItem(EXPO_PAYLOAD_STORAGE_KEY, JSON.stringify(payload || {}));
  } catch (_error) {
    // Ignore storage write failures.
  }
}

function persistImageRefreshToken(token) {
  try {
    localStorage.setItem(EXPO_IMAGE_TOKEN_STORAGE_KEY, token || "");
  } catch (_error) {
    // Ignore storage write failures.
  }
}

function bumpImageRefreshToken() {
  imageRefreshToken = String(Date.now());
  persistImageRefreshToken(imageRefreshToken);
}

function hydrateFromStorage() {
  imageRefreshToken = loadStoredImageToken();
  const cachedPayload = loadStoredPayload();

  if (cachedPayload) {
    lastRenderedPayload = cachedPayload;
    render(cachedPayload);
    hasRenderedSuccessfulMatch = true;
    updateStatus(t("statusSynced"));
    return;
  }

  hasRenderedSuccessfulMatch = false;
  lastRenderedPayload = null;
  resetLocalizationCards();
  updateStatus(t("statusWaitingMatch"));
}

function findStageUrl(stages, termGroups) {
  if (!Array.isArray(stages)) return "";

  for (const group of termGroups) {
    const terms = Array.isArray(group) ? group : [group];
    const match = stages.find((stage) => {
      const label = normalizeLabel(stage?.label);
      return terms.every((term) => label.includes(String(term).toLowerCase()));
    });
    if (match && match.url) {
      return String(match.url);
    }
  }
  return "";
}

function resetLocalizationCards() {
  setImage("locAImg", "");
  setImage("locBImg", "");
  setImage("locCImg", "");
  setImage("locDImg", "");
  setText("yoloConfidenceBadge", t("yoloConfidenceUnavailable"));
}

function formatYoloConfidence(localization) {
  const raw = Number(localization?.yolo_confidence);
  if (!Number.isFinite(raw)) return t("yoloConfidenceUnavailable");
  const pct = raw <= 1 ? raw * 100 : raw;
  return `${t("yoloConfidenceLabel")}: ${pct.toFixed(1)}%`;
}

function render(payload) {
  const stages = Array.isArray(payload?.stages) ? payload.stages : [];

  const yoloUrl = findStageUrl(stages, [
    ["localization a"],
    ["01_yolo_guess"],
    ["yolo", "guess"],
  ]);

  const maskUrl = findStageUrl(stages, [
    ["localization b"],
    ["top_bottom_mask_overlay_clean"],
    ["contrast", "mask"],
  ]);

  const axisUrl = findStageUrl(stages, [
    ["localization c"],
    ["axis_side_box_preview"],
    ["axis", "side", "bounds"],
  ]);

  const finalCropUrl =
    findStageUrl(stages, [["localization d"], ["final", "cropped", "image"]]) ||
    String(payload?.camera_image_url || "");

  setImage("locAImg", yoloUrl);
  setImage("locBImg", maskUrl);
  setImage("locCImg", axisUrl);
  setImage("locDImg", finalCropUrl);

  setText("yoloConfidenceBadge", formatYoloConfidence(payload?.localization || {}));
}

function applyLanguage(lang) {
  activeLanguage = lang === "fr" ? "fr" : "en";

  setText("kickerText", t("kicker"));
  setText("heroTitle", t("heroTitle"));
  setText("heroSubText", t("heroSub"));
  setText("summaryTitle", t("summaryTitle"));
  setText("summaryHint", t("summaryHint"));
  setText("locATitle", t("locATitle"));
  setText("locASub", t("locASub"));
  setText("locBTitle", t("locBTitle"));
  setText("locBSub", t("locBSub"));
  setText("locCTitle", t("locCTitle"));
  setText("locCSub", t("locCSub"));
  setText("locDTitle", t("locDTitle"));
  setText("locDSub", t("locDSub"));
  setText("mainPageLink", t("mainPageLink"));
  setText("localizationPageLink", t("localizationPageLink"));
  setText("updateBtn", t("updateBtn"));

  if (langEnBtn) langEnBtn.classList.toggle("active", activeLanguage === "en");
  if (langFrBtn) langFrBtn.classList.toggle("active", activeLanguage === "fr");

  if (lastRenderedPayload) {
    setText("yoloConfidenceBadge", formatYoloConfidence(lastRenderedPayload?.localization || {}));
  } else {
    setText("yoloConfidenceBadge", t("yoloConfidenceUnavailable"));
  }
}

async function fetchLatest() {
  updateStatus(t("statusLoading"));

  try {
    const response = await fetch("/api/latest-match", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const payload = await response.json();
    const data = payload?.data || null;

    if (!payload?.success || !data) {
      if (!hasRenderedSuccessfulMatch) {
        resetLocalizationCards();
      }
      updateStatus(payload?.message || t("statusNoMatches"));
      return;
    }

    bumpImageRefreshToken();
    lastRenderedPayload = data;
    render(data);
    persistLatestPayload(data);
    hasRenderedSuccessfulMatch = true;
    updateStatus(t("statusSynced"));
  } catch (error) {
    if (!hasRenderedSuccessfulMatch) {
      resetLocalizationCards();
    }
    updateStatus(`${t("statusFailedPrefix")} ${error.message}`, true);
  }
}

if (updateBtn) updateBtn.addEventListener("click", fetchLatest);
if (langEnBtn) langEnBtn.addEventListener("click", () => applyLanguage("en"));
if (langFrBtn) langFrBtn.addEventListener("click", () => applyLanguage("fr"));

window.addEventListener("storage", (event) => {
  if (
    event.key !== EXPO_PAYLOAD_STORAGE_KEY &&
    event.key !== EXPO_IMAGE_TOKEN_STORAGE_KEY
  ) {
    return;
  }
  hydrateFromStorage();
});

applyLanguage("en");
hydrateFromStorage();
