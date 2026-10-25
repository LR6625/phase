const updateBtn = document.getElementById("updateBtn");
const statusText = document.getElementById("statusText");
const scryfallStepRow = document.getElementById("scryfallStepRow");
const cameraStepRow = document.getElementById("cameraStepRow");
const combinedStepRow = document.getElementById("combinedStepRow");
const routeGraphic = document.getElementById("routeGraphic");
const routeStatus = document.getElementById("routeStatus");
const routeArrow = document.getElementById("routeArrow");
const routePrice = document.getElementById("routePrice");
const routeLeftStop = document.getElementById("routeLeftStop");
const routeMiddleStop = document.getElementById("routeMiddleStop");
const routeRightStop = document.getElementById("routeRightStop");
const langEnBtn = document.getElementById("langEnBtn");
const langFrBtn = document.getElementById("langFrBtn");
const webControlButtons = document.querySelectorAll("[data-web-control]");
let hasRenderedSuccessfulMatch = false;
let lastRenderedPayload = null;
const LOW_VALUE_MAX_CAD = 0.39;
let activeLanguage = "en";
let imageRefreshToken = "";

const EXPO_PAYLOAD_STORAGE_KEY = "mtgExpo.latestMatchPayload";
const EXPO_IMAGE_TOKEN_STORAGE_KEY = "mtgExpo.imageRefreshToken";

const I18N = {
  en: {
    updateBtn: "Update Example",
    kicker: "Magic: The Gathering Identification Engine",
    heroTitle: "P.H.A.S.E. Console",
    heroSub: "A visual breakdown of camera capture, hash generation, segmented matching and confidence scoring.",
    summaryTitle: "Live Match Summary",
    openScryfall: "Open Scryfall Page",
    labelCardName: "Card Name",
    labelSet: "Set",
    labelOrientation: "Orientation",
    labelDistance: "Distance",
    labelPrice: "Price (CAD)",
    labelSimilarity: "Similarity",
    compareTitle: "Captured Card vs Scryfall Match",
    captionCamera: "Machine Capture (Cropped + Upright)",
    captionScryfall: "Scryfall Ground Truth",
    pipelineTitle: "Hashing Pipeline Board",
    pipelineHint: "Rows are ordered for side-by-side science-fair explanation flow.",
    rowScryfallTitle: "Scryfall Steps 1-4 (Left to Right)",
    rowCameraTitle: "Captured Image Steps 1-4 (Left to Right)",
    rowCombinedTitle: "Step 6 + Step 7: Grid Panels and Segment Hash Match Map",
    confidenceTitle: "Confidence Reactor",
    labelFinalScore: "Final Score:",
    labelWeightedMatches: "Weighted Matches:",
    labelMatchedRegions: "Matched Regions:",
    labelStage1Distance: "Stage-1 Distance:",
    hashTitle: "64-Bit Hash Strings",
    labelCapturedHashHex: "Captured Hash (hex)",
    labelScryfallHashHex: "Scryfall Hash (hex)",
    labelHammingDistance: "Hamming Distance",
    labelCapturedBinary: "Captured 64-Bit Binary",
    labelScryfallBinary: "Scryfall 64-Bit Binary",
    pricePayloadTitle: "Price Payload",
    sortDirectionTitle: "Sort Direction",
    routeLeftLabel: "Left",
    routeMiddleLabel: "Middle",
    routeRightLabel: "Right",
    routeLeftSub: "High Value",
    routeMiddleSub: "Low Value",
    routeRightSub: "Unidentified",
    waitingCardData: "Waiting for card data...",
    statusWaitingMatch: "Waiting for a matched card...",
    statusLoading: "Loading latest successful match...",
    statusNoMatches: "No successful matches available yet.",
    statusSynced: "Display synced with latest successful match.",
    statusFailedPrefix: "Failed to load data:",
    statusControlSentPrefix: "Sent control:",
    statusControlFailedPrefix: "Control send failed:",
    noImagesYet: "No images available for this row yet.",
    stageFallback: "Stage",
    routeRight: "Route: RIGHT (unidentified / non-positive price)",
    routeMiddle: "Route: MIDDLE (low value)",
    routeLeft: "Route: LEFT (high value)",
    orientationUnknown: "Unknown",
    orientationUpright: "Upright",
    orientationFlipped: "180-degree orientation",
    confidenceExplainDefault:
      "Confidence blends weighted region matches, number of matched regions, and stage-1 shortlist distance. Higher weighted matches and more region agreement increase confidence.",
    controlTitleQ: "Mode select (q)",
    controlTitleA: "Automatic mode (a)",
    controlTitleS: "Manual mode (s)",
    controlTitleR: "Demo auto mode (r)",
    controlTitleEsc: "Exit process (ESC)",
  },
  fr: {
    updateBtn: "Mettre a jour",
    kicker: "Moteur d'identification Magic: The Gathering",
    heroTitle: "Console P.H.A.S.E.",
    heroSub: "Un aperçu visuel de la capture caméra, de la génération de hachage, de la correspondance segmentée et du score de confiance.",
    summaryTitle: "Resume de correspondance en direct",
    openScryfall: "Ouvrir la page Scryfall",
    labelCardName: "Nom de la carte",
    labelSet: "Extension",
    labelOrientation: "Orientation",
    labelDistance: "Distance",
    labelPrice: "Prix (CAD)",
    labelSimilarity: "Similarite",
    compareTitle: "Capture machine vs correspondance Scryfall",
    captionCamera: "Capture machine (recadree + redressee)",
    captionScryfall: "Reference Scryfall",
    pipelineTitle: "Tableau du pipeline de hachage",
    pipelineHint: "Lignes ordonnees pour une explication cote a cote.",
    rowScryfallTitle: "Scryfall etapes 1-4 (gauche a droite)",
    rowCameraTitle: "Capture etapes 1-4 (gauche a droite)",
    rowCombinedTitle: "Etapes 6 + 7 : panneaux grille et carte de correspondance",
    confidenceTitle: "Reacteur de confiance",
    labelFinalScore: "Score final :",
    labelWeightedMatches: "Correspondances ponderees :",
    labelMatchedRegions: "Regions correspondantes :",
    labelStage1Distance: "Distance etape 1 :",
    hashTitle: "Chaines de hachage 64 bits",
    labelCapturedHashHex: "Hachage capture (hex)",
    labelScryfallHashHex: "Hachage Scryfall (hex)",
    labelHammingDistance: "Distance de Hamming",
    labelCapturedBinary: "Binaire 64 bits capture",
    labelScryfallBinary: "Binaire 64 bits Scryfall",
    pricePayloadTitle: "Charge utile des prix",
    sortDirectionTitle: "Direction de tri",
    routeLeftLabel: "Gauche",
    routeMiddleLabel: "Milieu",
    routeRightLabel: "Droite",
    routeLeftSub: "Haute valeur",
    routeMiddleSub: "Faible valeur",
    routeRightSub: "Non identifie",
    waitingCardData: "En attente des donnees de carte...",
    statusWaitingMatch: "En attente d'une carte correspondante...",
    statusLoading: "Chargement de la derniere correspondance reussie...",
    statusNoMatches: "Aucune correspondance reussie disponible pour le moment.",
    statusSynced: "Affichage synchronise avec la derniere correspondance reussie.",
    statusFailedPrefix: "Echec du chargement :",
    statusControlSentPrefix: "Commande envoyee :",
    statusControlFailedPrefix: "Echec d'envoi de commande :",
    noImagesYet: "Aucune image disponible pour cette ligne.",
    stageFallback: "Etape",
    routeRight: "Route : DROITE (non identifie / prix non positif)",
    routeMiddle: "Route : MILIEU (faible valeur)",
    routeLeft: "Route : GAUCHE (haute valeur)",
    orientationUnknown: "Inconnue",
    orientationUpright: "Redressee",
    orientationFlipped: "Orientation a 180 degres",
    confidenceExplainDefault:
      "La confiance combine les correspondances regionales ponderees, le nombre de regions correspondantes et la distance de preselection de l'etape 1. Des correspondances ponderees plus elevees et un meilleur accord regional augmentent la confiance.",
    controlTitleQ: "Selection de mode (q)",
    controlTitleA: "Mode automatique (a)",
    controlTitleS: "Mode manuel (s)",
    controlTitleR: "Mode demo auto (r)",
    controlTitleEsc: "Quitter le processus (ESC)",
  },
};

const STAGE_LABELS_FR = {
  "Step 1: Camera Card Image": "Etape 1 : Image carte capturee",
  "Step 1: Scryfall Card Image": "Etape 1 : Image carte Scryfall",
  "Step 2: Camera Grayscale 32x32": "Etape 2 : Niveaux de gris capture 32x32",
  "Step 2: Scryfall Grayscale 32x32": "Etape 2 : Niveaux de gris Scryfall 32x32",
  "Step 3: Camera 8x8 Grayscale": "Etape 3 : Niveaux de gris capture 8x8",
  "Step 3: Scryfall 8x8 Grayscale": "Etape 3 : Niveaux de gris Scryfall 8x8",
  "Step 4: Camera 64-Bit Hash Grid (Diff Highlighted)": "Etape 4 : Grille hachage 64 bits capture (differences)",
  "Step 4: Scryfall 64-Bit Hash Grid (Diff Highlighted)": "Etape 4 : Grille hachage 64 bits Scryfall (differences)",
  "Step 6: Captured Grid + Segment Binary Panels": "Etape 6 : Grille capturee + panneaux binaires",
  "Step 6: Scryfall Grid + Segment Binary Panels": "Etape 6 : Grille Scryfall + panneaux binaires",
  "Step 7: Segment Hash Match Map": "Etape 7 : Carte de correspondance des segments",
};

function t(key) {
  return (I18N[activeLanguage] || I18N.en)[key] || key;
}

function localizeStageLabel(label) {
  const raw = String(label || "").trim();
  if (activeLanguage !== "fr") return raw;
  if (STAGE_LABELS_FR[raw]) return STAGE_LABELS_FR[raw];

  let localized = raw;
  localized = localized.replace(/^Step\s*/i, "Etape ");
  localized = localized.replace("Camera", "Capture");
  localized = localized.replace("Scryfall", "Scryfall");
  localized = localized.replace("Grid + Segment Binary Panels", "Grille + panneaux binaires de segments");
  localized = localized.replace("Segment Hash Match Map", "Carte de correspondance de hachage par segment");
  localized = localized.replace("Grayscale", "Niveaux de gris");
  localized = localized.replace("Diff Highlighted", "Differences surlignees");
  return localized;
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

function setLink(id, href) {
  const el = document.getElementById(id);
  if (!el) return;
  el.href = href || "#";
}

function formatNumber(value, digits = 2) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return n.toFixed(digits);
}

function formatPercent(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return n.toFixed(2);
}

function formatBits(bitString) {
  if (!bitString || typeof bitString !== "string") return "-";
  const clean = bitString.replace(/\s+/g, "");
  if (clean.length !== 64) return bitString;
  const chunks = clean.match(/.{1,8}/g);
  return chunks ? chunks.join(" ") : bitString;
}

function stageDelay(index) {
  return Math.min(0.5, index * 0.06);
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
    render(cachedPayload);
    hasRenderedSuccessfulMatch = true;
    updateStatus(t("statusSynced"));
    return;
  }

  hasRenderedSuccessfulMatch = false;
  resetDashboard();
  updateStatus(t("statusWaitingMatch"));
}

function confidenceDescription(rawScore10, adjustedPercent) {
  if (!Number.isFinite(rawScore10)) return "Confirmed match baseline starts at 99.2%.";
  if (rawScore10 >= 9) return `Confirmed match at ${formatNumber(adjustedPercent, 2)}%. Strong region agreement with a high contribution from the remaining 0.8%.`;
  if (rawScore10 >= 7.5) return `Confirmed match at ${formatNumber(adjustedPercent, 2)}%. High confidence with a moderate contribution from the remaining 0.8%.`;
  if (rawScore10 >= 5.5) return `Confirmed match at ${formatNumber(adjustedPercent, 2)}%. Moderate contribution from the remaining 0.8% confidence budget.`;
  return `Confirmed match at ${formatNumber(adjustedPercent, 2)}%. Baseline 99.2% retained while the remaining 0.8% contribution is lower.`;
}

function computeConfirmedConfidencePercent(rawScore10) {
  const raw = Number(rawScore10);
  const bounded = Number.isFinite(raw) ? Math.max(0, Math.min(10, raw)) : 0;
  // Confirmed match baseline 99.2%; raw /10 contributes remaining 0.8%.
  return 99.2 + (bounded / 10) * 0.8;
}

function applyLanguage(lang) {
  activeLanguage = lang === "fr" ? "fr" : "en";
  setText("kickerText", t("kicker"));
  setText("heroTitle", t("heroTitle"));
  setText("heroSubText", t("heroSub"));
  setText("summaryTitle", t("summaryTitle"));
  setText("labelCardName", t("labelCardName"));
  setText("labelSet", t("labelSet"));
  setText("labelOrientation", t("labelOrientation"));
  setText("labelDistance", t("labelDistance"));
  setText("labelPrice", t("labelPrice"));
  setText("labelSimilarity", t("labelSimilarity"));
  setText("compareTitle", t("compareTitle"));
  setText("captionCamera", t("captionCamera"));
  setText("captionScryfall", t("captionScryfall"));
  setText("pipelineTitle", t("pipelineTitle"));
  setText("pipelineHint", t("pipelineHint"));
  setText("rowScryfallTitle", t("rowScryfallTitle"));
  setText("rowCameraTitle", t("rowCameraTitle"));
  setText("rowCombinedTitle", t("rowCombinedTitle"));
  setText("confidenceTitle", t("confidenceTitle"));
  setText("labelFinalScore", t("labelFinalScore"));
  setText("labelWeightedMatches", t("labelWeightedMatches"));
  setText("labelMatchedRegions", t("labelMatchedRegions"));
  setText("labelStage1Distance", t("labelStage1Distance"));
  setText("hashTitle", t("hashTitle"));
  setText("labelCapturedHashHex", t("labelCapturedHashHex"));
  setText("labelScryfallHashHex", t("labelScryfallHashHex"));
  setText("labelHammingDistance", t("labelHammingDistance"));
  setText("labelCapturedBinary", t("labelCapturedBinary"));
  setText("labelScryfallBinary", t("labelScryfallBinary"));
  setText("pricePayloadTitle", t("pricePayloadTitle"));
  setText("sortDirectionTitle", t("sortDirectionTitle"));
  setText("routeLeftLabel", t("routeLeftLabel"));
  setText("routeMiddleLabel", t("routeMiddleLabel"));
  setText("routeRightLabel", t("routeRightLabel"));
  setText("routeLeftSub", t("routeLeftSub"));
  setText("routeMiddleSub", t("routeMiddleSub"));
  setText("routeRightSub", t("routeRightSub"));
  setText("updateBtn", t("updateBtn"));

  const scryLink = document.getElementById("scryfallLink");
  if (scryLink) scryLink.textContent = t("openScryfall");

  webControlButtons.forEach((btn) => {
    const cmd = String(btn.getAttribute("data-web-control") || "").toLowerCase();
    if (cmd === "q") btn.title = t("controlTitleQ");
    if (cmd === "a") btn.title = t("controlTitleA");
    if (cmd === "s") btn.title = t("controlTitleS");
    if (cmd === "r") btn.title = t("controlTitleR");
    if (cmd === "esc") btn.title = t("controlTitleEsc");
  });

  if (langEnBtn) langEnBtn.classList.toggle("active", activeLanguage === "en");
  if (langFrBtn) langFrBtn.classList.toggle("active", activeLanguage === "fr");

  if (lastRenderedPayload) {
    relocalizeRenderedStageTitles();
    updateRouteGraphic(lastRenderedPayload);
    setText("cardOrientation", formatOrientation(lastRenderedPayload));

    const confidence = lastRenderedPayload.confidence || {};
    const scoreRaw = Number(confidence.score_10);
    const scorePercent = computeConfirmedConfidencePercent(scoreRaw);
    setText(
      "confidenceExplain",
      localizeConfidenceExplanation(confidence.explanation, scoreRaw, scorePercent),
    );
  } else {
    setText("confidenceExplain", t("confidenceExplainDefault"));
  }
}

function updateStatus(message, isError = false) {
  if (!statusText) return;
  statusText.textContent = message;
  statusText.style.color = isError ? "#ffd6dd" : "";
}

function formatOrientation(payload) {
  const angle = Number(payload?.orientation_angle_deg);
  const mode = String(payload?.orientation_mode || "").toLowerCase();
  if (!Number.isFinite(angle)) return t("orientationUnknown");

  if (mode.includes("180")) {
    return `${angle.toFixed(2)} deg (${t("orientationFlipped")})`;
  }
  return `${angle.toFixed(2)} deg (${t("orientationUpright")})`;
}

function localizeConfidenceExplanation(explanation, rawScore10, adjustedPercent) {
  const incoming = String(explanation || "").trim();
  if (!incoming) return confidenceDescription(rawScore10, adjustedPercent);
  const defaultEn = I18N.en.confidenceExplainDefault;
  if (incoming === defaultEn) return t("confidenceExplainDefault");
  return incoming;
}

async function sendWebControl(command) {
  const cmd = String(command || "").trim().toLowerCase();
  if (!["q", "a", "s", "r", "esc"].includes(cmd)) return;

  try {
    const response = await fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command: cmd }),
    });

    const payload = await response.json();
    if (!response.ok || !payload?.success) {
      throw new Error(payload?.message || `HTTP ${response.status}`);
    }
    updateStatus(`${t("statusControlSentPrefix")} ${cmd.toUpperCase()}`);
  } catch (error) {
    updateStatus(`${t("statusControlFailedPrefix")} ${error.message}`, true);
  }
}

function resetConfidence() {
  setText("confidenceScore", "-");
  setText("confidenceFinalScore", "-");
  setText("confidenceWeighted", "-");
  setText("confidenceRegions", "-");
  setText("confidenceStage1", "-");
  setText("confidenceExplain", "-");
  const meter = document.getElementById("confidenceFill");
  if (meter) meter.style.width = "0%";
}


function sortByStepThenLabel(stages) {
  const getStep = (label) => {
    const m = String(label || "").match(/step\s*(\d+)/i);
    return m ? Number(m[1]) : 999;
  };
  return [...(stages || [])].sort((a, b) => {
    const sa = getStep(a?.label);
    const sb = getStep(b?.label);
    if (sa !== sb) return sa - sb;
    return String(a?.label || "").localeCompare(String(b?.label || ""));
  });
}

function renderStageRow(container, stages) {
  if (!container) return;
  container.innerHTML = "";
  if (!Array.isArray(stages) || stages.length === 0) {
    container.innerHTML = `<p class='hint'>${t("noImagesYet")}</p>`;
    return;
  }

  sortByStepThenLabel(stages).forEach((stage, index) => {
    const card = document.createElement("article");
    card.className = "stage-card";
    const labelLower = String(stage?.label || "").toLowerCase();
    if (labelLower.includes("scryfall") || labelLower.includes("segment hash match map")) {
      card.classList.add("scryfall-derived");
    }
    card.style.animationDelay = `${stageDelay(index)}s`;

    const rawLabel = String(stage?.label || t("stageFallback"));
    card.dataset.rawLabel = rawLabel;

    const title = document.createElement("h4");
    title.textContent = localizeStageLabel(rawLabel);

    const image = document.createElement("img");
    image.alt = stage.label || "Pipeline stage image";
    image.src = withCacheBust(stage.url || "");
    image.loading = "lazy";

    card.appendChild(title);
    card.appendChild(image);

    if (stage.caption) {
      const caption = document.createElement("p");
      caption.textContent = stage.caption;
      card.appendChild(caption);
    }

    container.appendChild(card);
  });
}

function relocalizeRenderedStageTitles() {
  const cards = document.querySelectorAll(".pipeline-row .stage-card");
  cards.forEach((card) => {
    const title = card.querySelector("h4");
    if (!title) return;
    const rawLabel = String(card.dataset.rawLabel || title.textContent || t("stageFallback"));
    title.textContent = localizeStageLabel(rawLabel);
  });
}

function stageHasLabel(stage, text) {
  return String(stage?.label || "").toLowerCase().includes(String(text || "").toLowerCase());
}

function clearRouteStops() {
  [routeLeftStop, routeMiddleStop, routeRightStop].forEach((el) => {
    if (el) el.classList.remove("active");
  });
}

function setRouteView(route, message) {
  if (!routeGraphic || !routeArrow) return;
  clearRouteStops();

  let arrowX = "50%";
  if (route === "left") {
    arrowX = "16.5%";
    if (routeLeftStop) routeLeftStop.classList.add("active");
  } else if (route === "middle") {
    arrowX = "50%";
    if (routeMiddleStop) routeMiddleStop.classList.add("active");
  } else if (route === "right") {
    arrowX = "83.5%";
    if (routeRightStop) routeRightStop.classList.add("active");
  }

  routeGraphic.style.setProperty("--route-x", arrowX);
  if (routeStatus) routeStatus.textContent = message || "";
}

function setRoutePrice(cadPrice) {
  if (!routePrice) return;
  const value = parseCadPrice(cadPrice);
  routePrice.textContent = Number.isFinite(value) ? `CAD $${value.toFixed(2)}` : "CAD -";
}

function parseCadPrice(value) {
  if (value === null || value === undefined || value === "") return null;
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function updateRouteGraphic(payload) {
  const cardName = String(payload?.card_name || "").trim();
  const isIdentified = Boolean(cardName && cardName !== "-");
  const cadPrice = parseCadPrice(payload?.card_cad_price);
  setRoutePrice(cadPrice);

  if (!isIdentified || cadPrice === null || cadPrice <= 0) {
    setRouteView("right", t("routeRight"));
    return;
  }

  if (cadPrice <= LOW_VALUE_MAX_CAD) {
    setRouteView("middle", `${t("routeMiddle")}: CAD $${cadPrice.toFixed(2)}`);
    return;
  }

  setRouteView("left", `${t("routeLeft")}: CAD $${cadPrice.toFixed(2)}`);
}

function renderPipelineBoard(stages) {
  const all = Array.isArray(stages) ? stages : [];
  const s1 = all.find((s) => stageHasLabel(s, "Step 1: Scryfall"));
  const s2 = all.find((s) => stageHasLabel(s, "Step 2: Scryfall"));
  const s3 = all.find((s) => stageHasLabel(s, "Step 3: Scryfall"));
  const s4 = all.find((s) => stageHasLabel(s, "Step 4: Scryfall"));
  const scryfallRow = [s1, s2, s3, s4].filter(Boolean);

  const c1 = all.find((s) => stageHasLabel(s, "Step 1: Camera"));
  const c2 = all.find((s) => stageHasLabel(s, "Step 2: Camera"));
  const c3 = all.find((s) => stageHasLabel(s, "Step 3: Camera"));
  const c4 = all.find((s) => stageHasLabel(s, "Step 4: Camera"));
  const cameraRow = [c1, c2, c3, c4].filter(Boolean);

  const gridRow = all.filter((s) => stageHasLabel(s, "Step 6:") && stageHasLabel(s, "Grid + Segment Binary Panels"));
  const segmentRow = all.filter((s) => stageHasLabel(s, "Step 7: Segment Hash Match Map"));
  const combinedRow = [...gridRow, ...segmentRow];

  renderStageRow(scryfallStepRow, scryfallRow);
  renderStageRow(cameraStepRow, cameraRow);
  renderStageRow(combinedStepRow, combinedRow);
}

function resetDashboard() {
  setText("cardName", "-");
  setText("cardSet", "-");
  setText("cardOrientation", "-");
  setText("cardDistance", "-");
  setText("cardCad", "-");
  setText("cameraHashHex", "-");
  setText("scryfallHashHex", "-");
  setText("hammingDistance", "-");
  setText("similarityPercent", "-");
  setText("cameraHashBits", "-");
  setText("scryfallHashBits", "-");
  setText("scryfallPriceBadge", "CAD -");
  setImage("cameraImg", "");
  setImage("scryfallImg", "");
  setLink("scryfallLink", "#");
  resetConfidence();
  renderStageRow(scryfallStepRow, []);
  renderStageRow(cameraStepRow, []);
  renderStageRow(combinedStepRow, []);

  const priceJson = document.getElementById("priceJson");
  if (priceJson) priceJson.textContent = "{}";
  setRoutePrice(null);
  setRouteView("middle", t("waitingCardData"));
}

function render(data) {
  const payload = data || {};
  lastRenderedPayload = payload;
  const confidence = payload.confidence || {};
  const scoreRaw = Number(confidence.score_10);
  const scorePercent = computeConfirmedConfidencePercent(scoreRaw);

  setText("cardName", payload.card_name);
  setText("cardSet", payload.set_name);
  setText("cardOrientation", formatOrientation(payload));
  setText("cardDistance", formatNumber(payload.best_distance, 3));
  setText("cardCad", payload.card_cad_price || "-");

  setLink("scryfallLink", payload.scryfall_url);
  setImage("cameraImg", payload.camera_image_url);
  setImage("scryfallImg", payload.scryfall_image_url);

  setText("cameraHashHex", payload.camera_hash_hex || "-");
  setText("scryfallHashHex", payload.scryfall_hash_hex || "-");
  setText("hammingDistance", formatNumber(payload.hamming_distance, 0));
  setText("similarityPercent", formatPercent(payload.similarity_percent));
  setText("cameraHashBits", formatBits(payload.camera_hash_bits));
  setText("scryfallHashBits", formatBits(payload.scryfall_hash_bits));

  setText("confidenceScore", formatNumber(scorePercent, 2));
  setText("confidenceFinalScore", formatNumber(confidence.final_score, 3));
  setText("confidenceWeighted", formatNumber(confidence.weighted_matches, 3));
  setText("confidenceRegions", String(confidence.matched_regions ?? "-"));
  setText("confidenceStage1", formatNumber(confidence.stage1_distance, 3));
  setText("confidenceExplain", localizeConfidenceExplanation(confidence.explanation, scoreRaw, scorePercent));

  const meter = document.getElementById("confidenceFill");
  if (meter) {
    const pct = Math.max(0, Math.min(100, scorePercent));
    meter.style.width = `${pct}%`;
  }

  const priceJson = document.getElementById("priceJson");
  if (priceJson) {
    priceJson.textContent = JSON.stringify(payload.price_payload || {}, null, 2);
  }

  updateRouteGraphic(payload);

  setText("scryfallPriceBadge", payload.card_cad_price ? `CAD ${payload.card_cad_price}` : "CAD -");
  renderPipelineBoard(payload.stages || []);
}

async function fetchLatest() {
  updateStatus(t("statusLoading"));

  try {
    const response = await fetch("/api/latest-match", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const payload = await response.json();
    const data = payload?.data || (payload?.card ? payload : null);
    if (!payload.success || !data) {
      if (!hasRenderedSuccessfulMatch) {
        resetDashboard();
      }
      updateStatus(payload.message || t("statusNoMatches"));
      return;
    }

    bumpImageRefreshToken();
    render(data);
    persistLatestPayload(data);
    hasRenderedSuccessfulMatch = true;
    updateStatus(t("statusSynced"));
  } catch (error) {
    if (!hasRenderedSuccessfulMatch) {
      resetDashboard();
    }
    updateStatus(`${t("statusFailedPrefix")} ${error.message}`, true);
  }
}

if (updateBtn) updateBtn.addEventListener("click", fetchLatest);
if (langEnBtn) langEnBtn.addEventListener("click", () => applyLanguage("en"));
if (langFrBtn) langFrBtn.addEventListener("click", () => applyLanguage("fr"));

webControlButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    sendWebControl(btn.getAttribute("data-web-control"));
  });
});

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
