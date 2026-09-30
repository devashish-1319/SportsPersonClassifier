"use strict";

const API_URL = "/classify_image";
const MAX_UPLOAD_SIDE = 1280; // downscale before upload: faster, and the server caps it anyway

const PLAYERS = {
    lionel_messi: { name: "Lionel Messi", img: "./images/messi.jpeg" },
    maria_sharapova: { name: "Maria Sharapova", img: "./images/sharapova.jpeg" },
    roger_federer: { name: "Roger Federer", img: "./images/federer.jpeg" },
    serena_williams: { name: "Serena Williams", img: "./images/serena.jpeg" },
    virat_kohli: { name: "Virat Kohli", img: "./images/virat.jpeg" },
};
const prettyName = (key) =>
    (PLAYERS[key] && PLAYERS[key].name) || key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
};

const dropzone = $("dropzone"), fileInput = $("fileInput"), stage = $("stage"), preview = $("preview");
const boxesEl = $("boxes"), scanEl = $("scan"), dzEmpty = $("dzEmpty");
const classifyBtn = $("classifyBtn"), clearBtn = $("clearBtn");
const placeholder = $("placeholder"), messageEl = $("message"), resultsEl = $("results");

let currentDataUrl = null;
let busy = false;

/* ---------- image intake ---------- */

function readAsDataUrl(blob) {
    return new Promise((resolve, reject) => {
        const r = new FileReader();
        r.onload = () => resolve(r.result);
        r.onerror = reject;
        r.readAsDataURL(blob);
    });
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error("decode"));
        img.src = src;
    });
}

/** Re-encode as a JPEG no larger than MAX_UPLOAD_SIDE. Also normalises formats like WebP/HEIC-in-Safari. */
async function normalise(dataUrl) {
    const img = await loadImage(dataUrl);
    const scale = Math.min(1, MAX_UPLOAD_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(img.naturalWidth * scale);
    canvas.height = Math.round(img.naturalHeight * scale);
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.9);
}

async function setImage(blobOrUrl) {
    try {
        const raw = typeof blobOrUrl === "string" ? await urlToDataUrl(blobOrUrl) : await readAsDataUrl(blobOrUrl);
        currentDataUrl = await normalise(raw);
    } catch {
        showMessage("That file doesn't look like an image we can read. Try a JPG, PNG or WebP.", "error");
        return;
    }
    preview.onload = syncBoxLayer;
    preview.src = currentDataUrl;
    boxesEl.replaceChildren();
    stage.hidden = false;
    dzEmpty.hidden = true;
    dropzone.classList.add("has-image");
    clearBtn.hidden = false;
    classifyBtn.disabled = false;
    resetResults();
}

async function urlToDataUrl(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error("fetch");
    return readAsDataUrl(await res.blob());
}

function clearImage() {
    currentDataUrl = null;
    preview.removeAttribute("src");
    boxesEl.replaceChildren();
    stage.hidden = true;
    dzEmpty.hidden = false;
    dropzone.classList.remove("has-image");
    clearBtn.hidden = true;
    classifyBtn.disabled = true;
    fileInput.value = "";
    resetResults();
}

/* The box layer must sit exactly over the rendered <img>, so percentages map to the picture. */
function syncBoxLayer() {
    const s = stage.getBoundingClientRect(), i = preview.getBoundingClientRect();
    Object.assign(boxesEl.style, {
        left: i.left - s.left + "px", top: i.top - s.top + "px", width: i.width + "px", height: i.height + "px",
    });
}
window.addEventListener("resize", syncBoxLayer);

/* ---------- results ---------- */

function resetResults() {
    resultsEl.hidden = true;
    resultsEl.replaceChildren();
    messageEl.hidden = true;
    placeholder.hidden = false;
    document.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
}

function showMessage(text, kind) {
    placeholder.hidden = true;
    resultsEl.hidden = true;
    messageEl.className = "message " + kind;
    messageEl.textContent = text;
    messageEl.hidden = false;
}

function confidenceRing(value, known) {
    const r = 30, c = 2 * Math.PI * r;
    const wrap = el("div", "ring" + (known ? "" : " unknown"));
    wrap.innerHTML =
        `<svg width="72" height="72" viewBox="0 0 72 72"><circle class="track" cx="36" cy="36" r="${r}"/>` +
        `<circle class="value" cx="36" cy="36" r="${r}" stroke-dasharray="${c}" stroke-dashoffset="${c}"/></svg>`;
    wrap.appendChild(el("div", "ring-num", Math.round(value) + "%"));
    // animate on next frame so the transition runs
    requestAnimationFrame(() => requestAnimationFrame(() => {
        wrap.querySelector(".value").style.strokeDashoffset = c * (1 - value / 100);
    }));
    return wrap;
}

function faceCard(face, index, total) {
    const card = el("article", "face-card");
    card.style.animationDelay = index * 90 + "ms";
    card.dataset.index = index;

    const head = el("div", "face-head");
    const avatar = el("div", "avatar");
    if (face.known && PLAYERS[face.class]) {
        const img = el("img");
        img.src = PLAYERS[face.class].img;
        img.alt = prettyName(face.class);
        avatar.appendChild(img);
    } else {
        avatar.appendChild(el("div", "unknown-av", "?"));
    }
    const id = el("div", "face-id");
    id.appendChild(el("p", "face-label", total > 1 ? `Face ${index + 1}` : "Best match"));
    id.appendChild(el("h3", "face-name" + (face.known ? "" : " unknown"), face.known ? prettyName(face.class) : "Not sure — unknown"));
    head.append(avatar, id, confidenceRing(face.confidence, face.known));
    card.appendChild(head);

    const probs = el("div", "probs");
    Object.entries(face.probabilities)
        .sort((a, b) => b[1] - a[1])
        .forEach(([name, p], i) => {
            const row = el("div", "prob-row" + (i === 0 ? " top" : ""));
            const fill = el("div", "prob-fill");
            const bar = el("div", "prob-bar");
            bar.appendChild(fill);
            row.append(el("span", "prob-name", prettyName(name)), bar, el("span", "prob-val", p.toFixed(1) + "%"));
            probs.appendChild(row);
            requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = p + "%"; }));
        });
    card.appendChild(probs);

    const setHighlight = (on) => {
        card.classList.toggle("highlight", on);
        const box = boxesEl.querySelector(`[data-index="${index}"]`);
        if (box) box.classList.toggle("highlight", on);
    };
    card.addEventListener("mouseenter", () => setHighlight(true));
    card.addEventListener("mouseleave", () => setHighlight(false));
    return card;
}

function renderResults(data) {
    const faces = data.faces || [];
    if (faces.length === 0) {
        showMessage("No face found. Try a clearer photo where the face is visible and reasonably large.", "warn");
        return;
    }
    placeholder.hidden = true;
    messageEl.hidden = true;
    resultsEl.replaceChildren();

    syncBoxLayer();
    boxesEl.replaceChildren();
    const [imgW, imgH] = data.image_size;
    faces.forEach((f, i) => {
        const [x, y, w, h] = f.box;
        const box = el("div", "face-box" + (f.known ? "" : " unknown"));
        box.dataset.index = i;
        box.style.cssText = `left:${x / imgW * 100}%;top:${y / imgH * 100}%;width:${w / imgW * 100}%;height:${h / imgH * 100}%;animation-delay:${i * 90}ms`;
        box.appendChild(el("span", "face-tag", i + 1));
        boxesEl.appendChild(box);
    });

    const known = faces.filter((f) => f.known);
    resultsEl.appendChild(el("p", "summary",
        `${faces.length} face${faces.length > 1 ? "s" : ""} detected` +
        (known.length ? ` · ${known.length} recognised` : " · none recognised")));
    faces.forEach((f, i) => resultsEl.appendChild(faceCard(f, i, faces.length)));
    resultsEl.hidden = false;

    const names = new Set(known.map((f) => f.class));
    document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", names.has(c.dataset.player)));
}

async function classify() {
    if (!currentDataUrl || busy) return;
    busy = true;
    classifyBtn.disabled = true;
    classifyBtn.querySelector(".btn-label").textContent = "Analysing…";
    classifyBtn.querySelector(".btn-spinner").hidden = false;
    scanEl.hidden = false;
    boxesEl.replaceChildren();
    resetResults();

    try {
        const res = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({ image_data: currentDataUrl }),
        });
        let data = null;
        try { data = await res.json(); } catch { /* non-JSON error page */ }
        if (!res.ok) throw new Error((data && data.error) || `Server error (${res.status}).`);
        renderResults(data);
    } catch (err) {
        const offline = err instanceof TypeError; // fetch network failure
        showMessage(offline ? "Couldn't reach the classifier server. Is it running?" : err.message, "error");
    } finally {
        busy = false;
        scanEl.hidden = true;
        classifyBtn.disabled = !currentDataUrl;
        classifyBtn.querySelector(".btn-label").textContent = "Classify";
        classifyBtn.querySelector(".btn-spinner").hidden = true;
    }
}

/* ---------- events ---------- */

dropzone.addEventListener("click", () => { if (!currentDataUrl) fileInput.click(); });
dropzone.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !currentDataUrl) { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", () => fileInput.files[0] && setImage(fileInput.files[0]));

["dragenter", "dragover"].forEach((t) => dropzone.addEventListener(t, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((t) => dropzone.addEventListener(t, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", (e) => {
    const file = [...e.dataTransfer.files].find((f) => f.type.startsWith("image/"));
    if (file) setImage(file);
});

document.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) setImage(item.getAsFile());
});

document.querySelectorAll(".sample").forEach((b) => b.addEventListener("click", () => setImage(b.dataset.src)));
classifyBtn.addEventListener("click", classify);
clearBtn.addEventListener("click", clearImage);
