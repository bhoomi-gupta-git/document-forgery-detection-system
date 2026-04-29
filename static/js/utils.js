/**
 * utils.js — Shared helpers and fetch wrapper.
 * All other JS modules import from here for API calls.
 */

'use strict';

/* ── Fetch wrapper ───────────────────────────────────────── */

/**
 * Fetch JSON from a URL with a .catch() safety net.
 * @param {string} url
 * @param {RequestInit} options
 * @returns {Promise<{data: any, ok: boolean, status: number}>}
 */
async function apiFetch(url, options = {}) {
    const defaults = {
        headers: { 'Accept': 'application/json' },
    };
    const config = { ...defaults, ...options };
    if (config.body && !(config.body instanceof FormData)) {
        config.headers['Content-Type'] = 'application/json';
    }

    try {
        const response = await fetch(url, config);
        const contentType = response.headers.get('Content-Type') || '';
        let data = null;
        if (contentType.includes('application/json')) {
            data = await response.json();
        }
        return { data, ok: response.ok, status: response.status };
    } catch (err) {
        console.error('[apiFetch] Network error:', url, err);
        return { data: null, ok: false, status: 0, error: err };
    }
}

/* ── Formatting helpers ──────────────────────────────────── */

/**
 * Format bytes as human-readable string e.g. "2.3 MB"
 * @param {number} bytes
 * @returns {string}
 */
function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Truncate a filename to maxLen characters keeping the extension.
 * @param {string} name
 * @param {number} maxLen
 * @returns {string}
 */
function truncateFilename(name, maxLen = 40) {
    if (name.length <= maxLen) return name;
    const ext = name.lastIndexOf('.') > 0 ? name.slice(name.lastIndexOf('.')) : '';
    const stem = name.slice(0, maxLen - ext.length - 3);
    return `${stem}…${ext}`;
}

/**
 * Format an ISO-8601 UTC timestamp to a localised readable string.
 * @param {string} isoString
 * @returns {string}
 */
function formatDateTime(isoString) {
    if (!isoString) return '—';
    try {
        return new Intl.DateTimeFormat(undefined, {
            dateStyle: 'medium',
            timeStyle: 'short',
        }).format(new Date(isoString));
    } catch {
        return isoString;
    }
}

/**
 * Format a confidence decimal (0.0–1.0) as a percentage string e.g. "87%"
 * @param {number} value
 * @returns {string}
 */
function formatConfidence(value) {
    if (value == null) return '—';
    return `${Math.round(value * 100)}%`;
}

/* ── Debounce ────────────────────────────────────────────── */

/**
 * Return a debounced version of fn that fires after `delay` ms of inactivity.
 * @param {Function} fn
 * @param {number} delay
 * @returns {Function}
 */
function debounce(fn, delay) {
    let timer;
    return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}

/* ── DOM helpers ─────────────────────────────────────────── */

/** Get a single element by selector. */
function qs(selector, root = document) {
    return root.querySelector(selector);
}

/** Get all elements matching a selector. */
function qsa(selector, root = document) {
    return Array.from(root.querySelectorAll(selector));
}

/** Show an element (removes 'hidden' class, sets display). */
function show(el) {
    if (el) el.classList.remove('hidden');
}

/** Hide an element (adds 'hidden' class). */
function hide(el) {
    if (el) el.classList.add('hidden');
}

/* ── Network offline detection ───────────────────────────── */

(function initOfflineDetection() {
    const toast = document.getElementById('offline-toast');
    if (!toast) return;

    function update() {
        if (navigator.onLine) {
            toast.classList.remove('offline-toast--visible');
        } else {
            toast.classList.add('offline-toast--visible');
        }
    }

    window.addEventListener('online', update);
    window.addEventListener('offline', update);
    update(); // check on load
})();

/* ── Exports (available globally in all pages) ───────────── */
window.DocForge = window.DocForge || {};
Object.assign(window.DocForge, {
    apiFetch,
    formatFileSize,
    truncateFilename,
    formatDateTime,
    formatConfidence,
    debounce,
    qs,
    qsa,
    show,
    hide,
});