/**
 * upload.js — S-01 Home / Upload screen interactions.
 *
 * Handles:
 *   - Drag-and-drop onto DropZone
 *   - Native file picker click
 *   - Client-side MIME type + size validation
 *   - FileChip appearance / clear
 *   - AnalyseButton enabled/disabled state
 *   - POST /api/upload with spinner
 *   - ValidationToast auto-dismiss
 */

'use strict';

(function () {
    const { formatFileSize, truncateFilename, apiFetch, qs, show, hide } = window.DocForge;

    /* ── Config ──────────────────────────────────────────────── */
    const MAX_SIZE_MB = 10;
    const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;
    const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'application/pdf'];
    const ALLOWED_EXTS = ['jpg', 'jpeg', 'png', 'pdf'];
    const TOAST_DURATION = 4000;

    /* ── State ───────────────────────────────────────────────── */
    let selectedFile = null;

    /* ── Elements ────────────────────────────────────────────── */
    const dropZone = qs('#drop-zone');
    const fileInput = qs('#file-input');
    const fileChip = qs('#file-chip');
    const chipName = qs('#chip-name');
    const chipSize = qs('#chip-size');
    const chipClear = qs('#chip-clear');
    const analyseBtn = qs('#analyse-btn');
    const toast = qs('#validation-toast');
    const toastMsg = qs('#toast-message');

    if (!dropZone) return; // not on the upload screen

    /* ── Drag-and-drop events ────────────────────────────────── */
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drop-zone--hover');
    });

    dropZone.addEventListener('dragleave', (e) => {
        if (!dropZone.contains(e.relatedTarget)) {
            dropZone.classList.remove('drop-zone--hover');
        }
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drop-zone--hover');
        const file = e.dataTransfer?.files?.[0];
        if (file) handleFile(file);
    });

    /* ── File picker click ───────────────────────────────────── */
    dropZone.addEventListener('click', (e) => {
        // Don't re-open picker if clicking the clear button
        if (e.target.closest('#chip-clear')) return;
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        const file = fileInput.files?.[0];
        if (file) handleFile(file);
        fileInput.value = ''; // reset so same file can be re-selected
    });

    /* ── FileChip clear ──────────────────────────────────────── */
    chipClear?.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFile();
    });

    /* ── Analyse button submit ───────────────────────────────── */
    analyseBtn?.addEventListener('click', async () => {
        if (!selectedFile || analyseBtn.disabled) return;
        await submitFile();
    });

    /* ── Core file handler ───────────────────────────────────── */
    function handleFile(file) {
        // Validate extension
        const ext = file.name.split('.').pop().toLowerCase();
        if (!ALLOWED_EXTS.includes(ext)) {
            showToast('Only JPG, PNG, or single-page PDF files are accepted.');
            dropZone.classList.add('drop-zone--error');
            return;
        }

        // Validate MIME type (where browser provides it)
        if (file.type && !ALLOWED_TYPES.includes(file.type)) {
            showToast('Only JPG, PNG, or single-page PDF files are accepted.');
            dropZone.classList.add('drop-zone--error');
            return;
        }

        // Validate size
        if (file.size > MAX_SIZE_BYTES) {
            showToast(`File size (${formatFileSize(file.size)}) exceeds the ${MAX_SIZE_MB} MB limit.`);
            dropZone.classList.add('drop-zone--error');
            return;
        }

        // Valid file
        dropZone.classList.remove('drop-zone--error');
        hideToast();
        setFile(file);
    }

    function setFile(file) {
        selectedFile = file;

        // Show FileChip
        if (chipName) chipName.textContent = truncateFilename(file.name, 40);
        if (chipSize) chipSize.textContent = formatFileSize(file.size);
        if (fileChip) show(fileChip);

        // Switch DropZone to active state
        dropZone.classList.add('drop-zone--active');

        // Enable button
        enableButton();
    }

    function clearFile() {
        selectedFile = null;
        if (fileChip) hide(fileChip);
        dropZone.classList.remove('drop-zone--active', 'drop-zone--error');
        disableButton();
        hideToast();
    }

    /* ── Submit ──────────────────────────────────────────────── */
    async function submitFile() {
        setButtonLoading(true);

        const formData = new FormData();
        formData.append('file', selectedFile);

        const { data, ok, status } = await apiFetch('/api/upload', {
            method: 'POST',
            body: formData,
        });

        setButtonLoading(false);

        if (!ok) {
            const msg = data?.error || 'Upload failed. Please try again.';
            showToast(msg);
            return;
        }

        if (data?.result_id) {
            // Navigate to progress screen
            window.location.href = `/analyse?result_id=${data.result_id}`;
        }
    }

    /* ── Button state helpers ────────────────────────────────── */
    function enableButton() {
        if (!analyseBtn) return;
        analyseBtn.disabled = false;
        analyseBtn.classList.remove('btn-primary--disabled');
        analyseBtn.innerHTML = 'Analyse Document';
    }

    function disableButton() {
        if (!analyseBtn) return;
        analyseBtn.disabled = true;
        analyseBtn.classList.add('btn-primary--disabled');
    }

    function setButtonLoading(loading) {
        if (!analyseBtn) return;
        analyseBtn.disabled = loading;
        if (loading) {
            analyseBtn.classList.add('btn-primary--loading');
            analyseBtn.innerHTML = `<span class="spinner"></span> Analysing…`;
        } else {
            analyseBtn.classList.remove('btn-primary--loading');
            analyseBtn.innerHTML = 'Analyse Document';
        }
    }

    /* ── Toast helpers ───────────────────────────────────────── */
    let toastTimer = null;

    function showToast(message) {
        if (!toast || !toastMsg) return;
        toastMsg.textContent = message;
        toast.classList.add('toast--visible');
        clearTimeout(toastTimer);
        toastTimer = setTimeout(hideToast, TOAST_DURATION);
    }

    function hideToast() {
        if (!toast) return;
        toast.classList.remove('toast--visible');
    }
})();