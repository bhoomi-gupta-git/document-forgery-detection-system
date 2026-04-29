/**
 * progress.js — S-02 Analysis Progress screen.
 *
 * Handles:
 *   - Polling /analyse/status every 1.5s
 *   - Advancing StageIndicator dots
 *   - TimeoutWarning after 12s
 *   - Error card after 15s (or server error)
 *   - Cancel link: abort + DELETE /analyse/:id + navigate S-01
 */

'use strict';

(function () {
    const { apiFetch, qs, show, hide } = window.DocForge;

    /* ── Config ─────────────────────────────────────────────── */
    const POLL_INTERVAL_MS = 1500;
    const TIMEOUT_WARNING_S = 12;
    const TIMEOUT_ERROR_S = 15;

    const STAGE_LABELS = [
        'Resizing and denoising image…',
        'Running ELA analysis…',
        'Running forgery detection model…',
        'Generating region heatmap…',
        'Saving result to database…',
    ];

    /* ── Read result_id from URL ─────────────────────────────── */
    const params = new URLSearchParams(window.location.search);
    const resultId = params.get('result_id');

    if (!resultId) {
        window.location.href = '/';
        return;
    }

    /* ── Elements ────────────────────────────────────────────── */
    const stageLabel = qs('#stage-label');
    const timeoutWarn = qs('#timeout-warning');
    const cancelBtn = qs('#cancel-btn');
    const stageDots = Array.from(document.querySelectorAll('.stage-bar__item'));

    /* ── State ───────────────────────────────────────────────── */
    let pollInterval = null;
    let elapsedS = 0;
    let lastStage = -1;
    let cancelled = false;

    /* ── Start polling ───────────────────────────────────────── */
    pollInterval = setInterval(poll, POLL_INTERVAL_MS);
    poll(); // immediate first call

    // Elapsed timer (drives timeout logic)
    const elapsedTimer = setInterval(() => {
        elapsedS++;

        if (elapsedS >= TIMEOUT_WARNING_S && timeoutWarn) {
            timeoutWarn.classList.add('status-card__timeout--visible');
        }

        if (elapsedS >= TIMEOUT_ERROR_S) {
            clearInterval(elapsedTimer);
            clearInterval(pollInterval);
            showErrorState('Analysis timed out. Please retry with a smaller file.');
        }
    }, 1000);

    /* ── Cancel button ───────────────────────────────────────── */
    cancelBtn?.addEventListener('click', async () => {
        cancelled = true;
        clearInterval(pollInterval);
        clearInterval(elapsedTimer);

        await apiFetch(`/analyse/${resultId}`, { method: 'DELETE' });
        window.location.href = '/';
    });

    /* ── Poll function ───────────────────────────────────────── */
    async function poll() {
        if (cancelled) return;

        const { data, ok } = await apiFetch(`/analyse/status?result_id=${resultId}`);

        if (!ok || !data) {
            console.warn('[progress] Poll failed');
            return;
        }

        const { stage, status } = data;

        // Advance stepper if stage moved forward
        if (stage !== lastStage) {
            setStage(stage);
            lastStage = stage;
        }

        if (status === 'complete') {
            clearInterval(pollInterval);
            clearInterval(elapsedTimer);
            // Smooth fade then navigate
            setTimeout(() => {
                window.location.href = `/result/${resultId}`;
            }, 500);
        }

        if (status === 'error') {
            clearInterval(pollInterval);
            clearInterval(elapsedTimer);
            showErrorState('Analysis failed. Please try again with a different file.');
        }
    }

    /* ── Stage stepper ───────────────────────────────────────── */
    function setStage(stageIndex) {
        stageDots.forEach((item, i) => {
            item.classList.remove('stage-bar__item--active', 'stage-bar__item--complete');
            if (i < stageIndex) item.classList.add('stage-bar__item--complete');
            if (i === stageIndex) item.classList.add('stage-bar__item--active');
        });

        if (stageLabel) {
            stageLabel.textContent = STAGE_LABELS[stageIndex] || STAGE_LABELS[0];
        }
    }

    /* ── Error state ─────────────────────────────────────────── */
    function showErrorState(message) {
        const card = qs('#status-card');
        if (!card) return;

        card.innerHTML = `
      <div class="error-card error-card--error" style="box-shadow:none;border:none;">
        <div class="error-card__icon">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
        </div>
        <h2 class="error-card__title">Analysis Failed</h2>
        <p class="error-card__body">${message}</p>
        <a href="/" class="error-card__action">Retry Upload</a>
      </div>
    `;
    }
})();