/**
 * results.js — S-03 Results and S-05 Saved Detail interactions.
 *
 * Handles:
 *   - Image lightbox (click to expand, ESC / click-outside to close)
 *   - ForgeryTypeRow hover tooltip
 *   - Annotated image missing fallback
 */

'use strict';

(function () {
    const { qs, qsa } = window.DocForge;

    /* ── Lightbox ────────────────────────────────────────────── */
    const lightbox = qs('#lightbox');
    const lightboxImg = qs('#lightbox-img');
    const lightboxClose = qs('#lightbox-close');

    function openLightbox(src) {
        if (!lightbox || !lightboxImg) return;
        lightboxImg.src = src;
        lightbox.classList.add('lightbox--visible');
        document.body.style.overflow = 'hidden';
    }

    function closeLightbox() {
        if (!lightbox) return;
        lightbox.classList.remove('lightbox--visible');
        document.body.style.overflow = '';
    }

    // Click on comparison panels to open lightbox
    qsa('.img-comparison__panel[data-lightbox]').forEach((panel) => {
        panel.addEventListener('click', () => {
            const src = panel.dataset.lightbox;
            if (src) openLightbox(src);
        });
    });

    // ESC key to close
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeLightbox();
    });

    // Click outside image to close
    lightbox?.addEventListener('click', (e) => {
        if (e.target === lightbox) closeLightbox();
    });

    lightboxClose?.addEventListener('click', closeLightbox);

    /* ── Annotated image fallback ────────────────────────────── */
    const annotatedPanel = qs('#annotated-panel');
    const annotatedImg = annotatedPanel?.querySelector('img');

    if (annotatedImg) {
        annotatedImg.addEventListener('error', () => {
            if (annotatedPanel) {
                annotatedPanel.innerHTML = `
          <div class="img-comparison__placeholder">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="1.5" style="color:var(--rule)">
              <rect x="3" y="3" width="18" height="18" rx="2"/>
              <path d="m3 9 4-4 4 4 4-4 4 4"/>
            </svg>
            <span style="font-size:var(--text-xs);color:var(--muted)">
              Overlay image unavailable
            </span>
            <button onclick="location.reload()"
                    style="font-size:var(--text-xs);color:var(--accent);
                           background:none;border:none;cursor:pointer;
                           text-decoration:underline;">
              Retry
            </button>
          </div>
          <div class="img-comparison__caption">Detected Regions</div>
        `;
                // Remove lightbox trigger since image is gone
                annotatedPanel.removeAttribute('data-lightbox');
                annotatedPanel.style.cursor = 'default';
            }
        });
    }
})();