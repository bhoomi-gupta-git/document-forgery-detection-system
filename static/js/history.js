/**
 * history.js — S-04 History Dashboard interactions.
 *
 * Handles:
 *   - Initial table load with skeleton rows
 *   - Debounced search (300ms)
 *   - Filter pill toggle (All / Authentic / Forged)
 *   - Sortable column headers (toggle asc/desc)
 *   - Pagination prev/next with scroll-to-top
 *   - Row click → navigate to /result/:id/detail
 *   - Empty state rendering
 *   - S-04 scroll position saved/restored (for BreadcrumbBar back navigation)
 */

'use strict';

(function () {
    const { apiFetch, formatDateTime, debounce, qs, qsa, show, hide } = window.DocForge;

    /* ── State ───────────────────────────────────────────────── */
    let state = {
        page: 1,
        limit: 10,
        verdict: null,   // null | 'authentic' | 'forged'
        sort: 'date',
        dir: 'desc',
        search: '',
        total: 0,
    };

    /* ── Elements ────────────────────────────────────────────── */
    const tableBody = qs('#history-tbody');
    const recordCount = qs('#record-count');
    const searchInput = qs('#search-input');
    const searchClear = qs('#search-clear');
    const filterPills = qsa('.filter-pills__pill');
    const prevBtn = qs('#page-prev');
    const nextBtn = qs('#page-next');
    const pageInfo = qs('#page-info');
    const emptyState = qs('#empty-state');
    const emptyMsg = qs('#empty-message');
    const tableWrapper = qs('#table-wrapper');

    if (!tableBody) return; // not on history screen

    /* ── Initial load ────────────────────────────────────────── */
    fetchHistory();

    /* ── Search ──────────────────────────────────────────────── */
    searchInput?.addEventListener(
        'input',
        debounce((e) => {
            state.search = e.target.value.trim();
            state.page = 1;
            toggleSearchClear();
            fetchHistory();
        }, 300)
    );

    searchClear?.addEventListener('click', () => {
        searchInput.value = '';
        state.search = '';
        state.page = 1;
        toggleSearchClear();
        fetchHistory();
    });

    function toggleSearchClear() {
        if (!searchClear) return;
        if (state.search) {
            searchClear.classList.add('search-input__clear--visible');
        } else {
            searchClear.classList.remove('search-input__clear--visible');
        }
    }

    /* ── Filter pills ────────────────────────────────────────── */
    filterPills.forEach((pill) => {
        pill.addEventListener('click', () => {
            filterPills.forEach((p) => p.classList.remove('filter-pills__pill--active'));
            pill.classList.add('filter-pills__pill--active');

            const val = pill.dataset.verdict || null;
            state.verdict = val === 'all' ? null : val;
            state.page = 1;
            fetchHistory();
        });
    });

    /* ── Column sort ─────────────────────────────────────────── */
    qsa('th[data-sort]').forEach((th) => {
        th.addEventListener('click', () => {
            const col = th.dataset.sort;
            if (state.sort === col) {
                state.dir = state.dir === 'desc' ? 'asc' : 'desc';
            } else {
                state.sort = col;
                state.dir = 'desc';
            }
            state.page = 1;

            // Update sort arrow UI
            qsa('th[data-sort]').forEach((h) => {
                h.classList.remove('sort--active');
                const arrow = h.querySelector('.sort-arrow');
                if (arrow) arrow.textContent = '↕';
            });
            th.classList.add('sort--active');
            const arrow = th.querySelector('.sort-arrow');
            if (arrow) arrow.textContent = state.dir === 'desc' ? '↓' : '↑';

            fetchHistory();
        });
    });

    /* ── Pagination ──────────────────────────────────────────── */
    prevBtn?.addEventListener('click', () => {
        if (state.page > 1) {
            state.page--;
            fetchHistory();
            scrollToTop();
        }
    });

    nextBtn?.addEventListener('click', () => {
        const maxPage = Math.ceil(state.total / state.limit);
        if (state.page < maxPage) {
            state.page++;
            fetchHistory();
            scrollToTop();
        }
    });

    function scrollToTop() {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    /* ── Save scroll position before leaving ─────────────────── */
    document.addEventListener('click', (e) => {
        const row = e.target.closest('.record-table__row[data-doc-id]');
        if (row) {
            sessionStorage.setItem('history_scroll', window.scrollY);
        }
    });

    // Restore scroll position if coming back from detail view
    const savedScroll = sessionStorage.getItem('history_scroll');
    if (savedScroll) {
        window.scrollTo({ top: parseInt(savedScroll, 10) });
        sessionStorage.removeItem('history_scroll');
    }

    /* ── Fetch + render ──────────────────────────────────────── */
    async function fetchHistory() {
        showSkeleton();

        const params = new URLSearchParams({
            page: state.page,
            limit: state.limit,
            sort: state.sort,
            dir: state.dir,
        });
        if (state.verdict) params.set('verdict', state.verdict);
        if (state.search) params.set('q', state.search);

        const { data, ok } = await apiFetch(`/api/history?${params}`);

        if (!ok || !data) {
            showFetchError();
            return;
        }

        state.total = data.total || 0;
        renderTable(data.items || []);
        renderPagination(data.page, data.total, data.limit);

        if (recordCount) {
            recordCount.textContent = `${data.total} record${data.total !== 1 ? 's' : ''}`;
        }
    }

    /* ── Render table rows ───────────────────────────────────── */
    function renderTable(items) {
        if (!tableBody) return;

        if (items.length === 0) {
            tableBody.innerHTML = '';
            showEmptyState();
            return;
        }

        hideEmptyState();

        tableBody.innerHTML = items.map((item, index) => {
            const verdictLabel = item.verdict === 'authentic' ? 'Authentic'
                : item.verdict === 'forged' ? 'Potentially Forged'
                    : 'Unknown';
            const verdictClass = item.verdict === 'authentic' ? 'badge--authentic'
                : item.verdict === 'forged' ? 'badge--forged'
                    : 'badge--neutral';
            const confPct = item.confidence_pct != null ? `${item.confidence_pct}%` : '—';
            const rowNum = (state.page - 1) * state.limit + index + 1;

            return `
        <tr class="record-table__row"
            data-doc-id="${item.document_id || item.result_id}"
            tabindex="0"
            role="button"
            aria-label="View result for ${escapeHtml(item.filename_orig || '')}">
          <td>${rowNum}</td>
          <td class="font-mono" style="font-size:13px">
            ${escapeHtml(item.filename_orig || '—')}
          </td>
          <td>${formatDateTime(item.uploaded_at)}</td>
          <td><span class="badge ${verdictClass}">${verdictLabel}</span></td>
          <td>${confPct}</td>
          <td>
            <a href="/result/${item.document_id || item.result_id}/detail"
               class="text-accent text-sm"
               style="font-weight:500">
              View →
            </a>
          </td>
        </tr>
      `;
        }).join('');

        // Row click navigation
        qsa('.record-table__row[data-doc-id]', tableBody).forEach((row) => {
            row.addEventListener('click', (e) => {
                if (e.target.tagName === 'A') return; // let the link handle it
                const id = row.dataset.docId;
                if (id) window.location.href = `/result/${id}/detail`;
            });
            row.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') row.click();
            });
        });
    }

    /* ── Pagination UI ───────────────────────────────────────── */
    function renderPagination(page, total, limit) {
        const maxPage = Math.max(1, Math.ceil(total / limit));
        if (pageInfo) pageInfo.textContent = `Page ${page} of ${maxPage}`;
        if (prevBtn) prevBtn.disabled = page <= 1;
        if (nextBtn) nextBtn.disabled = page >= maxPage;
    }

    /* ── Skeleton rows ───────────────────────────────────────── */
    function showSkeleton() {
        if (!tableBody) return;
        hideEmptyState();
        tableBody.innerHTML = Array.from({ length: state.limit }, () => `
      <tr class="record-table__row record-table__row--skeleton">
        <td><div class="skeleton-line skeleton-line--xs"></div></td>
        <td><div class="skeleton-line skeleton-line--md"></div></td>
        <td><div class="skeleton-line skeleton-line--sm"></div></td>
        <td><div class="skeleton-line skeleton-line--xs"></div></td>
        <td><div class="skeleton-line skeleton-line--xs"></div></td>
        <td><div class="skeleton-line skeleton-line--xs"></div></td>
      </tr>
    `).join('');
    }

    /* ── Empty state ─────────────────────────────────────────── */
    function showEmptyState() {
        if (emptyState) show(emptyState);
        if (tableWrapper) tableWrapper.style.display = 'none';
        if (emptyMsg) {
            emptyMsg.textContent = state.search || state.verdict
                ? 'No records match your search. Clear filters to see all results.'
                : 'No analyses yet. Upload your first document.';
        }
    }

    function hideEmptyState() {
        if (emptyState) hide(emptyState);
        if (tableWrapper) tableWrapper.style.display = '';
    }

    function showFetchError() {
        if (tableBody) {
            tableBody.innerHTML = `
        <tr><td colspan="6" style="text-align:center;padding:var(--sp-4);color:var(--warn)">
          Could not load history. Check your connection and refresh the page.
        </td></tr>
      `;
        }
    }

    /* ── Helpers ─────────────────────────────────────────────── */
    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }
})();