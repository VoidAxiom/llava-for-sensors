/* llava-for-sensors · docs/script.js
 *
 * Vanilla JS. No build. Three responsibilities:
 *   1. Theme toggle (light / dark) with localStorage persistence and
 *      a system-preference fallback for first visit.
 *   2. Depth-tab switching inside each .algo-card (high / medium / low).
 *   3. Mermaid + KaTeX initialization, with re-render on theme change
 *      so diagrams pick up the new colors.
 */

(function () {
  'use strict';

  /* ---------------- Mermaid: pre-init + raw-text capture ----------------
   *
   * Do this BEFORE applyTheme runs, so we (a) suppress mermaid's
   * startOnLoad auto-init (which would replace each .mermaid block's
   * innerHTML with rendered SVG) and (b) snapshot every block's raw
   * graph definition exactly once. The prior implementation captured
   * lazily on first toggle, which only worked because mermaid's
   * startOnLoad hadn't fired yet — fragile w.r.t. script ordering and
   * any future change to mermaid's auto-init timing.
   */
  if (window.mermaid) {
    try {
      window.mermaid.initialize({ startOnLoad: false });
    } catch (err) {
      console.warn('mermaid.initialize failed:', err);
    }
  }
  document.querySelectorAll('.mermaid').forEach((node) => {
    if (!node.dataset.original) {
      node.dataset.original = node.innerHTML;
    }
  });

  /* ---------------- Theme toggle ---------------- */

  const root        = document.documentElement;
  const toggleBtn   = document.getElementById('theme-toggle');
  const STORAGE_KEY = 'llava-for-sensors:theme';

  function preferredTheme() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'light' || stored === 'dark') return stored;
    if (window.matchMedia &&
        window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    return 'light';
  }

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    if (toggleBtn) {
      toggleBtn.setAttribute(
        'aria-label',
        theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'
      );
    }
    if (window.mermaid) {
      try {
        window.mermaid.initialize({
          startOnLoad: false,
          theme: theme === 'dark' ? 'dark' : 'default',
          themeVariables: {
            fontFamily: 'JetBrains Mono, SF Mono, Menlo, monospace',
            fontSize:   '14px',
          },
        });
        const nodes = document.querySelectorAll('.mermaid');
        nodes.forEach((node) => {
          if (node.dataset.original) {
            node.innerHTML = node.dataset.original;
          }
          node.removeAttribute('data-processed');
        });
        window.mermaid.run({ nodes });
      } catch (err) {
        console.warn('mermaid theme re-render failed:', err);
      }
    }
  }

  applyTheme(preferredTheme());

  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const next = root.getAttribute('data-theme') === 'dark'
        ? 'light' : 'dark';
      localStorage.setItem(STORAGE_KEY, next);
      applyTheme(next);
    });
  }

  // Respect later system changes only when the user hasn't explicitly
  // chosen a theme this session.
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)')
      .addEventListener('change', (ev) => {
        if (!localStorage.getItem(STORAGE_KEY)) {
          applyTheme(ev.matches ? 'dark' : 'light');
        }
      });
  }

  /* ---------------- Depth-tab switching ---------------- */

  document.querySelectorAll('.algo-card').forEach((card) => {
    const tabs   = card.querySelectorAll('.depth-tab');
    const panels = card.querySelectorAll('.depth-panel');

    tabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        const target = tab.dataset.depth;
        tabs.forEach((t) => {
          const active = t.dataset.depth === target;
          t.classList.toggle('is-active', active);
          t.setAttribute('aria-selected', active ? 'true' : 'false');
        });
        panels.forEach((p) => {
          p.classList.toggle('is-active', p.dataset.depth === target);
        });
      });

      tab.addEventListener('keydown', (ev) => {
        if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
        ev.preventDefault();
        const order   = Array.from(tabs);
        const current = order.indexOf(tab);
        const next    = ev.key === 'ArrowRight'
          ? (current + 1) % order.length
          : (current - 1 + order.length) % order.length;
        order[next].focus();
        order[next].click();
      });
    });
  });

  /* ---------------- KaTeX auto-render ---------------- */

  let mathAttempts = 0;
  const MATH_MAX_ATTEMPTS = 50; // ~5 s @ 100 ms — give up rather than poll forever
  function tryRenderMath() {
    if (window.renderMathInElement) {
      window.renderMathInElement(document.body, {
        delimiters: [
          { left: '\\[', right: '\\]', display: true },
          { left: '$$', right: '$$', display: true },
          { left: '\\(', right: '\\)', display: false },
        ],
        throwOnError: false,
      });
      return;
    }
    if (++mathAttempts >= MATH_MAX_ATTEMPTS) {
      console.warn(
        'KaTeX never loaded after',
        MATH_MAX_ATTEMPTS * 100,
        'ms; math will not render.'
      );
      return;
    }
    setTimeout(tryRenderMath, 100);
  }
  // Defer until after the page settles.
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', tryRenderMath);
  } else {
    tryRenderMath();
  }
})();
