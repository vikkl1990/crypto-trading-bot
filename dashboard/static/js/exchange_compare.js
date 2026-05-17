/* Exchange Compare tab — Delta India vs Bybit (vs Paper baseline)
 * Renders into #exchange-compare-panel inside the new "Exchange Compare" tab.
 * 2026-04-26
 */
(function () {
  const PANEL_ID = 'exchange-compare-panel';
  const REFRESH_MS = 30_000;
  let currentDays = 7;

  function escHtml(s) {
    if (s == null) return '';
    return String(s).replace(/[&<>"']/g, c => ({
      '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
    }[c]));
  }

  function fmt(v, d) {
    if (v == null || isNaN(v)) return '—';
    return Number(v).toFixed(d != null ? d : 2);
  }
  function fmtMoney(v) {
    if (v == null || isNaN(v)) return '—';
    const n = Number(v);
    const sign = n >= 0 ? '+' : '−';
    const abs = Math.abs(n);
    if (abs >= 1000) return sign + '$' + (abs / 1000).toFixed(2) + 'k';
    return sign + '$' + abs.toFixed(2);
  }
  function pnlClass(v) {
    if (v == null) return '';
    if (v > 0) return 'xc-cell--val--ok';
    if (v < 0) return 'xc-cell--val--danger';
    return '';
  }
  function makerClass(p) {
    if (p == null) return '';
    if (p >= 30) return 'xc-cell--val--ok';
    if (p >= 5)  return 'xc-cell--val--warn';
    return 'xc-cell--val--danger';
  }
  function shareClass(s) {
    if (s == null) return '';
    if (s >= 0.5) return 'xc-cell--val--ok';
    if (s >= 0)   return '';
    return 'xc-cell--val--danger';
  }

  function pickWinner(metric, vals, higherBetter = true) {
    // vals: { delta_india: x, bybit: y } — return key whose value is best
    let best = null, bestVal = null;
    for (const [k, v] of Object.entries(vals)) {
      if (v == null || isNaN(v)) continue;
      if (best === null || (higherBetter ? v > bestVal : v < bestVal)) {
        best = k; bestVal = v;
      }
    }
    return best;
  }

  function buildVerdict(byExch) {
    const delta = byExch.delta_india;
    const bybit = byExch.bybit;
    if (!delta || !bybit) {
      return null;
    }
    if (delta.n < 5 || bybit.n < 5) {
      return `Need ≥5 trades per exchange to call a verdict. Currently delta=${delta.n}, bybit=${bybit.n}.`;
    }
    const pnlDelta = delta.pnl_total - bybit.pnl_total;
    const winnerPnl = pnlDelta > 0 ? 'Delta India' : 'Bybit';
    const lift = Math.abs(pnlDelta);
    const makerDeltaPp = (bybit.maker_pct || 0) - (delta.maker_pct || 0);
    const wrDeltaPp = (bybit.wr_pct || 0) - (delta.wr_pct || 0);

    let verdict = `Over the last ${currentDays}d, <strong>${winnerPnl}</strong> earned <strong>${fmtMoney(lift)}</strong> more `;
    verdict += `(Δ${pnlDelta > 0 ? '+' : '−'}$${Math.abs(pnlDelta).toFixed(2)}). `;
    verdict += `Maker rate: Bybit ${(bybit.maker_pct || 0).toFixed(0)}% vs Delta ${(delta.maker_pct || 0).toFixed(0)}% (${makerDeltaPp >= 0 ? '+' : ''}${makerDeltaPp.toFixed(0)}pp). `;
    verdict += `WR delta: ${wrDeltaPp >= 0 ? '+' : ''}${wrDeltaPp.toFixed(1)}pp.`;
    return verdict;
  }

  function render(d) {
    const el = document.getElementById(PANEL_ID);
    if (!el) return;
    const byExch = d.by_exchange || {};
    const paper = d.overall_paper || {};
    const delta = byExch.delta_india || {};
    const bybit = byExch.bybit || {};

    if ((!delta.n || delta.n === 0) && (!bybit.n || bybit.n === 0)) {
      el.innerHTML = `
        <div class="xc-header">
          <span class="xc-title">Exchange Compare — last ${currentDays}d</span>
          <div class="xc-controls">${daysSelector()}</div>
        </div>
        <div class="xc-empty">No closed shadow/real trades yet. Bybit simulator runs every minute — wait a few minutes.</div>
      `;
      wireDaysSel();
      return;
    }

    const winnerPnl = pickWinner('pnl', { delta_india: delta.pnl_total, bybit: bybit.pnl_total }, true);
    const winnerWr  = pickWinner('wr',  { delta_india: delta.wr_pct,    bybit: bybit.wr_pct    }, true);
    const winnerPf  = pickWinner('pf',  { delta_india: delta.pf,        bybit: bybit.pf        }, true);
    const winnerMk  = pickWinner('mk',  { delta_india: delta.maker_pct, bybit: bybit.maker_pct }, true);
    const winnerSh  = pickWinner('sh',  { delta_india: delta.sharpe,    bybit: bybit.sharpe    }, true);
    const winnerDd  = pickWinner('dd',  { delta_india: delta.max_dd_usd, bybit: bybit.max_dd_usd }, false);

    function cellClass(metric, exch) {
      if (metric === 'pnl' && winnerPnl === exch) return 'xc-cell xc-cell--val xc-cell--win';
      if (metric === 'wr'  && winnerWr  === exch) return 'xc-cell xc-cell--val xc-cell--win';
      if (metric === 'pf'  && winnerPf  === exch) return 'xc-cell xc-cell--val xc-cell--win';
      if (metric === 'mk'  && winnerMk  === exch) return 'xc-cell xc-cell--val xc-cell--win';
      if (metric === 'sh'  && winnerSh  === exch) return 'xc-cell xc-cell--val xc-cell--win';
      if (metric === 'dd'  && winnerDd  === exch) return 'xc-cell xc-cell--val xc-cell--win';
      return 'xc-cell xc-cell--val';
    }

    function valWithClass(metric, exch, formatted, raw) {
      const baseClass = cellClass(metric, exch);
      let extra = '';
      if (metric === 'pnl' || metric === 'avg') extra = ' ' + pnlClass(raw);
      if (metric === 'mk') extra = ' ' + makerClass(raw);
      if (metric === 'sh') extra = ' ' + shareClass(raw);
      return `<div class="${baseClass}${extra}">${formatted}</div>`;
    }

    const v = (m, e, fn, raw) => valWithClass(m, e, raw == null ? '—' : fn(raw), raw);
    const vDelta = (m, fn) => v(m, 'delta_india', fn, delta[m === 'pnl' ? 'pnl_total' : m === 'avg' ? 'avg' : m === 'wr' ? 'wr_pct' : m === 'mk' ? 'maker_pct' : m === 'pf' ? 'pf' : m === 'sh' ? 'sharpe' : m === 'dd' ? 'max_dd_usd' : m === 'n' ? 'n' : m === 'fees' ? 'total_fees' : m]);
    const vByt   = (m, fn) => v(m, 'bybit',       fn, bybit[m === 'pnl' ? 'pnl_total' : m === 'avg' ? 'avg' : m === 'wr' ? 'wr_pct' : m === 'mk' ? 'maker_pct' : m === 'pf' ? 'pf' : m === 'sh' ? 'sharpe' : m === 'dd' ? 'max_dd_usd' : m === 'n' ? 'n' : m === 'fees' ? 'total_fees' : m]);

    const rows = [
      { metric: 'n',    label: 'N (closed)',  fmt: x => Math.round(x).toString() },
      { metric: 'wr',   label: 'Win Rate',    fmt: x => x.toFixed(0) + '%' },
      { metric: 'pf',   label: 'Profit Factor', fmt: x => x.toFixed(2) },
      { metric: 'sh',   label: 'Sharpe (per-trade)', fmt: x => x.toFixed(2) },
      { metric: 'mk',   label: 'Maker Rate',  fmt: x => x.toFixed(0) + '%' },
      { metric: 'avg',  label: 'Avg P&L / trade', fmt: x => fmtMoney(x) },
      { metric: 'pnl',  label: 'Total P&L',   fmt: x => fmtMoney(x) },
      { metric: 'fees', label: 'Total Fees',  fmt: x => '−$' + Math.abs(x).toFixed(2) },
      { metric: 'dd',   label: 'Max Drawdown', fmt: x => '−$' + Math.abs(x).toFixed(2) },
    ];

    const rowsHtml = rows.map(r => `
      <div class="xc-cell xc-cell--metric">${escHtml(r.label)}</div>
      ${vDelta(r.metric, r.fmt)}
      ${vByt(r.metric, r.fmt)}
      <div class="xc-cell xc-cell--val" data-na="1">paper</div>
    `).join('');

    el.innerHTML = `
      <div class="xc-header">
        <span class="xc-title">Exchange Compare — last ${currentDays}d</span>
        <div class="xc-controls">${daysSelector()}</div>
      </div>
      <div class="xc-grid">
        <div class="xc-cell xc-cell--header">METRIC</div>
        <div class="xc-cell xc-cell--header">DELTA INDIA</div>
        <div class="xc-cell xc-cell--header">BYBIT</div>
        <div class="xc-cell xc-cell--header">PAPER (ref)</div>
        ${rowsHtml}
      </div>
      ${paper.n ? `
      <div class="xc-paper-row">
        Paper baseline (${currentDays}d): n=<strong>${paper.n}</strong> ·
        WR <strong>${(paper.wr_pct || 0).toFixed(0)}%</strong> ·
        PF <strong>${paper.pf != null ? paper.pf.toFixed(2) : '—'}</strong> ·
        Total <strong>${fmtMoney(paper.pnl_total)}</strong>
      </div>` : ''}
      ${(() => { const v = buildVerdict(byExch); return v ? `<div class="xc-verdict">${v}</div>` : ''; })()}
    `;
    wireDaysSel();
  }

  function daysSelector() {
    return `
      Window:
      <select id="xc-days-sel" class="xc-sel">
        <option value="1"   ${currentDays==1?'selected':''}>1d</option>
        <option value="7"   ${currentDays==7?'selected':''}>7d</option>
        <option value="30"  ${currentDays==30?'selected':''}>30d</option>
        <option value="90"  ${currentDays==90?'selected':''}>90d</option>
      </select>`;
  }

  function wireDaysSel() {
    const s = document.getElementById('xc-days-sel');
    if (s) s.addEventListener('change', e => { currentDays = parseInt(e.target.value, 10); refresh(); });
  }

  async function refresh() {
    try {
      const r = await fetch('/api/exchange-comparison?days=' + currentDays, { cache: 'no-store' });
      if (!r.ok) throw new Error('http ' + r.status);
      const d = await r.json();
      render(d);
    } catch (e) {
      console.warn('exchange-compare fetch failed:', e);
    }
  }

  function boot() {
    refresh();
    setInterval(refresh, REFRESH_MS);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
