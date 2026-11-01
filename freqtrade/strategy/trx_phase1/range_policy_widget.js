/* Read-only view of the same authenticated responses used by the account page. */
(() => {
  let latest = null;
  let scheduled = false;
  const format = (value, digits = 4) => Number.isFinite(Number(value))
    ? Number(value).toLocaleString('zh-CN', { maximumFractionDigits: digits, minimumFractionDigits: digits })
    : '待核对';
  function draw() {
    scheduled = false;
    const anchor = document.getElementById('live-configuration-title');
    if (!anchor || !latest) return;
    const policy = latest.active_range_policy || latest.dashboard?.signal?.range_policy;
    if (!policy) return;
    let panel = document.getElementById('active-range-policy');
    if (!panel) {
      panel = document.createElement('section');
      panel.id = 'active-range-policy';
      panel.style.cssText = 'border:1px solid #94a3b8;border-radius:8px;padding:16px;margin:18px 0;line-height:1.8';
      const section = anchor.closest('section') || anchor.parentElement;
      section.parentElement.insertBefore(panel, section);
    }
    const rows = [
      latest.desired === 'running' ? '当前已启用：活跃价格区间策略' : '已配置：活跃价格区间策略（暂停中）',
      `半衰期 ${policy.half_life_days} 天 · 区间宽度上限 ${policy.width_cap_pct}% · 每日00:00更新`,
      `区间内：买入 ${format(policy.inside_buy_fraction * 100, 0)}% · 买入门槛 ${format(policy.inside_buy_threshold * 100, 3)}% · 卖出门槛 ${format(policy.inside_sell_threshold * 100, 3)}%`,
      `区间外：买入 ${format(policy.outside_buy_fraction * 100, 0)}% · 买入门槛 ${format(policy.outside_buy_threshold * 100, 2)}% · 卖出门槛 ${format(policy.outside_sell_threshold * 100, 2)}%`,
    ];
    if (policy.ready) {
      rows.push(`今日价格带 ${format(policy.band.lower, 6)} – ${format(policy.band.upper, 6)} USDT；最新确认收盘价 ${format(policy.latest_close, 6)}，位于${policy.inside ? '区间内' : '区间外'}。`);
      const buy = latest.dashboard?.signal?.sides?.buy;
      const intersectionFactor = policy.pause_upper_p75_buys
        ? ` × 交集系数 ${format(policy.intersection_buy_multiplier, 0)}` : '';
      rows.push(`最新确认收盘价对应基础比例 ${format(Number(policy.base_buy_fraction) * 100, 0)}% × P75系数 ${format(policy.sizing_multiplier, 2)}${intersectionFactor}。`);
      if (buy?.target && buy.preview_basis === 'next_eligible_buy_price_regime_p75_and_filter') {
        rows.push(`买入预览目标价 ${format(buy.target, 8)} USDT，位于${buy.preview_target_inside ? '区间内' : '区间外'}；目标价对应比例 ${format(Number(buy.target_fraction) * 100, 2)}%，预算 ${format(buy.target_budget_estimate)} USDT。`);
      }
      if (policy.pause_upper_p75_buys) {
        rows.push(policy.new_buy_paused
          ? '交集过滤已启用：最新收盘价同时高于价格带上界和P75，暂停新增买入；已有持仓继续按原规则卖出。'
          : '交集过滤已启用：最新收盘价未同时高于价格带上界和P75，新增买入继续检查原信号条件。');
      }
    } else rows.push('等待当日区间与已确认行情更新。');
    rows.push('买入预览按当前报价向下寻找可触发价格，以目标价所在区间、P75和交集过滤统一估算；确认时重新核验，原资金、成本与收益起点延续。');
    const signature = JSON.stringify(rows);
    if (panel.dataset.signature !== signature) {
      panel.replaceChildren(...rows.map((text, index) => {
        const line = document.createElement(index === 0 ? 'h4' : 'p');
        line.textContent = text;
        if (index === 0) line.style.cssText = 'font-weight:600;margin-bottom:8px';
        if (index === rows.length - 1) line.style.cssText = 'font-size:12px;color:#64748b;margin-top:8px';
        return line;
      }));
      panel.dataset.signature = signature;
    }
    const runningValues = {
      '首次 TRX / USDGO 预算': '历史首仓已完成；本次切换沿用现有持仓，不重复建仓。',
      '买入 / 卖出门槛': `区间内 ${format(policy.inside_buy_threshold * 100, 3)}% / ${format(policy.inside_sell_threshold * 100, 3)}%；区间外 ${format(policy.outside_buy_threshold * 100, 2)}% / ${format(policy.outside_sell_threshold * 100, 2)}%。`,
      '基础买入 / 卖出比例': `买入：区间内 ${format(policy.inside_buy_fraction * 100, 0)}% / 区间外 ${format(policy.outside_buy_fraction * 100, 0)}%；基础卖出 ${format(latest.effective_config?.sell_fraction * 100, 0)}%；保留盈利批次与P75系数。`,
      '买卖数量计算基准': '买入按信号产生时的区间位置选择基础比例，再应用P75系数、实际可用资金和费用核验；卖出保留盈利批次限制，留存TRX利息不作为普通卖出本金。',
    };
    for (const row of document.querySelectorAll('tr')) {
      const cells = row.children;
      if (cells.length !== 3) continue;
      const value = runningValues[cells[0].textContent.trim()];
      if (value && cells[2].textContent !== value) cells[2].textContent = value;
    }
    for (const line of document.querySelectorAll('p')) {
      const node = line.firstChild;
      if (node?.nodeType === Node.TEXT_NODE && node.textContent.includes('数量按恰好触及门槛价预估，尚未下单；确认和资金到账后重新核验。')) {
        node.textContent = node.textContent.replace('数量按恰好触及门槛价预估，尚未下单；确认和资金到账后重新核验。', '买入预览以目标价所在区间、P75和交集过滤统一估算；尚未下单，确认和资金到账后重新核验。');
      }
    }
    for (const line of document.querySelectorAll('.signal-explanation p')) {
      const node = line.firstChild;
      if (node?.nodeType === Node.TEXT_NODE && node.textContent.includes('按门槛价预估比例：') && line.parentElement.textContent.includes('预估买入预算：')) {
        node.textContent = node.textContent.replace('按门槛价预估比例：', '按预览目标价估算比例：');
      }
    }
  }
  function schedule() {
    if (!scheduled) { scheduled = true; requestAnimationFrame(draw); }
  }
  function relevant(url) {
    try {
      const parsed = new URL(url, location.href);
      return parsed.origin === location.origin && ['/api/v1/trx/live', '/api/v1/trx/live/view'].includes(parsed.pathname);
    } catch { return false; }
  }
  function receive(body) {
    const runtime = body?.runtime || body?.data?.runtime;
    if (runtime) {
      if (runtime.active_range_policy && !runtime.dashboard?.signal?.sides?.buy?.preview_basis) return;
      latest = runtime; schedule();
    }
  }
  const open = XMLHttpRequest.prototype.open;
  const send = XMLHttpRequest.prototype.send;
  const urls = new WeakMap();
  XMLHttpRequest.prototype.open = function (method, url, ...args) {
    urls.set(this, String(url));
    return open.call(this, method, url, ...args);
  };
  XMLHttpRequest.prototype.send = function (...args) {
    if (relevant(urls.get(this))) this.addEventListener('load', () => {
      try { receive(this.responseType === 'json' ? this.response : JSON.parse(this.responseText)); } catch {}
    }, { once: true });
    return send.apply(this, args);
  };
  const fetch = window.fetch;
  window.fetch = function (...args) {
    return fetch.apply(this, args).then(response => {
      if (relevant(response.url)) response.clone().json().then(receive).catch(() => {});
      return response;
    });
  };
  new MutationObserver(schedule).observe(document.documentElement, { childList: true, subtree: true });
  schedule();
})();
