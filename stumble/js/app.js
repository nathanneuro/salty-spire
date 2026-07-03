/* Stumble app: keeps a pool of candidates per category topped up in the
 * background, and on each stumble picks a category (taste-weighted), then an
 * item from that pool (taste-weighted with exploration). Rating an item
 * updates the model and auto-advances to the next stumble.
 */
(function () {
  const taste = Stumble.taste;
  const CATS = Stumble.sources.categories;
  const ENABLED_KEY = 'stumble.enabled.v1';
  const POOL_LOW = 6;

  const pools = {};   // cat id -> items[]
  const fetching = {}; // cat id -> Promise|null
  let current = null;
  let busy = false;

  const $ = (id) => document.getElementById(id);
  const card = $('card');

  /* ---------- tiny DOM helper (textContent only — API data never hits innerHTML) ---------- */
  function el(tagName, cls, text) {
    const n = document.createElement(tagName);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  /* ---------- enabled categories ---------- */
  function loadEnabled() {
    try {
      const v = JSON.parse(localStorage.getItem(ENABLED_KEY));
      if (Array.isArray(v) && v.length) return new Set(v.filter((c) => CATS.some((x) => x.id === c)));
    } catch (e) { /* fall through */ }
    return new Set(CATS.map((c) => c.id));
  }
  const enabled = loadEnabled();

  function saveEnabled() {
    localStorage.setItem(ENABLED_KEY, JSON.stringify([...enabled]));
  }

  function renderToggles() {
    const nav = $('cat-toggles');
    nav.textContent = '';
    for (const c of CATS) {
      const b = el('button', 'cat-toggle' + (enabled.has(c.id) ? ' on' : ''));
      b.append(el('span', 'cat-icon', c.icon), el('span', null, c.label));
      b.setAttribute('aria-pressed', enabled.has(c.id));
      b.addEventListener('click', () => {
        if (enabled.has(c.id)) {
          if (enabled.size === 1) return toast('Keep at least one category on');
          enabled.delete(c.id);
        } else {
          enabled.add(c.id);
          topUp(c.id);
        }
        saveEnabled();
        renderToggles();
      });
      nav.append(b);
    }
  }

  /* ---------- candidate pools ---------- */
  function topUp(catId) {
    const pool = pools[catId] || (pools[catId] = []);
    if (pool.length >= POOL_LOW || fetching[catId]) return fetching[catId];
    const src = CATS.find((c) => c.id === catId);
    fetching[catId] = src.fetch()
      .then((items) => {
        const have = new Set(pool.map((i) => i.id));
        for (const it of items) {
          if (!taste.hasSeen(it.id) && !have.has(it.id)) pool.push(it);
        }
      })
      .catch((e) => console.warn('[stumble]', catId, e))
      .finally(() => { fetching[catId] = null; });
    return fetching[catId];
  }

  async function nextItem() {
    const cats = [...enabled];
    // Try a few times: pick a category, make sure its pool has something.
    for (let attempt = 0; attempt < 4; attempt++) {
      const catId = taste.pickCategory(cats);
      await topUp(catId);
      const pool = pools[catId];
      if (pool && pool.length) {
        const item = taste.pickItem(pool);
        pool.splice(pool.indexOf(item), 1);
        return item;
      }
    }
    // Last resort: anything from any enabled pool.
    for (const catId of cats) {
      if (pools[catId] && pools[catId].length) return pools[catId].shift();
    }
    return null;
  }

  /* ---------- rendering ---------- */
  function stopAudio() {
    const a = card.querySelector('audio');
    if (a) a.pause();
  }

  function renderItem(item) {
    stopAudio();
    card.textContent = '';
    card.className = 'card show-' + item.category;

    const inner = el('div', 'card-inner');

    if (item.category === 'art') {
      const img = el('img', 'art-img');
      img.src = item.image;
      img.alt = item.title;
      inner.append(img);
      const meta = el('div', 'meta');
      meta.append(el('h2', 'title', item.title));
      meta.append(el('p', 'byline', item.artist + (item.date ? ' · ' + item.date : '')));
      if (item.medium) meta.append(el('p', 'fine', item.medium));
      inner.append(meta);
    } else if (item.category === 'poetry') {
      const meta = el('div', 'meta');
      meta.append(el('h2', 'title', item.title));
      meta.append(el('p', 'byline', item.artist));
      inner.append(meta);
      const body = el('div', 'poem');
      let stanza = el('p', 'stanza');
      for (const line of item.lines) {
        if (line.trim() === '') {
          if (stanza.childNodes.length) { body.append(stanza); stanza = el('p', 'stanza'); }
        } else {
          if (stanza.childNodes.length) stanza.append(document.createElement('br'));
          stanza.append(document.createTextNode(line));
        }
      }
      if (stanza.childNodes.length) body.append(stanza);
      inner.append(body);
    } else if (item.category === 'story') {
      if (item.cover) {
        const img = el('img', 'story-cover');
        img.src = item.cover;
        img.alt = '';
        inner.append(img);
      }
      const meta = el('div', 'meta');
      meta.append(el('h2', 'title', item.title));
      meta.append(el('p', 'byline', item.artist));
      if (item.downloads) meta.append(el('p', 'fine', item.downloads.toLocaleString() + ' downloads'));
      const read = el('a', 'read-link', 'Read it →');
      read.href = item.readLink;
      read.target = '_blank';
      read.rel = 'noopener';
      meta.append(read);
      inner.append(meta);
    } else if (item.category === 'music') {
      if (item.artwork) {
        const img = el('img', 'music-art');
        img.src = item.artwork;
        img.alt = '';
        inner.append(img);
      }
      const meta = el('div', 'meta');
      meta.append(el('h2', 'title', item.title));
      meta.append(el('p', 'byline', item.artist + (item.album ? ' · ' + item.album : '')));
      const audio = document.createElement('audio');
      audio.controls = true;
      audio.src = item.preview;
      audio.className = 'preview';
      meta.append(audio);
      inner.append(meta);
      audio.play().catch(() => { /* autoplay may be blocked until first gesture */ });
    }

    // Footer: tag chips + source credit.
    const foot = el('div', 'card-foot');
    const chips = el('div', 'chips');
    for (const t of item.tags.slice(0, 6)) chips.append(el('span', 'chip', t.label));
    foot.append(chips);
    const credit = el('a', 'credit', item.credit);
    credit.href = item.link;
    credit.target = '_blank';
    credit.rel = 'noopener';
    foot.append(credit);
    inner.append(foot);

    card.append(inner);
    requestAnimationFrame(() => card.classList.add('dealt'));
  }

  function renderLoading() {
    stopAudio();
    card.textContent = '';
    card.className = 'card';
    card.append(el('div', 'spinner'));
  }

  function renderError() {
    card.textContent = '';
    card.className = 'card';
    const box = el('div', 'welcome');
    box.append(el('h1', null, 'Nothing found out there'));
    box.append(el('p', null,
      'Couldn’t reach the archives — maybe a network hiccup, or an ad-blocker ' +
      'is filtering the APIs. Give it another stumble in a moment.'));
    card.append(box);
  }

  let toastTimer;
  function toast(msg) {
    const t = $('toast');
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, 2200);
  }

  /* ---------- actions ---------- */
  async function stumble() {
    if (busy) return;
    busy = true;
    renderLoading();
    try {
      const item = await nextItem();
      if (!item) {
        renderError();
        current = null;
      } else {
        current = item;
        taste.markSeen(item.id);
        renderItem(item);
      }
    } finally {
      busy = false;
    }
    // Keep enabled pools warm in the background.
    for (const catId of enabled) topUp(catId);
  }

  function rate(up) {
    if (!current || busy) return;
    taste.rate(current, up);
    const btn = up ? $('btn-up') : $('btn-down');
    btn.classList.remove('pulse');
    void btn.offsetWidth; // restart the animation
    btn.classList.add('pulse');
    current = null;
    stumble();
  }

  /* ---------- taste panel ---------- */
  function renderTastePanel() {
    const body = $('taste-body');
    body.textContent = '';
    if (!taste.ratings) {
      body.append(el('p', 'fine', 'Nothing yet — rate a few stumbles and your taste profile will grow here.'));
      return;
    }
    body.append(el('p', 'fine', taste.ratings + ' ratings so far'));
    for (const c of CATS) {
      const liked = taste.likedTags(c.id, 8);
      const disliked = taste.dislikedTags(c.id, 4);
      if (!liked.length && !disliked.length) continue;
      const sec = el('div', 'taste-cat');
      sec.append(el('h3', null, c.icon + ' ' + c.label));
      if (liked.length) {
        const row = el('div', 'chips');
        for (const t of liked) row.append(el('span', 'chip liked', t.label + ' +' + (t.up - t.down)));
        sec.append(row);
      }
      if (disliked.length) {
        const row = el('div', 'chips');
        for (const t of disliked) row.append(el('span', 'chip disliked', t.label + ' −' + (t.down - t.up)));
        sec.append(row);
      }
      body.append(sec);
    }
  }

  /* ---------- wiring ---------- */
  $('btn-stumble').addEventListener('click', stumble);
  $('btn-up').addEventListener('click', () => rate(true));
  $('btn-down').addEventListener('click', () => rate(false));

  $('taste-btn').addEventListener('click', () => {
    renderTastePanel();
    $('taste-panel').hidden = false;
  });
  $('taste-close').addEventListener('click', () => { $('taste-panel').hidden = true; });
  $('taste-reset').addEventListener('click', () => {
    if (confirm('Forget everything Stumble has learned about your taste?')) {
      taste.reset();
      renderTastePanel();
      toast('Taste profile cleared');
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === ' ' || e.key === 'ArrowRight') { e.preventDefault(); stumble(); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); rate(true); }
    else if (e.key === 'ArrowDown') { e.preventDefault(); rate(false); }
    else if (e.key === 'Escape') { $('taste-panel').hidden = true; }
  });

  renderToggles();
  // Warm the pools while the welcome card is up, so the first stumble is instant.
  for (const catId of enabled) topUp(catId);
})();
