/* Stumble taste model.
 *
 * Every item carries tags like {key:"art:style:impressionism", label:"impressionism", kind:"style"}.
 * Each thumbs up/down increments counters on the item's tags and on its category.
 * A tag's affinity is a Laplace-smoothed win rate: (up+1)/(up+down+2), so an
 * unseen tag sits at a neutral 0.5 and single votes don't swing it hard.
 * An item's score is the confidence-weighted mean of its tags' affinities;
 * selection stays stochastic (see pickItem) so exploration never dies out.
 */
window.Stumble = window.Stumble || {};

Stumble.taste = (function () {
  const KEY = 'stumble.taste.v1';
  const SEEN_CAP = 1500;

  function blank() {
    return { categories: {}, tags: {}, seen: [], ratings: 0 };
  }

  function load() {
    try {
      const raw = localStorage.getItem(KEY);
      if (!raw) return blank();
      const s = JSON.parse(raw);
      if (!s || typeof s !== 'object' || !s.tags) return blank();
      return Object.assign(blank(), s);
    } catch (e) {
      return blank();
    }
  }

  let state = load();
  let seenSet = new Set(state.seen);

  function save() {
    try {
      localStorage.setItem(KEY, JSON.stringify(state));
    } catch (e) { /* private mode etc. — the toy still works, it just forgets */ }
  }

  function affinity(stat) {
    return (stat.u + 1) / (stat.u + stat.d + 2);
  }

  function catStat(cat) {
    return state.categories[cat] || { u: 0, d: 0 };
  }

  // Confidence-weighted mean of tag affinities; unknown tags count as neutral.
  function score(item) {
    let num = 0, den = 0;
    for (const tag of item.tags) {
      const stat = state.tags[tag.key];
      if (stat) {
        const w = 1 + Math.log1p(stat.u + stat.d);
        num += affinity(stat) * w;
        den += w;
      } else {
        num += 0.5;
        den += 1;
      }
    }
    return den ? num / den : 0.5;
  }

  function rate(item, up) {
    for (const tag of item.tags) {
      const stat = state.tags[tag.key] ||
        (state.tags[tag.key] = { u: 0, d: 0, label: tag.label, kind: tag.kind });
      up ? stat.u++ : stat.d++;
    }
    const cs = state.categories[item.category] ||
      (state.categories[item.category] = { u: 0, d: 0 });
    up ? cs.u++ : cs.d++;
    state.ratings++;
    markSeen(item.id);
    save();
  }

  function markSeen(id) {
    if (seenSet.has(id)) return;
    seenSet.add(id);
    state.seen.push(id);
    if (state.seen.length > SEEN_CAP) {
      const dropped = state.seen.splice(0, state.seen.length - SEEN_CAP);
      for (const d of dropped) seenSet.delete(d);
    }
    save();
  }

  function hasSeen(id) {
    return seenSet.has(id);
  }

  function weightedPick(items, weightFn) {
    let total = 0;
    const weights = items.map((it) => {
      const w = Math.max(weightFn(it), 0);
      total += w;
      return w;
    });
    let r = Math.random() * total;
    for (let i = 0; i < items.length; i++) {
      r -= weights[i];
      if (r <= 0) return items[i];
    }
    return items[items.length - 1];
  }

  // Mostly exploit (score^4 sharpens the preference), but 20% of stumbles
  // are a pure roll of the dice — that's the "stumble" part.
  function pickItem(pool) {
    if (!pool.length) return null;
    if (Math.random() < 0.2) return pool[Math.floor(Math.random() * pool.length)];
    return weightedPick(pool, (it) => Math.pow(score(it), 4) + 0.02);
  }

  // Categories the user likes come up more often, but every enabled
  // category keeps a floor so none of them ever goes fully silent.
  function pickCategory(cats) {
    if (!cats.length) return null;
    return weightedPick(cats, (c) => 0.25 + affinity(catStat(c)));
  }

  function likedTags(cat, n) {
    return topTags(cat, n, (s) => s.u > s.d, (s) => s.u - s.d);
  }

  function dislikedTags(cat, n) {
    return topTags(cat, n, (s) => s.d > s.u, (s) => s.d - s.u);
  }

  function topTags(cat, n, filter, rank) {
    const prefix = cat + ':';
    return Object.entries(state.tags)
      .filter(([k, s]) => k.startsWith(prefix) && filter(s))
      .sort((a, b) => rank(b[1]) - rank(a[1]))
      .slice(0, n)
      .map(([k, s]) => ({ key: k, label: s.label, kind: s.kind, up: s.u, down: s.d }));
  }

  function reset() {
    state = blank();
    seenSet = new Set();
    save();
  }

  return {
    score, rate, markSeen, hasSeen, pickItem, pickCategory,
    likedTags, dislikedTags, reset,
    get ratings() { return state.ratings; },
    categoryStat: catStat,
  };
})();
