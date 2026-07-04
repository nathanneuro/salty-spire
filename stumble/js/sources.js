/* Stumble content sources. All free, keyless, CORS-friendly public APIs:
 *
 *   art    — Art Institute of Chicago + Metropolitan Museum of Art
 *   poetry — PoetryDB
 *   story  — Project Gutenberg (via Gutendex)
 *   music  — iTunes Search API (30-second previews)
 *
 * Each source's fetch() returns a batch of candidate items shaped as:
 *   { id, category, title, tags:[{key,label,kind}], ...category-specific fields }
 * Queries are seeded either from a wander list (explore) or from the user's
 * top liked tags (exploit), so what we *fetch* drifts toward taste too —
 * not just what we pick from the pool.
 */
window.Stumble = window.Stumble || {};

Stumble.sources = (function () {
  const taste = Stumble.taste;

  const SEEDS = {
    art: ['landscape', 'portrait', 'impressionism', 'woodblock', 'still life', 'abstract',
      'mythology', 'gold', 'dream', 'sea', 'garden', 'winter', 'dance', 'moon', 'cats',
      'battle', 'saints', 'textile', 'sculpture', 'surrealism', 'city', 'flowers', 'birds'],
    story: ['short stories', 'ghost stories', 'science fiction', 'detective fiction',
      'fairy tales', 'adventure stories', 'gothic fiction', 'humor', 'fantasy fiction',
      'horror tales', 'love stories', 'sea stories', 'western stories'],
    music: ['jazz', 'ambient', 'classical piano', 'indie folk', 'blues', 'bossa nova',
      'post-rock', 'soul', 'synthwave', 'baroque', 'afrobeat', 'trip hop', 'bluegrass',
      'shoegaze', 'string quartet', 'funk', 'dream pop', 'minimalism', 'flamenco',
      'motown', 'gospel', 'delta blues', 'krautrock', 'ragtime'],
  };

  function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

  function shuffle(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  async function fetchJSON(url) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 12000);
    try {
      const res = await fetch(url, { signal: ctrl.signal });
      if (!res.ok) throw new Error('HTTP ' + res.status + ' from ' + new URL(url).host);
      return await res.json();
    } finally {
      clearTimeout(timer);
    }
  }

  // 60% of the time, if we know enough about the user's taste in this
  // category, search for something they like; otherwise wander.
  function pickQuery(cat, kinds) {
    const liked = taste.likedTags(cat, 8)
      .filter((t) => !kinds || kinds.includes(t.kind));
    if (liked.length >= 2 && Math.random() < 0.6) return pick(liked).label;
    return pick(SEEDS[cat]);
  }

  function tag(cat, kind, label) {
    label = String(label).trim();
    return { key: cat + ':' + kind + ':' + label.toLowerCase(), label, kind };
  }

  function dedupeTags(tags) {
    const seen = new Set();
    return tags.filter((t) => {
      if (!t.label || t.label.length > 48 || seen.has(t.key)) return false;
      seen.add(t.key);
      return true;
    });
  }

  // Cheap stable id for items whose API has no numeric id (poems).
  function hashId(str) {
    let h = 5381;
    for (let i = 0; i < str.length; i++) h = ((h << 5) + h + str.charCodeAt(i)) >>> 0;
    return h.toString(36);
  }

  /* ---------------- art ---------------- */

  async function fetchArtAIC() {
    const q = pickQuery('art', ['style', 'subject', 'artist', 'place', 'classification', 'medium']);
    const fields = 'id,title,artist_title,date_display,medium_display,place_of_origin,' +
      'term_titles,style_titles,classification_titles,image_id';
    const base = 'https://api.artic.edu/api/v1/artworks/search?q=' + encodeURIComponent(q) +
      '&query[term][is_public_domain]=true&fields=' + fields + '&limit=30';
    let data = await fetchJSON(base + '&page=' + (1 + Math.floor(Math.random() * 6)));
    if (!data.data || !data.data.length) data = await fetchJSON(base + '&page=1');
    const iiif = (data.config && data.config.iiif_url) || 'https://www.artic.edu/iiif/2';
    return (data.data || [])
      .filter((a) => a.image_id)
      .map((a) => ({
        id: 'aic-' + a.id,
        category: 'art',
        title: a.title || 'Untitled',
        artist: a.artist_title || 'Unknown artist',
        date: a.date_display || '',
        medium: a.medium_display || '',
        image: iiif + '/' + a.image_id + '/full/843,/0/default.jpg',
        link: 'https://www.artic.edu/artworks/' + a.id,
        credit: 'Art Institute of Chicago',
        tags: dedupeTags([
          a.artist_title && tag('art', 'artist', a.artist_title),
          a.place_of_origin && tag('art', 'place', a.place_of_origin),
          ...(a.style_titles || []).map((s) => tag('art', 'style', s)),
          ...(a.classification_titles || []).slice(0, 3).map((s) => tag('art', 'classification', s)),
          ...(a.term_titles || []).slice(0, 6).map((s) => tag('art', 'subject', s)),
        ].filter(Boolean)),
      }));
  }

  async function fetchArtMet() {
    const q = pickQuery('art', ['style', 'subject', 'artist', 'place', 'classification', 'medium']);
    const search = await fetchJSON(
      'https://collectionapi.metmuseum.org/public/collection/v1/search?hasImages=true&q=' +
      encodeURIComponent(q));
    const ids = shuffle((search.objectIDs || []).slice(0, 150)).slice(0, 8);
    const settled = await Promise.allSettled(ids.map((id) => fetchJSON(
      'https://collectionapi.metmuseum.org/public/collection/v1/objects/' + id)));
    return settled
      .filter((r) => r.status === 'fulfilled' && r.value && r.value.primaryImageSmall)
      .map((r) => r.value)
      .map((o) => ({
        id: 'met-' + o.objectID,
        category: 'art',
        title: o.title || 'Untitled',
        artist: o.artistDisplayName || o.culture || 'Unknown artist',
        date: o.objectDate || '',
        medium: o.medium || '',
        image: o.primaryImageSmall,
        link: o.objectURL || 'https://www.metmuseum.org',
        credit: 'The Metropolitan Museum of Art',
        tags: dedupeTags([
          o.artistDisplayName && tag('art', 'artist', o.artistDisplayName),
          o.culture && tag('art', 'place', o.culture),
          o.period && tag('art', 'style', o.period),
          o.department && tag('art', 'classification', o.department),
          o.classification && tag('art', 'classification', o.classification),
          o.medium && tag('art', 'medium', o.medium.split(',')[0]),
          ...((o.tags || []).slice(0, 6).map((t) => t.term && tag('art', 'subject', t.term))),
        ].filter(Boolean)),
      }));
  }

  async function fetchArt() {
    const order = Math.random() < 0.5 ? [fetchArtAIC, fetchArtMet] : [fetchArtMet, fetchArtAIC];
    try {
      const items = await order[0]();
      if (items.length) return items;
    } catch (e) { /* fall through to the other museum */ }
    return order[1]();
  }

  /* ---------------- poetry ---------------- */

  function lengthBucket(n) {
    if (n <= 16) return 'short poem';
    if (n <= 40) return 'medium-length poem';
    return 'long poem';
  }

  function toPoem(p) {
    const lines = p.lines || [];
    return {
      id: 'poem-' + hashId(p.author + '|' + p.title),
      category: 'poetry',
      title: p.title,
      artist: p.author,
      lines,
      credit: 'PoetryDB',
      link: 'https://poetrydb.org',
      tags: dedupeTags([
        tag('poetry', 'author', p.author),
        tag('poetry', 'length', lengthBucket(lines.length)),
      ]),
    };
  }

  async function fetchPoetry() {
    const likedAuthors = taste.likedTags('poetry', 6).filter((t) => t.kind === 'author');
    if (likedAuthors.length && Math.random() < 0.5) {
      try {
        const res = await fetchJSON('https://poetrydb.org/author/' +
          encodeURIComponent(pick(likedAuthors).label));
        if (Array.isArray(res) && res.length) return shuffle(res).slice(0, 10).map(toPoem);
      } catch (e) { /* fall back to random */ }
    }
    const res = await fetchJSON('https://poetrydb.org/random/12');
    return (Array.isArray(res) ? res : []).map(toPoem);
  }

  /* ---------------- short stories ---------------- */

  function subjectTags(subjects) {
    const out = [];
    for (const s of subjects || []) {
      for (const piece of s.split(' -- ')) {
        const clean = piece.replace(/\.$/, '').trim();
        if (clean && clean.length <= 40) out.push(tag('story', 'subject', clean));
      }
    }
    return out.slice(0, 7);
  }

  async function fetchStories() {
    const topic = pickQuery('story', ['subject']);
    const base = 'https://gutendex.com/books?languages=en&topic=' + encodeURIComponent(topic);
    let data;
    try {
      data = await fetchJSON(base + '&page=' + (1 + Math.floor(Math.random() * 4)));
    } catch (e) {
      data = await fetchJSON(base); // out-of-range page → 404; first page always exists
    }
    if (!data.results || !data.results.length) data = await fetchJSON(base);
    return (data.results || []).map((b) => {
      const author = (b.authors && b.authors[0] && b.authors[0].name) || 'Anonymous';
      return {
        id: 'gut-' + b.id,
        category: 'story',
        title: b.title,
        artist: author,
        cover: b.formats && b.formats['image/jpeg'],
        link: 'https://www.gutenberg.org/ebooks/' + b.id,
        readLink: (b.formats && (b.formats['text/html'] ||
          b.formats['text/html; charset=utf-8'])) ||
          'https://www.gutenberg.org/ebooks/' + b.id,
        downloads: b.download_count,
        credit: 'Project Gutenberg',
        tags: dedupeTags([
          tag('story', 'author', author),
          ...subjectTags(b.subjects),
        ]),
      };
    });
  }

  /* ---------------- music ---------------- */

  function decade(dateStr) {
    const y = new Date(dateStr).getFullYear();
    return isFinite(y) ? Math.floor(y / 10) * 10 + 's' : null;
  }

  async function fetchMusic() {
    const q = pickQuery('music', ['genre', 'artist']);
    const data = await fetchJSON('https://itunes.apple.com/search?term=' +
      encodeURIComponent(q) + '&media=music&entity=song&limit=30');
    return (data.results || [])
      .filter((r) => r.previewUrl)
      .map((r) => ({
        id: 'itunes-' + r.trackId,
        category: 'music',
        title: r.trackName,
        artist: r.artistName,
        album: r.collectionName || '',
        genre: r.primaryGenreName || '',
        artwork: (r.artworkUrl100 || '').replace('100x100', '400x400'),
        preview: r.previewUrl,
        link: r.trackViewUrl || 'https://music.apple.com',
        credit: 'iTunes preview',
        tags: dedupeTags([
          r.primaryGenreName && tag('music', 'genre', r.primaryGenreName),
          r.artistName && tag('music', 'artist', r.artistName),
          r.releaseDate && decade(r.releaseDate) && tag('music', 'decade', decade(r.releaseDate)),
        ].filter(Boolean)),
      }));
  }

  return {
    categories: [
      { id: 'art', label: 'Art', icon: '🖼', fetch: fetchArt },
      { id: 'poetry', label: 'Poetry', icon: '✒️', fetch: fetchPoetry },
      { id: 'story', label: 'Stories', icon: '📖', fetch: fetchStories },
      { id: 'music', label: 'Music', icon: '🎵', fetch: fetchMusic },
    ],
  };
})();
