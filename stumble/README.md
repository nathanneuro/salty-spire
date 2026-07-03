# 🎲 Stumble

A little serendipity engine in the spirit of the old **StumbleUpon**: hit the
button and it deals you a painting, a poem, a short story, or a song. Rate each
one 👍/👎 and it learns your taste, nudging future stumbles toward what you
like while still wandering enough to surprise you.

## Run it

No build, no server, no keys. Either just open the file:

```
open stumble/index.html        # macOS
xdg-open stumble/index.html    # Linux
```

…or serve the folder if you prefer:

```
cd stumble && python3 -m http.server 8000
# then visit http://localhost:8000
```

## Where the content comes from

All free, public, keyless APIs, fetched straight from your browser:

| Category | Source |
|---|---|
| 🖼 Art | [Art Institute of Chicago](https://api.artic.edu/docs/) + [The Met](https://metmuseum.github.io/) open collections |
| ✒️ Poetry | [PoetryDB](https://poetrydb.org) |
| 📖 Stories | [Project Gutenberg](https://www.gutenberg.org) via [Gutendex](https://gutendex.com) |
| 🎵 Music | [iTunes Search API](https://performance-partners.apple.com/search-api) (30-second previews) |

## How it learns

Everything is local — the taste model lives in `localStorage`, nothing leaves
your browser.

- Every item carries tags (artist, style, subject, genre, author, era, …).
- Each rating bumps up/down counters on those tags. A tag's affinity is a
  Laplace-smoothed win rate `(up+1)/(up+down+2)`, so one vote never dominates.
- An item's score is the confidence-weighted average of its tags' affinities.
- Picking the next stumble is stochastic on purpose: 80% of the time it samples
  proportional to `score⁴` from a candidate pool, 20% of the time it's a pure
  roll of the dice. Categories you like come up more often, but none ever goes
  silent.
- Search queries themselves also drift toward your taste: about half the
  fetches search the archives for your top-liked tags, the rest wander a
  seed list.

Open the **✦ taste** panel to see what it thinks you like (and to make it
forget everything).

## Controls

| Key | Action |
|---|---|
| `space` / `→` | stumble |
| `↑` | like |
| `↓` | pass |

## Files

```
stumble/
├── index.html      # shell
├── css/style.css   # the look
└── js/
    ├── taste.js    # the learning: tag stats, scoring, selection
    ├── sources.js  # the archives: fetchers + wander seeds
    └── app.js      # the app: pools, rendering, controls
```
