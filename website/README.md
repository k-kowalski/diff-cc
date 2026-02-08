# diff_cc website

This folder is a static microsite (no build step) meant for GitHub Pages.

## Local preview

From the repo root:

```bash
python -m http.server 8000
```

Then open:

- `http://localhost:8000/website/` (landing page)
- `http://localhost:8000/website/demo.html` (TFJS realtime story demo)

Notes:
- Use a local server (not `file://`) so `fetch()` can load `assets/*.obj`.
- `demo.html` is split into two narrative sections:
  1. symmetric cube -> squished subdivided target,
  2. hierarchy recovery example using `target1_subd2.obj`.
- `website/assets/` stores only target meshes plus visual assets; cage meshes are generated in JS.

## Deploy (GitHub Pages)

Common options:
- Deploy the `website/` folder as the Pages root (via Actions).
- Or copy `website/` contents into a separate repo dedicated to the site.
