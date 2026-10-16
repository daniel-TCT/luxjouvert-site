// Lux site Worker: serves the static site and forwards /api/memories/* to the
// lux-memories service on Railway, so the browser only ever talks to luxjouvert.xyz.

// ↓ Replace with your lux-memories Railway domain (no trailing slash)
   const MEMORIES_ORIGIN = 'https://lux-memories-production-0065.up.railway.app';

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname.startsWith('/api/memories/')) {
      const target = new URL(url.pathname + url.search, MEMORIES_ORIGIN);
      return fetch(new Request(target, request));
    }

    // Pretty URL: luxjouvert.xyz/memories → memories.html
    if (url.pathname === '/memories') {
      return env.ASSETS.fetch(new Request(new URL('/memories.html', url), request));
    }

    return env.ASSETS.fetch(request);
  },
};
