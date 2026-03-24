/**
 * OKN Analytics — Upload API (Cloudflare Pages Function)
 * =====================================================
 * Secure file upload endpoint. Validates and commits CSV files
 * to the GitHub repository. Files arrive pre-renamed by the frontend.
 *
 * Required Cloudflare environment variables:
 *   UPLOAD_PASSWORD_HASH  — SHA-256 hex hash of the team password
 *   GITHUB_PAT            — Fine-grained GitHub Personal Access Token
 *   GITHUB_REPO           — e.g. "CyberSystema/okn-analytics"
 *   TOKEN_SECRET          — Random string for signing session tokens
 */

// ══════════════════════════════════════
// ALLOWED TARGET FILES (case-insensitive check)
// ══════════════════════════════════════

const ALLOWED = {
  instagram: [
    'content.csv', 'Follows.csv', 'Interactions.csv', 'Link clicks.csv',
    'Reach.csv', 'Views.csv', 'Visits.csv', 'Audience.csv',
  ],
  tiktok: [
    'Content.csv', 'Overview.csv', 'Viewers.csv', 'FollowerHistory.csv',
    'FollowerActivity.csv', 'FollowerGender.csv', 'FollowerTopTerritories.csv',
  ],
};

// Build case-insensitive lookup: lowercase → correct case
const FILENAME_MAP = {};
for (const [platform, files] of Object.entries(ALLOWED)) {
  FILENAME_MAP[platform] = {};
  for (const f of files) {
    FILENAME_MAP[platform][f.toLowerCase()] = f;
  }
}

const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
const TOKEN_EXPIRY_MS = 60 * 60 * 1000; // 1 hour
const RATE_LIMIT_MAX = 30; // per hour per IP
const RATE_LIMIT_WINDOW = 60 * 60 * 1000;
const rateLimits = new Map();

// ══════════════════════════════════════
// MAIN HANDLER
// ══════════════════════════════════════

export async function onRequestPost(context) {
  const { request, env } = context;
  const headers = corsHeaders();

  try {
    const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
    if (isRateLimited(ip)) {
      return respond({ ok: false, error: 'Too many requests. Try again later.' }, 429, headers);
    }

    const body = await request.json();

    if (body.action === 'auth') {
      return await handleAuth(body, env, headers);
    } else if (body.action === 'upload') {
      return await handleUpload(body, env, headers, ip);
    } else {
      return respond({ ok: false, error: 'Invalid action' }, 400, headers);
    }
  } catch (e) {
    return respond({ ok: false, error: 'Server error' }, 500, headers);
  }
}

export async function onRequestOptions() {
  return new Response(null, { headers: corsHeaders() });
}

function corsHeaders() {
  return {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
  };
}

function respond(data, status, headers) {
  return new Response(JSON.stringify(data), { status, headers });
}

// ══════════════════════════════════════
// AUTH
// ══════════════════════════════════════

async function handleAuth(body, env, headers) {
  const password = body.password;
  if (!password || typeof password !== 'string') {
    return respond({ ok: false, error: 'Password required' }, 400, headers);
  }

  const storedHash = (env.UPLOAD_PASSWORD_HASH || '').toLowerCase().trim();
  if (!storedHash) {
    return respond({ ok: false, error: 'Upload not configured. Set UPLOAD_PASSWORD_HASH.' }, 500, headers);
  }

  const hash = await sha256(password);
  if (hash !== storedHash) {
    return respond({ ok: false, error: 'Wrong password' }, 401, headers);
  }

  const token = await generateToken(env);
  return respond({ ok: true, token }, 200, headers);
}

// ══════════════════════════════════════
// UPLOAD
// ══════════════════════════════════════

async function handleUpload(body, env, headers, ip) {
  // Verify session
  if (!await verifyToken(body.token, env)) {
    return respond({ ok: false, error: 'Session expired. Please sign in again.' }, 401, headers);
  }

  const { platform, filename, content } = body;

  // Validate platform
  if (!platform || !ALLOWED[platform]) {
    return respond({ ok: false, error: 'Invalid platform' }, 400, headers);
  }

  // Validate filename (case-insensitive)
  if (!filename || typeof filename !== 'string') {
    return respond({ ok: false, error: 'No filename' }, 400, headers);
  }
  const normalizedName = filename.toLowerCase().trim();
  const correctName = FILENAME_MAP[platform][normalizedName];
  if (!correctName) {
    return respond({ ok: false, error: `File "${filename}" is not allowed for ${platform}` }, 400, headers);
  }

  // Validate content
  if (!content || typeof content !== 'string') {
    return respond({ ok: false, error: 'No file content' }, 400, headers);
  }

  // Check size (base64 is ~33% larger)
  if ((content.length * 3 / 4) > MAX_FILE_SIZE) {
    return respond({ ok: false, error: 'File too large (max 5MB)' }, 400, headers);
  }

  // Validate it looks like CSV
  try {
    const decoded = atob(content.slice(0, 1000));
    if (!decoded.includes(',') && !decoded.includes('\t') && !decoded.includes('\n')) {
      return respond({ ok: false, error: 'File does not appear to be a valid CSV' }, 400, headers);
    }
  } catch (e) {
    return respond({ ok: false, error: 'Invalid file encoding' }, 400, headers);
  }

  // Build safe path — ONLY data/{platform}/
  const path = `data/${platform}/${correctName}`;

  // Commit to GitHub
  try {
    const result = await commitToGitHub(env, path, content, `📤 Upload ${correctName} (${platform})`);
    if (result.ok) {
      recordRateLimit(ip);
      return respond({ ok: true, path }, 200, headers);
    } else {
      return respond({ ok: false, error: result.error }, 500, headers);
    }
  } catch (e) {
    return respond({ ok: false, error: 'GitHub commit failed: ' + e.message }, 500, headers);
  }
}

// ══════════════════════════════════════
// GITHUB API
// ══════════════════════════════════════

async function commitToGitHub(env, path, contentBase64, message) {
  const repo = env.GITHUB_REPO || 'CyberSystema/okn-analytics';
  const token = env.GITHUB_PAT;
  const branch = env.GITHUB_BRANCH || 'main';

  if (!token) return { ok: false, error: 'GitHub token not configured' };

  const apiUrl = `https://api.github.com/repos/${repo}/contents/${path}`;
  const authHeaders = {
    'Authorization': `Bearer ${token}`,
    'Accept': 'application/vnd.github.v3+json',
    'User-Agent': 'OKN-Analytics-Upload',
  };

  // Get existing file SHA (needed for updates)
  let sha = null;
  try {
    const existing = await fetch(apiUrl + `?ref=${branch}`, { headers: authHeaders });
    if (existing.status === 200) {
      const data = await existing.json();
      sha = data.sha;
    }
  } catch (e) { /* file doesn't exist yet */ }

  // Create or update
  const payload = { message, content: contentBase64, branch };
  if (sha) payload.sha = sha;

  const res = await fetch(apiUrl, {
    method: 'PUT',
    headers: { ...authHeaders, 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (res.status === 200 || res.status === 201) {
    return { ok: true };
  }
  const err = await res.json().catch(() => ({}));
  return { ok: false, error: err.message || `GitHub API error (${res.status})` };
}

// ══════════════════════════════════════
// CRYPTO & TOKENS
// ══════════════════════════════════════

async function sha256(text) {
  const data = new TextEncoder().encode(text);
  const hash = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(hash)).map(b => b.toString(16).padStart(2, '0')).join('');
}

async function generateToken(env) {
  const secret = env.TOKEN_SECRET || 'okn-default-secret';
  const payload = Date.now().toString();
  const key = await crypto.subtle.importKey('raw', new TextEncoder().encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const sig = await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(payload));
  return payload + '.' + Array.from(new Uint8Array(sig)).map(b => b.toString(16).padStart(2, '0')).join('');
}

async function verifyToken(token, env) {
  if (!token || typeof token !== 'string') return false;
  const parts = token.split('.');
  if (parts.length !== 2) return false;
  const timestamp = parseInt(parts[0]);
  if (isNaN(timestamp) || Date.now() - timestamp > TOKEN_EXPIRY_MS) return false;

  const secret = env.TOKEN_SECRET || 'okn-default-secret';
  const key = await crypto.subtle.importKey('raw', new TextEncoder().encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const sig = await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(parts[0]));
  const expected = Array.from(new Uint8Array(sig)).map(b => b.toString(16).padStart(2, '0')).join('');
  return expected === parts[1];
}

// ══════════════════════════════════════
// RATE LIMITING
// ══════════════════════════════════════

function isRateLimited(ip) {
  const now = Date.now();
  const record = rateLimits.get(ip);
  if (!record) return false;
  if (now - record.start > RATE_LIMIT_WINDOW) { rateLimits.delete(ip); return false; }
  return record.count >= RATE_LIMIT_MAX;
}

function recordRateLimit(ip) {
  const now = Date.now();
  const record = rateLimits.get(ip);
  if (!record || now - record.start > RATE_LIMIT_WINDOW) {
    rateLimits.set(ip, { start: now, count: 1 });
  } else {
    record.count++;
  }
}
