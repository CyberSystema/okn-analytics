/**
 * OKN Analytics — Upload API (Cloudflare Pages Function)
 * =====================================================
 * Secure file upload endpoint that validates and commits CSV files
 * to the GitHub repository.
 *
 * Required Cloudflare environment variables (set in Pages dashboard):
 *   UPLOAD_PASSWORD_HASH  — SHA-256 hex hash of the team password
 *   GITHUB_PAT            — Fine-grained GitHub Personal Access Token
 *   GITHUB_REPO           — e.g. "CyberSystema/okn-analytics"
 *   TOKEN_SECRET          — Random string for signing session tokens
 */

// ══════════════════════════════════════
// ALLOWED FILES — whitelist
// ══════════════════════════════════════

const ALLOWED = {
  instagram: [
    'content.csv', 'follows.csv', 'interactions.csv', 'link clicks.csv',
    'reach.csv', 'views.csv', 'visits.csv', 'audience.csv',
  ],
  tiktok: [
    'content.csv', 'overview.csv', 'viewers.csv', 'followerhistory.csv',
    'followeractivity.csv', 'followergender.csv', 'followertopterritories.csv',
  ],
};

const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB
const TOKEN_EXPIRY_MS = 60 * 60 * 1000; // 1 hour

// Rate limiting: in-memory (resets on cold start, but good enough)
const rateLimits = new Map();
const RATE_LIMIT_MAX = 20; // per hour
const RATE_LIMIT_WINDOW = 60 * 60 * 1000;

// ══════════════════════════════════════
// MAIN HANDLER
// ══════════════════════════════════════

export async function onRequestPost(context) {
  const { request, env } = context;

  // CORS headers
  const headers = {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
  };

  try {
    // Rate limit by IP
    const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
    if (isRateLimited(ip)) {
      return new Response(JSON.stringify({ ok: false, error: 'Too many requests. Try again later.' }), { status: 429, headers });
    }

    const body = await request.json();
    const action = body.action;

    if (action === 'auth') {
      return await handleAuth(body, env, headers);
    } else if (action === 'upload') {
      return await handleUpload(body, env, headers, ip);
    } else {
      return new Response(JSON.stringify({ ok: false, error: 'Invalid action' }), { status: 400, headers });
    }
  } catch (e) {
    return new Response(JSON.stringify({ ok: false, error: 'Server error' }), { status: 500, headers });
  }
}

// Handle OPTIONS for CORS preflight
export async function onRequestOptions() {
  return new Response(null, {
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}

// ══════════════════════════════════════
// AUTH
// ══════════════════════════════════════

async function handleAuth(body, env, headers) {
  const password = body.password;
  if (!password || typeof password !== 'string') {
    return new Response(JSON.stringify({ ok: false, error: 'Password required' }), { status: 400, headers });
  }

  // Hash the submitted password and compare
  const hash = await sha256(password);
  const storedHash = (env.UPLOAD_PASSWORD_HASH || '').toLowerCase().trim();

  if (!storedHash) {
    return new Response(JSON.stringify({ ok: false, error: 'Upload not configured. Set UPLOAD_PASSWORD_HASH in Cloudflare.' }), { status: 500, headers });
  }

  if (hash !== storedHash) {
    return new Response(JSON.stringify({ ok: false, error: 'Wrong password' }), { status: 401, headers });
  }

  // Generate session token (HMAC-signed, 1hr expiry)
  const token = await generateToken(env);

  return new Response(JSON.stringify({ ok: true, token }), { status: 200, headers });
}

// ══════════════════════════════════════
// UPLOAD
// ══════════════════════════════════════

async function handleUpload(body, env, headers, ip) {
  // Verify session token
  const tokenValid = await verifyToken(body.token, env);
  if (!tokenValid) {
    return new Response(JSON.stringify({ ok: false, error: 'Session expired. Please sign in again.' }), { status: 401, headers });
  }

  const { platform, filename, content } = body;

  // Validate platform
  if (!platform || !ALLOWED[platform]) {
    return new Response(JSON.stringify({ ok: false, error: 'Invalid platform' }), { status: 400, headers });
  }

  // Validate filename
  const normalizedName = (filename || '').toLowerCase().trim();
  if (!ALLOWED[platform].includes(normalizedName)) {
    return new Response(JSON.stringify({ ok: false, error: `File "${filename}" is not allowed for ${platform}` }), { status: 400, headers });
  }

  // Validate content exists and is base64
  if (!content || typeof content !== 'string') {
    return new Response(JSON.stringify({ ok: false, error: 'No file content' }), { status: 400, headers });
  }

  // Check file size (base64 is ~33% larger than raw)
  const estimatedSize = (content.length * 3) / 4;
  if (estimatedSize > MAX_FILE_SIZE) {
    return new Response(JSON.stringify({ ok: false, error: 'File too large (max 5MB)' }), { status: 400, headers });
  }

  // Validate it looks like a CSV (decode first few bytes)
  try {
    const decoded = atob(content.slice(0, 1000));
    if (!decoded.includes(',') && !decoded.includes('\t')) {
      return new Response(JSON.stringify({ ok: false, error: 'File does not appear to be a valid CSV' }), { status: 400, headers });
    }
  } catch (e) {
    return new Response(JSON.stringify({ ok: false, error: 'Invalid file encoding' }), { status: 400, headers });
  }

  // Construct the path — ONLY data/instagram/ or data/tiktok/
  // Use the original filename case for TikTok (capitalized), lowercase for Instagram
  let targetFilename = filename;
  if (platform === 'instagram') {
    // Instagram files: match the expected case
    const caseMap = {
      'content.csv': 'content.csv',
      'follows.csv': 'Follows.csv',
      'interactions.csv': 'Interactions.csv',
      'link clicks.csv': 'Link clicks.csv',
      'reach.csv': 'Reach.csv',
      'views.csv': 'Views.csv',
      'visits.csv': 'Visits.csv',
      'audience.csv': 'Audience.csv',
    };
    targetFilename = caseMap[normalizedName] || filename;
  } else if (platform === 'tiktok') {
    const caseMap = {
      'content.csv': 'Content.csv',
      'overview.csv': 'Overview.csv',
      'viewers.csv': 'Viewers.csv',
      'followerhistory.csv': 'FollowerHistory.csv',
      'followeractivity.csv': 'FollowerActivity.csv',
      'followergender.csv': 'FollowerGender.csv',
      'followertopterritories.csv': 'FollowerTopTerritories.csv',
    };
    targetFilename = caseMap[normalizedName] || filename;
  }

  const path = `data/${platform}/${targetFilename}`;

  // Commit to GitHub
  try {
    const result = await commitToGitHub(env, path, content, `📤 Upload ${targetFilename} (${platform})`);
    if (result.ok) {
      recordRateLimit(ip);
      return new Response(JSON.stringify({ ok: true, path, message: 'File uploaded successfully' }), { status: 200, headers });
    } else {
      return new Response(JSON.stringify({ ok: false, error: result.error }), { status: 500, headers });
    }
  } catch (e) {
    return new Response(JSON.stringify({ ok: false, error: 'GitHub commit failed: ' + e.message }), { status: 500, headers });
  }
}

// ══════════════════════════════════════
// GITHUB API
// ══════════════════════════════════════

async function commitToGitHub(env, path, contentBase64, message) {
  const repo = env.GITHUB_REPO || 'CyberSystema/okn-analytics';
  const token = env.GITHUB_PAT;
  const branch = env.GITHUB_BRANCH || 'main';

  if (!token) {
    return { ok: false, error: 'GitHub token not configured' };
  }

  const apiUrl = `https://api.github.com/repos/${repo}/contents/${path}`;

  // Check if file already exists (need SHA to update)
  let sha = null;
  try {
    const existing = await fetch(apiUrl + `?ref=${branch}`, {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'OKN-Analytics-Upload',
      },
    });
    if (existing.status === 200) {
      const data = await existing.json();
      sha = data.sha;
    }
  } catch (e) {
    // File doesn't exist yet, that's fine
  }

  // Create or update file
  const payload = {
    message,
    content: contentBase64,
    branch,
  };
  if (sha) payload.sha = sha;

  const res = await fetch(apiUrl, {
    method: 'PUT',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Accept': 'application/vnd.github.v3+json',
      'User-Agent': 'OKN-Analytics-Upload',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (res.status === 200 || res.status === 201) {
    return { ok: true };
  } else {
    const err = await res.json().catch(() => ({}));
    return { ok: false, error: err.message || `GitHub API error (${res.status})` };
  }
}

// ══════════════════════════════════════
// CRYPTO & TOKEN UTILS
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
  const sigHex = Array.from(new Uint8Array(sig)).map(b => b.toString(16).padStart(2, '0')).join('');
  return payload + '.' + sigHex;
}

async function verifyToken(token, env) {
  if (!token || typeof token !== 'string') return false;
  const parts = token.split('.');
  if (parts.length !== 2) return false;

  const timestamp = parseInt(parts[0]);
  if (isNaN(timestamp)) return false;

  // Check expiry
  if (Date.now() - timestamp > TOKEN_EXPIRY_MS) return false;

  // Verify signature
  const secret = env.TOKEN_SECRET || 'okn-default-secret';
  const key = await crypto.subtle.importKey('raw', new TextEncoder().encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const sig = await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(parts[0]));
  const sigHex = Array.from(new Uint8Array(sig)).map(b => b.toString(16).padStart(2, '0')).join('');

  return sigHex === parts[1];
}

// ══════════════════════════════════════
// RATE LIMITING
// ══════════════════════════════════════

function isRateLimited(ip) {
  const now = Date.now();
  const record = rateLimits.get(ip);
  if (!record) return false;
  // Clean old entries
  if (now - record.start > RATE_LIMIT_WINDOW) {
    rateLimits.delete(ip);
    return false;
  }
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
