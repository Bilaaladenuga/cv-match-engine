# Deployment Guide

## Free Tier Deployment (Render + Vercel)

### Prerequisites
- GitHub repository (already have: https://github.com/Bilaaladenuga/cv-match-engine.git)
- Render account (free): https://render.com
- Vercel account (free): https://vercel.com

---

## Step 1: Deploy Backend to Render

1. **Log in to Render** → https://dashboard.render.com
2. **Click "New +"** → Select "Blueprint"
3. **Connect GitHub repo** → Select `cv-match-engine`
4. **Render will detect `backend/render.yaml`** and create:
   - Web Service: `career-match-api`
   - PostgreSQL Database: `career-match-db`
5. **Click "Apply"** → Wait for deployment (~5-10 minutes)
6. **Note the URL**: `https://career-match-api.onrender.com`

### Verify Backend
```bash
curl https://career-match-api.onrender.com/health
```

---

## Step 2: Deploy Frontend to Vercel

1. **Log in to Vercel** → https://vercel.com/dashboard
2. **Click "New Project"**
3. **Import GitHub repo** → Select `cv-match-engine`
4. **Configure project**:
   - Framework: Next.js
   - Root Directory: `frontend`
   - Build Command: `npm run build`
   - Output Directory: `.next`
5. **Add Environment Variables**:
   - `NEXT_PUBLIC_API_URL` = `https://career-match-api.onrender.com/api`
6. **Click "Deploy"** → Wait for deployment (~2-3 minutes)
7. **Note the URL**: `https://cv-match-engine.vercel.app`

---

## Step 3: Update CORS (if needed)

After deployment, point the backend at the frontend's real origin:

1. Go to Render Dashboard → `career-match-api` → Environment
2. Update `CORS_ORIGINS` with the deployed frontend URL(s)
3. Save — the service auto-redeploys

### CORS_ORIGINS format

The value accepts any of these shapes:

```
["https://cv-match-engine.vercel.app"]      # JSON array (Render/Railway)
https://a.example.com,https://b.example.com  # comma-separated
https://cv-match-engine.vercel.app           # single origin
```

Wildcard subdomains are supported and are compiled into a regex, so
Vercel preview deployments work without listing each one:

```
["https://cv-match-engine.vercel.app", "https://*.vercel.app"]
```

Notes:

- A bare `*` is accepted, but it forces `allow_credentials` **off** — the
  CORS spec forbids a wildcard origin on credentialed requests. Prefer an
  explicit allowlist.
- Trailing slashes are not part of an origin. Use `https://app.vercel.app`,
  not `https://app.vercel.app/`.
- Changes only take effect after the backend restarts (the value is read at
  startup).

### If the browser still reports a CORS error

A missing `Access-Control-Allow-Origin` header is the symptom, but the
cause is often upstream:

1. **Render free tier cold start** — the first request after idle can return
   a 502/504 from the proxy with no CORS headers. Retry once; it usually
   succeeds on the warm instance.
2. **Wrong `NEXT_PUBLIC_API_URL`** — it must include the `/api` suffix
   (`https://career-match-api.onrender.com/api`). A 404 is not a CORS bug,
   but browsers can surface it as one.
3. **Origin mismatch** — open DevTools → Network, inspect the failing
   request's `Origin:` header, and make sure that exact string is in
   `CORS_ORIGINS`.

---

## Step 4: Run Database Migrations

```bash
# SSH into Render shell (or use Render Dashboard > Shell)
cd backend
alembic upgrade head
```

---

## Cost Summary

| Service | Plan | Cost |
|---|---|---|
| Render Backend | Free | $0/month |
| Render PostgreSQL | Free | $0/month |
| Vercel Frontend | Free | $0/month |
| **Total** | | **$0/month** |

### Free Tier Limits
- **Render**: 750 hours/month, spins down after 15 min inactivity
- **Vercel**: 100GB bandwidth, unlimited builds
- **Render PostgreSQL**: 90 days, 1GB storage

---

## Production Upgrades (When Ready)

| Upgrade | Cost | What You Get |
|---|---|---|
| Render Starter ($7/mo) | $7/month | No spin-down, faster builds |
| Render PostgreSQL ($7/mo) | $7/month | Persistent, more storage |
| Vercel Pro ($20/mo) | $20/month | More bandwidth, analytics |
| Railway ($5/mo) | $5/month | Always-on, no cold starts |

---

## Alternative: Railway (Always-On)

If you want no cold starts:

1. **Railway**: https://railway.app
2. **Create new project** → Deploy from GitHub
3. **Add PostgreSQL** plugin
4. **Set environment variables**
5. **Deploy**: ~$5/month total

---

## Troubleshooting

### Backend won't start
- Check Render logs: Dashboard → career-match-api → Logs
- Common issue: ML model loading takes time (first request ~30s)
- Solution: Add `STARTUP_DELAY=60` env var

### Frontend can't connect to API
- Check `NEXT_PUBLIC_API_URL` in Vercel env vars
- Verify backend is running: `curl https://career-match-api.onrender.com/health`
- Check CORS settings in backend

### Database connection errors
- Verify `DATABASE_URL` is set in Render env vars
- Check PostgreSQL is running: Dashboard → career-match-db → Info
