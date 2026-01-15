# BuyBuddy AI Platform - Deployment Guide

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (Next.js)  →  Vercel                          │
│  Backend (FastAPI)   →  Railway                         │
│  Database            →  Supabase (already configured)   │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

- GitHub account (repo should be pushed to GitHub)
- Railway account: https://railway.app (sign up with GitHub)
- Vercel account: https://vercel.com (sign up with GitHub)

---

## Step 1: Deploy Backend to Railway

### 1.1 Create Railway Project

1. Go to https://railway.app and sign in with GitHub
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose `buybuddy-ai` repository
5. **Important:** Set the root directory to `apps/api`

### 1.2 Configure Environment Variables

In Railway dashboard, go to your service → Variables, and add:

```env
# Supabase
SUPABASE_URL=https://qvyxpfcwfktxnaeavkxx.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Buybuddy Legacy API
BUYBUDDY_API_URL=https://api-legacy.buybuddy.co/api/v1
BUYBUDDY_USERNAME=your-username
BUYBUDDY_PASSWORD=your-password

# Qdrant (if using)
QDRANT_URL=your-qdrant-url
QDRANT_API_KEY=your-qdrant-api-key

# Optional: External APIs
GEMINI_API_KEY=your-gemini-api-key
HF_TOKEN=your-huggingface-token
RUNPOD_API_KEY=your-runpod-api-key
```

### 1.3 Deploy

Railway will automatically:
- Detect the Dockerfile
- Build the container
- Deploy the service
- Provide a URL like: `buybuddy-api-production.up.railway.app`

### 1.4 Get Your API URL

After deployment, copy the Railway URL. You'll need it for the frontend.

---

## Step 2: Deploy Frontend to Vercel

### 2.1 Create Vercel Project

1. Go to https://vercel.com and sign in with GitHub
2. Click "Add New..." → "Project"
3. Import `buybuddy-ai` repository
4. **Important:** Set the root directory to `apps/web`
5. Framework Preset: Next.js (auto-detected)

### 2.2 Configure Environment Variables

In Vercel dashboard, go to your project → Settings → Environment Variables:

```env
# API URL (Railway backend)
NEXT_PUBLIC_API_URL=https://your-railway-app.up.railway.app

# Supabase (public keys only)
NEXT_PUBLIC_SUPABASE_URL=https://qvyxpfcwfktxnaeavkxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
```

### 2.3 Deploy

Click "Deploy" and Vercel will:
- Install dependencies with pnpm
- Build the Next.js app
- Deploy to edge network
- Provide a URL like: `buybuddy-ai.vercel.app`

---

## Updating the Application

### Backend (Railway)
```bash
git add .
git commit -m "Update API"
git push origin main
```
Railway automatically rebuilds and deploys.

### Frontend (Vercel)
```bash
git add .
git commit -m "Update frontend"
git push origin main
```
Vercel automatically rebuilds and deploys.

---

## Custom Domains (Optional)

### Railway
1. Go to your service → Settings → Domains
2. Add custom domain (e.g., `api.yourdomain.com`)
3. Update DNS records as instructed

### Vercel
1. Go to your project → Settings → Domains
2. Add custom domain (e.g., `app.yourdomain.com`)
3. Update DNS records as instructed

---

## Monitoring

### Railway
- Logs: Service → Deployments → View Logs
- Metrics: Service → Metrics tab
- Health: `/health` endpoint

### Vercel
- Logs: Project → Logs tab
- Analytics: Project → Analytics tab
- Functions: Project → Functions tab

---

## Troubleshooting

### Backend not starting
1. Check Railway logs for errors
2. Verify all environment variables are set
3. Test locally with Docker:
   ```bash
   cd apps/api
   docker build -t buybuddy-api .
   docker run -p 8000:8000 --env-file .env buybuddy-api
   ```

### Frontend build failing
1. Check Vercel logs
2. Ensure `NEXT_PUBLIC_API_URL` is set correctly
3. Test locally:
   ```bash
   cd apps/web
   pnpm build
   ```

### CORS errors
Update `cors_origins` in backend config to include your Vercel domain.

---

## Cost Estimate

| Service | Free Tier | Paid |
|---------|-----------|------|
| Railway | $5/month credit | ~$5-20/month |
| Vercel | Hobby (free) | Pro $20/month |
| Supabase | Free tier | Pro $25/month |

For a small project, you can run entirely on free tiers!
