# 🚀 VoxGuard AI Deployment Guide

This repository is fully configured for deployment on **Render** (Backend API) and **Vercel** (Frontend Dashboard).

---

## 1. ⚙️ Backend Deployment on Render

Render hosts the Python FastAPI application with PyTorch, ONNX Runtime, and Praat DSP dependencies.

### Deployment Steps:
1. Push your repository to **GitHub**.
2. Go to [Render Dashboard](https://dashboard.render.com/) and click **New +** ──► **Web Service**.
3. Connect your GitHub repository (`VoxGuard-AI-Deepfake-Detector`).
4. Select **Docker** or **Python** as the Runtime:
   * **If using Docker** (Recommended for audio system libraries):
     * **Dockerfile Path**: `Dockerfile`
     * **Port**: `8000`
   * **If using Python Native**:
     * **Build Command**: `pip install -r requirements.txt`
     * **Start Command**: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
5. Click **Create Web Service**.
6. Copy your live Render service URL (e.g., `https://voxguard-api.onrender.com`).

---

## 2. 🌐 Frontend Deployment on Vercel

Vercel hosts the web interface (`web/`) with high-speed global CDN delivery and automatic proxying.

### Deployment Steps:
1. Go to [Vercel Dashboard](https://vercel.com/dashboard) and click **Add New** ──► **Project**.
2. Import your GitHub repository.
3. In Project Settings:
   * **Root Directory**: `web` (or leave default root if using `vercel.json`).
   * **Framework Preset**: Other / Static HTML.
4. Update `vercel.json` with your live Render URL:
   ```json
   {
     "version": 2,
     "cleanUrls": true,
     "rewrites": [
       {
         "source": "/api/:path*",
         "destination": "https://YOUR-RENDER-SERVICE-NAME.onrender.com/api/:path*"
       },
       {
         "source": "/health",
         "destination": "https://YOUR-RENDER-SERVICE-NAME.onrender.com/health"
       }
     ]
   }
   ```
5. Click **Deploy**.

---

## 3. 🧪 Verification & Testing

Once both services are deployed:
1. Open your Vercel URL (e.g., `https://voxguard.vercel.app`).
2. Verify that the **Engine Status** indicator shows **Engine Ready** (online green dot).
3. Drag & drop an audio file or record speech using the microphone.
4. Click **Run Forensic Analysis** to receive real-time deepfake detection reports.
