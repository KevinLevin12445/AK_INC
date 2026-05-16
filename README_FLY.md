# Deploy AK-INC Terminal en Fly.io

Este repo contiene una app Streamlit en `terminal/app.py`.

## Qué necesitas
- Cuenta en Fly.io
- (Recomendado) Docker en tu PC para que `fly launch` construya imagen.

## Pasos
1. Instala Fly CLI (en tu PC): https://fly.io/docs/flyctl/install/
2. Login:
   - `fly auth login`
3. Desde la raíz del repo (donde está `terminal/fly.toml`):
   - `fly launch --no-deploy`
4. Deploy:
   - `fly deploy`
5. Abrir:
   - `fly status`

## Notas
- Dockerfile usado: `terminal/Dockerfile.vercel` (de propósito está configurado para Streamlit en `PORT`).
- Puerto interno: 8080.


