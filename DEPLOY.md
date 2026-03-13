# Despliegue

## API FastAPI en Render

1. Sube el repositorio a GitHub.
2. En Render, crea un servicio nuevo desde el repo.
3. Si quieres usar despliegue con archivo, Render detectará `render.yaml`.
4. Verifica que el servicio use:
   - Build command: `python3 -m pip install -r project/requirements.txt`
   - Start command: `python3 -m uvicorn project.api.main:app --host 0.0.0.0 --port $PORT`
5. Configura `CORS_ORIGINS` con tu dominio frontend.
6. Espera que Render publique una URL, por ejemplo:
   - `https://pneumonia-api.onrender.com`

## Frontend Next.js en Vercel

1. En Vercel, importa el repositorio.
2. Configura Root Directory: `app`.
3. Agrega variable de entorno:
   - `NEXT_PUBLIC_ANALYSIS_API_URL=https://TU_API_RENDER.onrender.com/predict`
4. Haz deploy.

## Variables recomendadas

- API:
  - `CORS_ORIGINS=https://tu-frontend.vercel.app,http://localhost:3000,http://127.0.0.1:3000`
- Frontend:
  - `NEXT_PUBLIC_ANALYSIS_API_URL=https://tu-api.onrender.com/predict`

## Validación

1. Abre frontend desplegado.
2. Sube una radiografía.
3. Verifica respuesta en pantalla.
4. Verifica en la API:
   - `GET /` responde 200
   - `POST /predict` responde JSON con `prediction` y `confidence`
