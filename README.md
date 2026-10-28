# 🌾 Producción de trigo en Argentina

App del Trabajo Práctico de *Introducción a las metodologías y procesos de Ciencia de Datos* (UGR).
Muestra la evolución histórica del trigo (1927–2025), las zonas que más producen y un pronóstico
de la próxima campaña con un modelo Random Forest.

**Datos:** Secretaría de Agricultura, Ganadería y Pesca — [datos.gob.ar](https://datos.gob.ar/dataset/trigo-siembra-cosecha-produccion-rendimiento)

## Correrla en tu compu

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicarla (Streamlit Community Cloud, gratis)

1. Subir esta carpeta a un repositorio **público** de GitHub (tal cual: `app.py`, `requirements.txt`, `data/`, `.streamlit/`).
2. Entrar a [share.streamlit.io](https://share.streamlit.io) con la cuenta de GitHub.
3. **Create app** → elegir el repo, rama `main` y archivo `app.py` → **Deploy**.
4. En un par de minutos queda el link (`https://<nombre>.streamlit.app`). Ese es el link para la notebook.

## Actualizar con una campaña nueva

1. Descargar el CSV actualizado de datos.gob.ar y reemplazar `data/trigo-serie-1927-2025.csv`
   (guardarlo en UTF-8; el original de datos.gob.ar viene en `cp1252`).
2. Hacer commit y push: Streamlit Cloud redespliega solo y el modelo se reentrena con los datos nuevos.
