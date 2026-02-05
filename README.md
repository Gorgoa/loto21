# 🎰 EuroMillones Analyzer Pro

Aplicación de análisis estadístico avanzado para EuroMillones con Machine Learning, Algoritmos Genéticos y Simulaciones Monte Carlo.

> ⚠️ **Aviso:** Esta app es un experimento matemático y educativo. Los sorteos de lotería son eventos aleatorios (i.i.d.). NO se garantiza ninguna predicción real. No se fomenta el juego.

## 🚀 Instalación Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy en Streamlit Community Cloud

1. Sube `app.py` y `requirements.txt` a un repositorio de GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repo y selecciona `app.py`
4. ¡Listo!

## 📋 Funcionalidades

| Pestaña | Descripción |
|---------|-------------|
| 🏠 Inicio | Último sorteo, resumen rápido, top números calientes/fríos |
| 📊 Estadísticas | Frecuencias, parejas, tríos, equilibrio, chi², Poisson, autocorrelación, tendencias |
| 🎯 Sets y Apuestas | 10 sets de 21 números (7 métodos), generación de apuestas, sistemas reducidos |
| 🧬 GA Optimizer | Algoritmo genético (DEAP) para optimizar combinaciones 5+2 |
| 🤖 ML Predictor | Random Forest, XGBoost, K-Means clustering |
| 📈 Backtesting | Test retrospectivo de apuestas vs sorteos reales + comparación vs aleatorio |
| 🎲 Simulaciones | Monte Carlo para probabilidades empíricas y valor esperado |

## 📦 Dependencias

- **UI:** Streamlit, Plotly
- **Datos:** Pandas, NumPy, OpenPyXL
- **Estadística:** SciPy
- **ML:** scikit-learn, XGBoost
- **GA:** DEAP
- **Scraping:** Requests, BeautifulSoup4

## 📊 Fuentes de Datos

- Google Sheets con histórico completo (~1.900 sorteos desde 2004)
- Upload manual de archivos Excel/CSV
- Datos sintéticos para demo (1.900 sorteos generados)
