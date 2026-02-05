"""
🎰 EuroMillones Analyzer Pro
=============================
Aplicación de análisis estadístico, Machine Learning y Algoritmos Genéticos
para el estudio matemático del sorteo EuroMillones.

⚠️ AVISO: Esta app es un experimento matemático y educativo.
Los sorteos de lotería son eventos aleatorios i.i.d. (independientes e
idénticamente distribuidos). NO se garantiza ninguna predicción real.
No se fomenta el juego ni se ofrece consejo de apuestas.

Autor: Claude AI | Fecha: Febrero 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2_contingency, poisson
from itertools import combinations
from collections import Counter
import random
import warnings
import io
import datetime
import json
import math

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎰 EuroMillones Analyzer Pro",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constantes del juego
NUM_MIN, NUM_MAX = 1, 50          # Números principales: 1-50
STAR_MIN, STAR_MAX = 1, 12       # Estrellas: 1-12
NUMS_PER_DRAW = 5                 # Se eligen 5 números
STARS_PER_DRAW = 2                # Se eligen 2 estrellas

# Categorías de premios EuroMillones (aciertos_nums, aciertos_estrellas): (nombre, premio_aprox_€)
PRIZE_CATEGORIES = {
    (5, 2): ("1ª - Bote", 50_000_000),
    (5, 1): ("2ª", 500_000),
    (5, 0): ("3ª", 50_000),
    (4, 2): ("4ª", 5_000),
    (4, 1): ("5ª", 200),
    (3, 2): ("6ª", 100),
    (4, 0): ("7ª", 50),
    (2, 2): ("8ª", 20),
    (3, 1): ("9ª", 15),
    (3, 0): ("10ª", 12),
    (1, 2): ("11ª", 10),
    (2, 1): ("12ª", 8),
    (2, 0): ("13ª", 4),
}
TICKET_COST = 2.50  # Coste por apuesta en €

# URL de descarga de datos
DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRy91wfK2JteoMi1ZOhGm0D1RKJfDTbEOj6rfnrB6-X7n2Q1nfFwBZBpcivHRdg3pSwxSQgLA3KpW7v/pub?output=xlsx"

# ─────────────────────────────────────────────────────────────
# ESTILOS CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;
        text-align: center; color: white;
    }
    .main-header h1 { color: #f5c842; margin: 0; font-size: 2rem; }
    .main-header p { color: #a0c4ff; margin: 0.3rem 0 0; font-size: 0.95rem; }
    .stat-card {
        background: linear-gradient(135deg, #0f3460, #16213e);
        padding: 1rem 1.2rem; border-radius: 10px; text-align: center;
        color: white; border: 1px solid #1a3a6e;
    }
    .stat-card h3 { color: #f5c842; margin: 0; font-size: 1.5rem; }
    .stat-card p { color: #a0c4ff; margin: 0.2rem 0 0; font-size: 0.85rem; }
    .warning-box {
        background: #2d1b00; border: 1px solid #f5c842; border-radius: 8px;
        padding: 0.8rem; margin: 0.5rem 0; color: #f5c842; font-size: 0.85rem;
    }
    .ball { display: inline-flex; align-items: center; justify-content: center;
        width: 42px; height: 42px; border-radius: 50%; font-weight: bold;
        font-size: 1rem; margin: 2px; color: white; }
    .ball-num { background: linear-gradient(135deg, #1a73e8, #0d47a1); }
    .ball-star { background: linear-gradient(135deg, #f5c842, #e6a800); color: #1a1a2e; }
    .prize-hit { background: #0a3d0a; border: 1px solid #2ecc71; border-radius: 6px;
        padding: 0.4rem 0.8rem; margin: 0.2rem; display: inline-block; color: #2ecc71; }
    div[data-testid="stMetric"] { background: #16213e; padding: 0.8rem;
        border-radius: 8px; border: 1px solid #1a3a6e; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FUNCIONES DE CARGA Y PROCESAMIENTO DE DATOS
# ─────────────────────────────────────────────────────────────

def generate_synthetic_data(n_draws=1900):
    """Genera datos sintéticos de EuroMillones para demo/fallback."""
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp("2026-02-04"), periods=n_draws, freq="3D")
    rows = []
    for d in dates:
        nums = sorted(np.random.choice(range(1, 51), 5, replace=False))
        stars = sorted(np.random.choice(range(1, 13), 2, replace=False))
        rows.append({
            "Fecha": d,
            "Num1": nums[0], "Num2": nums[1], "Num3": nums[2],
            "Num4": nums[3], "Num5": nums[4],
            "Estrella1": stars[0], "Estrella2": stars[1],
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600)
def load_data_from_url(url):
    """Intenta cargar datos desde URL (Google Sheets)."""
    try:
        import requests
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content), engine="openpyxl")
        return df, True
    except Exception as e:
        return None, False


def normalize_dataframe(df):
    """Normaliza columnas del DataFrame al formato estándar."""
    df = df.copy()

    # Intentar detectar columnas de fecha
    date_cols = [c for c in df.columns if any(k in str(c).lower() for k in ["fecha", "date", "día", "dia"])]
    num_cols = [c for c in df.columns if any(k in str(c).lower() for k in ["num", "bola", "ball", "número"])]
    star_cols = [c for c in df.columns if any(k in str(c).lower() for k in ["estrell", "star", "lucky"])]

    # Si las columnas ya tienen el formato esperado
    expected_num = ["Num1", "Num2", "Num3", "Num4", "Num5"]
    expected_star = ["Estrella1", "Estrella2"]

    if all(c in df.columns for c in expected_num + expected_star):
        if date_cols:
            df["Fecha"] = pd.to_datetime(df[date_cols[0]], errors="coerce", dayfirst=True)
        elif "Fecha" not in df.columns:
            df["Fecha"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq="3D")
    else:
        # Intentar mapear columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 7:
            # Asumimos primeras 5 numéricas = números, siguientes 2 = estrellas
            for i, col in enumerate(expected_num):
                if i < len(numeric_cols):
                    df[col] = df[numeric_cols[i]].astype(int)
            for i, col in enumerate(expected_star):
                if 5 + i < len(numeric_cols):
                    df[col] = df[numeric_cols[5 + i]].astype(int)
        if date_cols:
            df["Fecha"] = pd.to_datetime(df[date_cols[0]], errors="coerce", dayfirst=True)
        elif "Fecha" not in df.columns:
            df["Fecha"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq="3D")

    # Validación: números 1-50, estrellas 1-12
    for col in ["Num1", "Num2", "Num3", "Num4", "Num5"]:
        if col in df.columns:
            df[col] = df[col].clip(1, 50).astype(int)
    for col in ["Estrella1", "Estrella2"]:
        if col in df.columns:
            df[col] = df[col].clip(1, 12).astype(int)

    df = df.sort_values("Fecha").reset_index(drop=True)
    return df


def get_all_numbers(df):
    """Extrae todos los números principales como lista plana."""
    cols = ["Num1", "Num2", "Num3", "Num4", "Num5"]
    return df[cols].values.flatten().tolist()


def get_all_stars(df):
    """Extrae todas las estrellas como lista plana."""
    cols = ["Estrella1", "Estrella2"]
    return df[cols].values.flatten().tolist()


def get_draw_numbers(row):
    """Extrae números y estrellas de una fila."""
    nums = [int(row["Num1"]), int(row["Num2"]), int(row["Num3"]),
            int(row["Num4"]), int(row["Num5"])]
    stars = [int(row["Estrella1"]), int(row["Estrella2"])]
    return nums, stars


# ─────────────────────────────────────────────────────────────
# FUNCIONES ESTADÍSTICAS
# ─────────────────────────────────────────────────────────────

def calc_frequencies(numbers, min_val, max_val):
    """Calcula frecuencias de aparición."""
    counter = Counter(numbers)
    return {n: counter.get(n, 0) for n in range(min_val, max_val + 1)}


def calc_pair_frequencies(df, top_n=20):
    """Calcula las parejas de números más frecuentes."""
    pair_counter = Counter()
    for _, row in df.iterrows():
        nums = sorted([int(row[f"Num{i}"]) for i in range(1, 6)])
        for pair in combinations(nums, 2):
            pair_counter[pair] += 1
    return pair_counter.most_common(top_n)


def calc_trio_frequencies(df, top_n=15):
    """Calcula los tríos de números más frecuentes."""
    trio_counter = Counter()
    for _, row in df.iterrows():
        nums = sorted([int(row[f"Num{i}"]) for i in range(1, 6)])
        for trio in combinations(nums, 3):
            trio_counter[trio] += 1
    return trio_counter.most_common(top_n)


def chi_square_test(frequencies, expected_freq=None):
    """Test chi-cuadrado de uniformidad."""
    observed = list(frequencies.values())
    if expected_freq is None:
        expected_freq = np.mean(observed)
    expected = [expected_freq] * len(observed)
    chi2, p_value = stats.chisquare(observed, f_exp=expected)
    return chi2, p_value


def calc_autocorrelation(numbers_series, max_lag=10):
    """Calcula autocorrelación temporal para una serie de frecuencias."""
    series = pd.Series(numbers_series)
    autocorr = [series.autocorr(lag=i) for i in range(1, max_lag + 1)]
    return autocorr


def calc_balance_stats(df):
    """Calcula estadísticas de equilibrio (pares/impares, bajos/altos)."""
    results = []
    for _, row in df.iterrows():
        nums = [int(row[f"Num{i}"]) for i in range(1, 6)]
        n_even = sum(1 for n in nums if n % 2 == 0)
        n_odd = 5 - n_even
        n_low = sum(1 for n in nums if n <= 25)
        n_high = 5 - n_low
        decades = [0] * 5
        for n in nums:
            decades[min((n - 1) // 10, 4)] += 1
        results.append({
            "Pares": n_even, "Impares": n_odd,
            "Bajos(1-25)": n_low, "Altos(26-50)": n_high,
            "Dec1(1-10)": decades[0], "Dec2(11-20)": decades[1],
            "Dec3(21-30)": decades[2], "Dec4(31-40)": decades[3],
            "Dec5(41-50)": decades[4],
        })
    return pd.DataFrame(results)


def calc_consecutive_stats(df):
    """Calcula frecuencia de números consecutivos en cada sorteo."""
    consec_counts = []
    for _, row in df.iterrows():
        nums = sorted([int(row[f"Num{i}"]) for i in range(1, 6)])
        consec = sum(1 for i in range(len(nums) - 1) if nums[i + 1] - nums[i] == 1)
        consec_counts.append(consec)
    return consec_counts


def calc_sum_stats(df):
    """Calcula la suma total de números por sorteo."""
    sums = []
    for _, row in df.iterrows():
        s = sum(int(row[f"Num{i}"]) for i in range(1, 6))
        sums.append(s)
    return sums


# ─────────────────────────────────────────────────────────────
# GENERACIÓN DE SETS Y APUESTAS
# ─────────────────────────────────────────────────────────────

def generate_set_21(df, method="balanced", n_recent=50):
    """
    Genera un set de 21 números principales usando diferentes métodos.
    Métodos: balanced, hot, cold, recent_hot, pairs, decades, random_smart
    """
    all_nums = get_all_numbers(df)
    freq = calc_frequencies(all_nums, NUM_MIN, NUM_MAX)

    recent_df = df.tail(n_recent) if len(df) > n_recent else df
    recent_nums = get_all_numbers(recent_df)
    recent_freq = calc_frequencies(recent_nums, NUM_MIN, NUM_MAX)

    sorted_by_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    sorted_by_recent = sorted(recent_freq.items(), key=lambda x: x[1], reverse=True)

    if method == "hot":
        # Top 21 más frecuentes históricamente
        selected = [n for n, _ in sorted_by_freq[:21]]

    elif method == "cold":
        # 10 más fríos + 11 más calientes (contrarian mix)
        cold = [n for n, _ in sorted_by_freq[-15:]]
        hot = [n for n, _ in sorted_by_freq[:10]]
        selected = list(set(cold + hot))[:21]

    elif method == "recent_hot":
        # Top 21 más frecuentes en últimos N sorteos
        selected = [n for n, _ in sorted_by_recent[:21]]

    elif method == "pairs":
        # Basado en las parejas más frecuentes
        pairs = calc_pair_frequencies(df, top_n=30)
        pair_nums = set()
        for (a, b), _ in pairs:
            pair_nums.add(a)
            pair_nums.add(b)
            if len(pair_nums) >= 21:
                break
        selected = list(pair_nums)[:21]

    elif method == "decades":
        # Equilibrado por décadas: ~4-5 de cada década
        selected = []
        for dec_start in [1, 11, 21, 31, 41]:
            dec_end = min(dec_start + 9, 50)
            dec_freq = {n: freq[n] for n in range(dec_start, dec_end + 1)}
            top = sorted(dec_freq.items(), key=lambda x: x[1], reverse=True)
            selected.extend([n for n, _ in top[:4]])
        # Completar hasta 21 con los más frecuentes restantes
        remaining = [n for n, _ in sorted_by_freq if n not in selected]
        selected.extend(remaining[:21 - len(selected)])

    elif method == "random_smart":
        # Aleatorio ponderado por frecuencia
        weights = np.array([freq[n] for n in range(1, 51)], dtype=float)
        weights /= weights.sum()
        selected = list(np.random.choice(range(1, 51), 21, replace=False, p=weights))

    else:  # balanced
        # Mezcla de calientes históricos, recientes y equilibrio por décadas
        hot_hist = [n for n, _ in sorted_by_freq[:10]]
        hot_recent = [n for n, _ in sorted_by_recent[:8]]
        cold_nums = [n for n, _ in sorted_by_freq[-5:]]
        combined = list(set(hot_hist + hot_recent + cold_nums))
        if len(combined) > 21:
            combined = combined[:21]
        elif len(combined) < 21:
            remaining = [n for n in range(1, 51) if n not in combined]
            random.shuffle(remaining)
            combined.extend(remaining[:21 - len(combined)])
        selected = combined

    return sorted(selected[:21])


def generate_10_sets(df, n_recent=50):
    """Genera 10 sets variados de 21 números."""
    methods = [
        ("Equilibrado", "balanced"),
        ("Calientes Históricos", "hot"),
        ("Fríos + Calientes", "cold"),
        ("Calientes Recientes", "recent_hot"),
        ("Basado en Parejas", "pairs"),
        ("Equilibrado por Décadas", "decades"),
        ("Aleatorio Ponderado 1", "random_smart"),
        ("Aleatorio Ponderado 2", "random_smart"),
        ("Equilibrado 2", "balanced"),
        ("Calientes Recientes 2", "recent_hot"),
    ]
    sets = []
    for name, method in methods:
        s = generate_set_21(df, method=method, n_recent=n_recent)
        sets.append((name, s))
    return sets


def generate_stars(df, n=2, n_recent=50):
    """Genera estrellas más probables."""
    all_stars = get_all_stars(df)
    freq = calc_frequencies(all_stars, STAR_MIN, STAR_MAX)
    recent_stars = get_all_stars(df.tail(n_recent) if len(df) > n_recent else df)
    recent_freq = calc_frequencies(recent_stars, STAR_MIN, STAR_MAX)

    # Combinar frecuencia histórica y reciente
    combined = {}
    for s in range(STAR_MIN, STAR_MAX + 1):
        combined[s] = freq[s] * 0.6 + recent_freq[s] * 0.4
    sorted_stars = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [s for s, _ in sorted_stars[:n]]


def generate_bets_from_set(set_21, df, n_bets=3, n_recent=50):
    """Genera N apuestas (5 nums + 2 estrellas) a partir de un set de 21."""
    all_nums = get_all_numbers(df)
    freq = calc_frequencies(all_nums, NUM_MIN, NUM_MAX)
    stars = generate_stars(df, n=4, n_recent=n_recent)

    bets = []
    for i in range(n_bets):
        if i == 0:
            # Primera apuesta: top 5 del set por frecuencia
            weighted = sorted([(n, freq.get(n, 0)) for n in set_21],
                              key=lambda x: x[1], reverse=True)
            nums = sorted([n for n, _ in weighted[:5]])
        else:
            # Siguientes: aleatorio ponderado dentro del set
            weights = np.array([freq.get(n, 1) for n in set_21], dtype=float)
            weights /= weights.sum()
            nums = sorted(np.random.choice(set_21, 5, replace=False, p=weights).tolist())

        bet_stars = sorted(random.sample(stars, 2))
        bets.append({"nums": nums, "stars": bet_stars})
    return bets


# ─────────────────────────────────────────────────────────────
# SISTEMA REDUCIDO
# ─────────────────────────────────────────────────────────────

def calc_reduced_system(set_21, guarantee="3si5"):
    """
    Calcula combinaciones para un sistema reducido.
    guarantee: '3si5' = garantiza 3 aciertos si 5 están en el set
               '4si5' = garantiza 4 aciertos si 5 están en el set
    Retorna lista de combinaciones de 5 números.
    """
    n = len(set_21)
    all_combos = list(combinations(set_21, 5))

    if guarantee == "3si5":
        # Covering design: seleccionar subconjunto que cubra todos los tríos
        trios_needed = set(combinations(set_21, 3))
        selected = []
        remaining_combos = list(all_combos)
        random.shuffle(remaining_combos)

        while trios_needed:
            best_combo = None
            best_cover = 0
            # Evaluar un subconjunto para eficiencia
            sample = remaining_combos[:min(500, len(remaining_combos))]
            for combo in sample:
                combo_trios = set(combinations(combo, 3))
                cover = len(combo_trios & trios_needed)
                if cover > best_cover:
                    best_cover = cover
                    best_combo = combo
            if best_combo is None:
                break
            selected.append(best_combo)
            combo_trios = set(combinations(best_combo, 3))
            trios_needed -= combo_trios
            if best_combo in remaining_combos:
                remaining_combos.remove(best_combo)
        return selected

    elif guarantee == "4si5":
        # Más estricto: cubrir todos los cuartetos
        quads_needed = set(combinations(set_21, 4))
        selected = []
        remaining_combos = list(all_combos)
        random.shuffle(remaining_combos)

        while quads_needed:
            best_combo = None
            best_cover = 0
            sample = remaining_combos[:min(500, len(remaining_combos))]
            for combo in sample:
                combo_quads = set(combinations(combo, 4))
                cover = len(combo_quads & quads_needed)
                if cover > best_cover:
                    best_cover = cover
                    best_combo = combo
            if best_combo is None:
                break
            selected.append(best_combo)
            combo_quads = set(combinations(best_combo, 4))
            quads_needed -= combo_quads
            if best_combo in remaining_combos:
                remaining_combos.remove(best_combo)
        return selected

    return all_combos[:100]  # Fallback


# ─────────────────────────────────────────────────────────────
# BACKTESTING
# ─────────────────────────────────────────────────────────────

def check_prize(bet_nums, bet_stars, draw_nums, draw_stars):
    """Comprueba si una apuesta tiene premio y devuelve categoría."""
    num_matches = len(set(bet_nums) & set(draw_nums))
    star_matches = len(set(bet_stars) & set(draw_stars))
    key = (num_matches, star_matches)
    if key in PRIZE_CATEGORIES:
        name, amount = PRIZE_CATEGORIES[key]
        return name, amount, num_matches, star_matches
    return None, 0, num_matches, star_matches


def backtest_bets(bets, df, n_draws=20):
    """
    Realiza backtesting de apuestas contra los últimos N sorteos.
    Retorna DataFrame con resultados detallados.
    """
    test_df = df.tail(n_draws)
    results = []
    total_cost = len(bets) * n_draws * TICKET_COST
    total_prizes = 0

    for idx, draw in test_df.iterrows():
        draw_nums, draw_stars = get_draw_numbers(draw)
        draw_date = draw["Fecha"]

        for b_idx, bet in enumerate(bets):
            prize_name, prize_amount, n_match, s_match = check_prize(
                bet["nums"], bet["stars"], draw_nums, draw_stars
            )
            total_prizes += prize_amount
            results.append({
                "Fecha": draw_date,
                "Apuesta": f"#{b_idx + 1}: {bet['nums']} + {bet['stars']}",
                "Sorteo": f"{draw_nums} + {draw_stars}",
                "Nums_OK": n_match,
                "Stars_OK": s_match,
                "Premio": prize_name or "-",
                "Importe_€": prize_amount,
            })

    results_df = pd.DataFrame(results)
    roi = ((total_prizes - total_cost) / total_cost * 100) if total_cost > 0 else 0

    summary = {
        "total_draws": n_draws,
        "total_bets": len(bets),
        "total_tickets": len(bets) * n_draws,
        "total_cost": total_cost,
        "total_prizes": total_prizes,
        "net_balance": total_prizes - total_cost,
        "roi_pct": roi,
        "prize_counts": results_df[results_df["Importe_€"] > 0]["Premio"].value_counts().to_dict(),
    }
    return results_df, summary


# ─────────────────────────────────────────────────────────────
# ALGORITMO GENÉTICO
# ─────────────────────────────────────────────────────────────

def run_genetic_algorithm(df, pop_size=50, n_gen=30, mut_prob=0.1, n_recent=100, progress_bar=None):
    """
    Algoritmo Genético para optimizar combinaciones de 5 números + 2 estrellas.
    Fitness: combinación de coincidencias históricas + equilibrio + diversidad.
    """
    try:
        from deap import base, creator, tools, algorithms
    except ImportError:
        st.error("❌ Librería DEAP no instalada. Ejecuta: pip install deap")
        return None, None

    # Datos para fitness
    recent_df = df.tail(n_recent) if len(df) > n_recent else df
    all_nums = get_all_numbers(df)
    freq = calc_frequencies(all_nums, NUM_MIN, NUM_MAX)
    max_freq = max(freq.values()) if freq.values() else 1

    all_stars = get_all_stars(df)
    star_freq = calc_frequencies(all_stars, STAR_MIN, STAR_MAX)

    historical_draws = []
    for _, row in recent_df.iterrows():
        nums, stars = get_draw_numbers(row)
        historical_draws.append((set(nums), set(stars)))

    # Setup DEAP (evitar recrear clases)
    if not hasattr(creator, "FitnessMax_EM"):
        creator.create("FitnessMax_EM", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual_EM"):
        creator.create("Individual_EM", list, fitness=creator.FitnessMax_EM)

    toolbox = base.Toolbox()

    def create_individual():
        nums = sorted(random.sample(range(NUM_MIN, NUM_MAX + 1), 5))
        stars = sorted(random.sample(range(STAR_MIN, STAR_MAX + 1), 2))
        return creator.Individual_EM(nums + stars)

    def evaluate(individual):
        nums = set(individual[:5])
        stars = set(individual[5:])

        # 1. Frecuencia normalizada (30%)
        freq_score = sum(freq.get(n, 0) for n in nums) / (5 * max_freq)

        # 2. Coincidencias históricas (40%)
        match_score = 0
        for h_nums, h_stars in historical_draws:
            n_match = len(nums & h_nums)
            s_match = len(stars & h_stars)
            if n_match >= 3:
                match_score += n_match * 2 + s_match
            elif n_match >= 2:
                match_score += n_match + s_match * 0.5
        match_score /= max(len(historical_draws), 1) * 12

        # 3. Equilibrio (20%)
        n_even = sum(1 for n in nums if n % 2 == 0)
        n_low = sum(1 for n in nums if n <= 25)
        balance_score = 1.0 - (abs(n_even - 2.5) + abs(n_low - 2.5)) / 5.0

        # 4. Cobertura de décadas (10%)
        decades = set((n - 1) // 10 for n in nums)
        decade_score = len(decades) / 5.0

        fitness = 0.3 * freq_score + 0.4 * match_score + 0.2 * balance_score + 0.1 * decade_score
        return (fitness,)

    def mutate(individual):
        if random.random() < 0.7:
            # Mutar un número
            idx = random.randint(0, 4)
            new_nums = list(set(range(NUM_MIN, NUM_MAX + 1)) - set(individual[:5]))
            individual[idx] = random.choice(new_nums)
            individual[:5] = sorted(individual[:5])
        else:
            # Mutar una estrella
            idx = random.randint(5, 6)
            new_stars = list(set(range(STAR_MIN, STAR_MAX + 1)) - set(individual[5:]))
            individual[idx] = random.choice(new_stars)
            individual[5:] = sorted(individual[5:])
        return (individual,)

    def crossover(ind1, ind2):
        # Crossover de números
        nums1 = set(ind1[:5])
        nums2 = set(ind2[:5])
        all_n = list(nums1 | nums2)
        if len(all_n) >= 10:
            new1 = sorted(random.sample(all_n, 5))
            new2 = sorted(random.sample(all_n, 5))
            ind1[:5] = new1
            ind2[:5] = new2
        # Crossover de estrellas
        stars1 = set(ind1[5:])
        stars2 = set(ind2[5:])
        all_s = list(stars1 | stars2)
        if len(all_s) >= 4:
            ind1[5:] = sorted(random.sample(all_s, 2))
            ind2[5:] = sorted(random.sample(all_s, 2))
        return ind1, ind2

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Evolución
    pop = toolbox.population(n=pop_size)
    stats_log = tools.Statistics(lambda ind: ind.fitness.values)
    stats_log.register("max", np.max)
    stats_log.register("avg", np.mean)

    fitness_history = []

    for gen in range(n_gen):
        # Evaluar
        invalid = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        record = stats_log.compile(pop)
        fitness_history.append({"gen": gen, "max": record["max"], "avg": record["avg"]})

        if progress_bar:
            progress_bar.progress((gen + 1) / n_gen, f"Generación {gen + 1}/{n_gen}")

        # Selección y reproducción
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.7:
                toolbox.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        for ind in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(ind)
                del ind.fitness.values

        pop[:] = offspring

    # Evaluar final
    invalid = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid)
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit

    # Mejores individuos
    top = tools.selBest(pop, 10)
    results = []
    for ind in top:
        results.append({
            "nums": sorted(ind[:5]),
            "stars": sorted(ind[5:]),
            "fitness": ind.fitness.values[0],
        })

    return results, pd.DataFrame(fitness_history)


# ─────────────────────────────────────────────────────────────
# MACHINE LEARNING
# ─────────────────────────────────────────────────────────────

def prepare_ml_features(df, window=10):
    """Prepara features para modelos ML basados en ventanas temporales."""
    features_list = []
    targets_list = []

    all_nums = get_all_numbers(df)
    global_freq = calc_frequencies(all_nums, NUM_MIN, NUM_MAX)

    for i in range(window, len(df)):
        window_df = df.iloc[i - window:i]
        w_nums = get_all_numbers(window_df)
        w_freq = calc_frequencies(w_nums, NUM_MIN, NUM_MAX)

        # Features: frecuencias en ventana para cada número (50 features)
        feat = [w_freq.get(n, 0) / (window * 5) for n in range(1, 51)]

        # Features adicionales: estrellas (12 features)
        w_stars = get_all_stars(window_df)
        s_freq = calc_frequencies(w_stars, STAR_MIN, STAR_MAX)
        feat.extend([s_freq.get(s, 0) / (window * 2) for s in range(1, 13)])

        # Features: equilibrio del último sorteo
        last_row = df.iloc[i - 1]
        last_nums = [int(last_row[f"Num{j}"]) for j in range(1, 6)]
        feat.append(sum(1 for n in last_nums if n % 2 == 0) / 5.0)
        feat.append(sum(1 for n in last_nums if n <= 25) / 5.0)
        feat.append(sum(last_nums) / 250.0)

        features_list.append(feat)

        # Target: números del sorteo actual (one-hot para 50 números)
        current_nums = [int(df.iloc[i][f"Num{j}"]) for j in range(1, 6)]
        target = [1 if n in current_nums else 0 for n in range(1, 51)]
        targets_list.append(target)

    return np.array(features_list), np.array(targets_list)


def train_random_forest(df, window=10, n_recent=200):
    """Entrena Random Forest para predecir números probables."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    subset = df.tail(n_recent + window) if len(df) > n_recent + window else df
    X, y = prepare_ml_features(subset, window)

    if len(X) < 20:
        return None, None

    # Train/test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    )
    model.fit(X_train, y_train)

    # Predecir probabilidades para el último sorteo
    last_features = X[-1:].copy()
    probas = []
    for estimator in model.estimators_:
        try:
            p = estimator.predict_proba(last_features)[0]
            probas.append(p[1] if len(p) > 1 else p[0])
        except Exception:
            probas.append(0.5)

    # Score
    score = model.score(X_test, y_test)

    return probas, score


def train_xgboost(df, window=10, n_recent=200):
    """Entrena XGBoost para predecir números probables."""
    try:
        from xgboost import XGBClassifier
        from sklearn.multioutput import MultiOutputClassifier
    except ImportError:
        return None, None

    subset = df.tail(n_recent + window) if len(df) > n_recent + window else df
    X, y = prepare_ml_features(subset, window)

    if len(X) < 20:
        return None, None

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = MultiOutputClassifier(
        XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                      random_state=42, use_label_encoder=False, eval_metric='logloss',
                      verbosity=0)
    )
    model.fit(X_train, y_train)

    last_features = X[-1:]
    probas = []
    for estimator in model.estimators_:
        try:
            p = estimator.predict_proba(last_features)[0]
            probas.append(p[1] if len(p) > 1 else p[0])
        except Exception:
            probas.append(0.5)

    score = model.score(X_test, y_test)
    return probas, score


def train_clustering(df, n_clusters=5, n_recent=200):
    """K-Means clustering de sorteos para identificar patrones."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    subset = df.tail(n_recent) if len(df) > n_recent else df
    features = []
    for _, row in subset.iterrows():
        nums = [int(row[f"Num{i}"]) for i in range(1, 6)]
        stars = [int(row["Estrella1"]), int(row["Estrella2"])]
        n_even = sum(1 for n in nums if n % 2 == 0)
        n_low = sum(1 for n in nums if n <= 25)
        total = sum(nums)
        spread = max(nums) - min(nums)
        features.append(nums + stars + [n_even, n_low, total, spread])

    X = np.array(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Encontrar cluster más frecuente en últimos 10 sorteos
    recent_labels = labels[-10:]
    hot_cluster = Counter(recent_labels).most_common(1)[0][0]

    # Obtener centroide del cluster caliente
    centroid = kmeans.cluster_centers_[hot_cluster]
    centroid_original = scaler.inverse_transform(centroid.reshape(1, -1))[0]

    # Números sugeridos del centroide
    suggested_nums = sorted([int(round(centroid_original[i])) for i in range(5)])
    suggested_nums = [max(1, min(50, n)) for n in suggested_nums]
    # Evitar duplicados
    while len(set(suggested_nums)) < 5:
        for i in range(len(suggested_nums)):
            if suggested_nums.count(suggested_nums[i]) > 1:
                suggested_nums[i] = random.choice([n for n in range(1, 51) if n not in suggested_nums])
        suggested_nums = sorted(suggested_nums)

    suggested_stars = sorted([int(round(centroid_original[i])) for i in range(5, 7)])
    suggested_stars = [max(1, min(12, s)) for s in suggested_stars]
    if suggested_stars[0] == suggested_stars[1]:
        suggested_stars[1] = min(12, suggested_stars[1] + 1)

    return {
        "labels": labels,
        "hot_cluster": hot_cluster,
        "n_clusters": n_clusters,
        "suggested_nums": suggested_nums,
        "suggested_stars": suggested_stars,
        "cluster_sizes": dict(Counter(labels)),
    }


# ─────────────────────────────────────────────────────────────
# SIMULACIÓN MONTE CARLO
# ─────────────────────────────────────────────────────────────

def monte_carlo_simulation(bet_nums, bet_stars, n_iterations=10000, progress_bar=None):
    """
    Simulación Monte Carlo: simula N sorteos aleatorios y calcula
    probabilidades empíricas de cada categoría de premio.
    """
    prize_counter = Counter()
    match_distribution = Counter()

    for i in range(n_iterations):
        draw_nums = sorted(random.sample(range(1, 51), 5))
        draw_stars = sorted(random.sample(range(1, 13), 2))

        n_match = len(set(bet_nums) & set(draw_nums))
        s_match = len(set(bet_stars) & set(draw_stars))
        match_distribution[(n_match, s_match)] += 1

        key = (n_match, s_match)
        if key in PRIZE_CATEGORIES:
            prize_counter[key] += 1

        if progress_bar and (i + 1) % (n_iterations // 20) == 0:
            progress_bar.progress((i + 1) / n_iterations)

    results = {}
    for key, (name, amount) in PRIZE_CATEGORIES.items():
        count = prize_counter.get(key, 0)
        prob = count / n_iterations
        results[name] = {
            "count": count,
            "prob": prob,
            "odds": f"1:{int(1 / prob)}" if prob > 0 else "0",
            "prize": amount,
            "expected_value": prob * amount,
        }

    total_ev = sum(r["expected_value"] for r in results.values())

    return results, total_ev, match_distribution


# ─────────────────────────────────────────────────────────────
# FUNCIONES DE VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────

def render_balls(nums, stars):
    """Renderiza números como bolas coloreadas."""
    html = ""
    for n in nums:
        html += f'<span class="ball ball-num">{n}</span>'
    html += "&nbsp;&nbsp;"
    for s in stars:
        html += f'<span class="ball ball-star">★{s}</span>'
    return html


def plot_frequency_chart(freq_dict, title, color="#1a73e8"):
    """Genera gráfico de barras de frecuencias con Plotly."""
    df_plot = pd.DataFrame(list(freq_dict.items()), columns=["Número", "Frecuencia"])
    df_plot = df_plot.sort_values("Número")

    fig = px.bar(df_plot, x="Número", y="Frecuencia", title=title,
                 color="Frecuencia", color_continuous_scale="Blues")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    return fig


def plot_heatmap_pairs(pairs, title="Parejas Más Frecuentes"):
    """Genera heatmap de parejas frecuentes."""
    if not pairs:
        return None
    data = []
    for (a, b), count in pairs[:15]:
        data.append({"Num A": str(a), "Num B": str(b), "Frecuencia": count})
    df_plot = pd.DataFrame(data)

    fig = px.bar(df_plot, x=[f"{r['Num A']}-{r['Num B']}" for _, r in df_plot.iterrows()],
                 y="Frecuencia", title=title, color="Frecuencia",
                 color_continuous_scale="Oranges")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=350, xaxis_title="Pareja",
    )
    return fig


# ─────────────────────────────────────────────────────────────
# INTERFAZ PRINCIPAL - STREAMLIT
# ─────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎰 EuroMillones Analyzer Pro</h1>
        <p>Análisis estadístico • Machine Learning • Algoritmos Genéticos • Simulaciones</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
        ⚠️ <b>Aviso Legal:</b> Esta aplicación es un <b>experimento matemático y educativo</b>.
        Los sorteos de lotería son eventos aleatorios (i.i.d.). NO se garantiza ninguna predicción.
        No se fomenta el juego ni se ofrece consejo de apuestas.
    </div>
    """, unsafe_allow_html=True)

    # ── Carga de datos ──
    if "df" not in st.session_state:
        st.session_state.df = None

    with st.sidebar:
        st.markdown("### 📊 Fuente de Datos")

        data_source = st.radio(
            "Seleccionar fuente:",
            ["📥 Descargar de Google Sheets", "📁 Subir archivo propio", "🎲 Datos sintéticos (demo)"],
            index=2,
        )

        if data_source == "📥 Descargar de Google Sheets":
            if st.button("🔄 Descargar datos", use_container_width=True):
                with st.spinner("Descargando..."):
                    df_raw, success = load_data_from_url(DATA_URL)
                    if success and df_raw is not None:
                        st.session_state.df = normalize_dataframe(df_raw)
                        st.success(f"✅ {len(st.session_state.df)} sorteos cargados")
                    else:
                        st.error("❌ Error al descargar. Prueba datos sintéticos.")

        elif data_source == "📁 Subir archivo propio":
            uploaded = st.file_uploader("Sube Excel o CSV", type=["xlsx", "xls", "csv"])
            if uploaded:
                try:
                    if uploaded.name.endswith(".csv"):
                        df_raw = pd.read_csv(uploaded)
                    else:
                        df_raw = pd.read_excel(uploaded, engine="openpyxl")
                    st.session_state.df = normalize_dataframe(df_raw)
                    st.success(f"✅ {len(st.session_state.df)} sorteos cargados")
                except Exception as e:
                    st.error(f"Error: {e}")

        else:  # Sintéticos
            if st.session_state.df is None or st.button("🎲 Generar datos demo", use_container_width=True):
                st.session_state.df = generate_synthetic_data(1900)
                st.info("ℹ️ Usando 1.900 sorteos sintéticos para demostración")

        if st.session_state.df is not None:
            df = st.session_state.df
            st.markdown("---")
            st.markdown(f"**Total sorteos:** {len(df)}")
            st.markdown(f"**Desde:** {df['Fecha'].min().strftime('%d/%m/%Y')}")
            st.markdown(f"**Hasta:** {df['Fecha'].max().strftime('%d/%m/%Y')}")

            st.markdown("---")
            n_recent = st.slider("📅 Sorteos recientes para análisis", 20, min(500, len(df)), 50)
        else:
            n_recent = 50

    if st.session_state.df is None:
        st.info("👈 Selecciona una fuente de datos en la barra lateral para comenzar.")
        return

    df = st.session_state.df

    # ── Tabs principales ──
    tabs = st.tabs([
        "🏠 Inicio",
        "📊 Estadísticas",
        "🎯 Sets y Apuestas",
        "🧬 GA Optimizer",
        "🤖 ML Predictor",
        "📈 Backtesting",
        "🎲 Simulaciones",
    ])

    # ════════════════════════════════════════════════════════════
    # TAB 0: INICIO
    # ════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("## 🏠 Último Sorteo y Resumen")

        last_draw = df.iloc[-1]
        last_nums, last_stars = get_draw_numbers(last_draw)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"### Último sorteo: {last_draw['Fecha'].strftime('%d/%m/%Y')}")
            st.markdown(render_balls(last_nums, last_stars), unsafe_allow_html=True)

        with col2:
            st.metric("Total Sorteos", f"{len(df):,}")
            st.metric("Años de datos", f"{(df['Fecha'].max() - df['Fecha'].min()).days // 365}")

        st.markdown("---")

        # Resumen rápido
        col1, col2, col3, col4 = st.columns(4)

        all_nums = get_all_numbers(df)
        freq = calc_frequencies(all_nums, NUM_MIN, NUM_MAX)
        top_nums = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]

        all_stars = get_all_stars(df)
        star_freq = calc_frequencies(all_stars, STAR_MIN, STAR_MAX)
        top_stars = sorted(star_freq.items(), key=lambda x: x[1], reverse=True)[:3]

        with col1:
            st.markdown('<div class="stat-card"><h3>🔥</h3><p>Nums más frecuentes</p></div>', unsafe_allow_html=True)
            for n, f in top_nums:
                st.write(f"**{n}** → {f} veces")

        with col2:
            st.markdown('<div class="stat-card"><h3>⭐</h3><p>Estrellas top</p></div>', unsafe_allow_html=True)
            for s, f in top_stars:
                st.write(f"**★{s}** → {f} veces")

        cold_nums = sorted(freq.items(), key=lambda x: x[1])[:5]
        with col3:
            st.markdown('<div class="stat-card"><h3>❄️</h3><p>Nums más fríos</p></div>', unsafe_allow_html=True)
            for n, f in cold_nums:
                st.write(f"**{n}** → {f} veces")

        with col4:
            sums = calc_sum_stats(df)
            st.markdown('<div class="stat-card"><h3>Σ</h3><p>Suma media</p></div>', unsafe_allow_html=True)
            st.write(f"**Media:** {np.mean(sums):.1f}")
            st.write(f"**Mediana:** {np.median(sums):.0f}")
            st.write(f"**Rango:** {min(sums)}-{max(sums)}")

        # Últimos 10 sorteos
        st.markdown("### 📋 Últimos 10 Sorteos")
        last_10 = df.tail(10).sort_values("Fecha", ascending=False)
        display_df = last_10[["Fecha", "Num1", "Num2", "Num3", "Num4", "Num5", "Estrella1", "Estrella2"]].copy()
        display_df["Fecha"] = display_df["Fecha"].dt.strftime("%d/%m/%Y")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════
    # TAB 1: ESTADÍSTICAS
    # ════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("## 📊 Análisis Estadístico Completo")

        stat_sub = st.radio(
            "Sección:",
            ["Frecuencias", "Parejas y Tríos", "Equilibrio", "Tests Estadísticos", "Tendencias"],
            horizontal=True,
        )

        if stat_sub == "Frecuencias":
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Frecuencia de Números (1-50)")
                fig = plot_frequency_chart(freq, "Frecuencia Histórica - Números")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### Frecuencia de Estrellas (1-12)")
                fig = plot_frequency_chart(star_freq, "Frecuencia Histórica - Estrellas", color="#f5c842")
                st.plotly_chart(fig, use_container_width=True)

            # Frecuencias recientes
            st.markdown(f"### Frecuencias en Últimos {n_recent} Sorteos")
            recent_df = df.tail(n_recent)
            recent_nums = get_all_numbers(recent_df)
            recent_freq = calc_frequencies(recent_nums, NUM_MIN, NUM_MAX)
            fig = plot_frequency_chart(recent_freq, f"Frecuencia Reciente ({n_recent} sorteos)")
            st.plotly_chart(fig, use_container_width=True)

        elif stat_sub == "Parejas y Tríos":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 👫 Top 20 Parejas Más Frecuentes")
                pairs = calc_pair_frequencies(df, top_n=20)
                fig = plot_heatmap_pairs(pairs)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                pairs_df = pd.DataFrame([(f"{a}-{b}", c) for (a, b), c in pairs],
                                        columns=["Pareja", "Frecuencia"])
                st.dataframe(pairs_df, hide_index=True, use_container_width=True)

            with col2:
                st.markdown("### 👥 Top 15 Tríos Más Frecuentes")
                trios = calc_trio_frequencies(df, top_n=15)
                trios_df = pd.DataFrame([(f"{a}-{b}-{c}", cnt) for (a, b, c), cnt in trios],
                                        columns=["Trío", "Frecuencia"])
                st.dataframe(trios_df, hide_index=True, use_container_width=True)

        elif stat_sub == "Equilibrio":
            balance = calc_balance_stats(df)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Distribución Pares/Impares")
                even_odd = balance["Pares"].value_counts().sort_index()
                fig = px.bar(x=even_odd.index, y=even_odd.values,
                             labels={"x": "Nº Pares (de 5)", "y": "Frecuencia"},
                             title="Distribución de números pares por sorteo")
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### Distribución Bajos/Altos")
                low_high = balance["Bajos(1-25)"].value_counts().sort_index()
                fig = px.bar(x=low_high.index, y=low_high.values,
                             labels={"x": "Nº Bajos 1-25 (de 5)", "y": "Frecuencia"},
                             title="Distribución bajos (1-25) por sorteo")
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Distribución por Décadas")
            decades_cols = ["Dec1(1-10)", "Dec2(11-20)", "Dec3(21-30)", "Dec4(31-40)", "Dec5(41-50)"]
            decade_means = balance[decades_cols].mean()
            fig = px.bar(x=["1-10", "11-20", "21-30", "31-40", "41-50"],
                         y=decade_means.values,
                         labels={"x": "Década", "y": "Media de números"},
                         title="Media de números por década por sorteo")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            # Consecutivos
            consec = calc_consecutive_stats(df)
            st.markdown("### Números Consecutivos")
            consec_counts = Counter(consec)
            fig = px.bar(x=list(consec_counts.keys()), y=list(consec_counts.values()),
                         labels={"x": "Nº pares consecutivos", "y": "Frecuencia"},
                         title="Frecuencia de números consecutivos por sorteo")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        elif stat_sub == "Tests Estadísticos":
            st.markdown("### 📐 Test Chi-Cuadrado de Uniformidad")

            chi2_num, p_num = chi_square_test(freq)
            chi2_star, p_star = chi_square_test(star_freq)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Chi² Números", f"{chi2_num:.2f}")
                st.metric("p-valor", f"{p_num:.4f}")
                if p_num > 0.05:
                    st.success("✅ No se rechaza H₀: distribución uniforme (p > 0.05)")
                else:
                    st.warning("⚠️ Se rechaza H₀: distribución NO uniforme (p ≤ 0.05)")

            with col2:
                st.metric("Chi² Estrellas", f"{chi2_star:.2f}")
                st.metric("p-valor", f"{p_star:.4f}")
                if p_star > 0.05:
                    st.success("✅ No se rechaza H₀: distribución uniforme (p > 0.05)")
                else:
                    st.warning("⚠️ Se rechaza H₀: distribución NO uniforme (p ≤ 0.05)")

            st.markdown("### 📊 Distribución Poisson de Frecuencias")
            freq_values = list(freq.values())
            mean_freq = np.mean(freq_values)
            st.write(f"Frecuencia media por número: **{mean_freq:.1f}**")

            # Comparar distribución observada vs Poisson
            x_range = range(min(freq_values), max(freq_values) + 1)
            observed_dist = Counter(freq_values)
            poisson_expected = [poisson.pmf(x, mean_freq) * 50 for x in x_range]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(x_range),
                                 y=[observed_dist.get(x, 0) for x in x_range],
                                 name="Observado"))
            fig.add_trace(go.Scatter(x=list(x_range), y=poisson_expected,
                                     name="Poisson esperado", mode="lines+markers"))
            fig.update_layout(title="Distribución de frecuencias vs Poisson",
                              template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 📈 Autocorrelación Temporal")
            # Calcular serie temporal de sumas
            sums = calc_sum_stats(df)
            autocorr = calc_autocorrelation(sums, max_lag=20)
            fig = px.bar(x=list(range(1, 21)), y=autocorr,
                         labels={"x": "Lag (sorteos)", "y": "Autocorrelación"},
                         title="Autocorrelación de la suma de números")
            fig.add_hline(y=1.96 / np.sqrt(len(sums)), line_dash="dash", line_color="red",
                          annotation_text="IC 95%")
            fig.add_hline(y=-1.96 / np.sqrt(len(sums)), line_dash="dash", line_color="red")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        elif stat_sub == "Tendencias":
            st.markdown("### 📈 Tendencias Temporales")

            # Suma por sorteo en el tiempo
            df_plot = df.copy()
            df_plot["Suma"] = df_plot.apply(lambda r: sum(int(r[f"Num{i}"]) for i in range(1, 6)), axis=1)
            df_plot["Media_Movil_20"] = df_plot["Suma"].rolling(20).mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_plot["Fecha"], y=df_plot["Suma"],
                                     mode="markers", name="Suma por sorteo",
                                     marker=dict(size=3, opacity=0.3)))
            fig.add_trace(go.Scatter(x=df_plot["Fecha"], y=df_plot["Media_Movil_20"],
                                     mode="lines", name="Media móvil (20)",
                                     line=dict(width=2, color="#f5c842")))
            fig.update_layout(title="Evolución de la suma de números",
                              template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Frecuencia acumulada de un número específico
            num_sel = st.selectbox("Selecciona un número para ver su tendencia:", range(1, 51), index=0)
            cumul = []
            count = 0
            for _, row in df.iterrows():
                nums = [int(row[f"Num{i}"]) for i in range(1, 6)]
                if num_sel in nums:
                    count += 1
                cumul.append(count)

            fig = px.line(x=df["Fecha"], y=cumul,
                          labels={"x": "Fecha", "y": "Apariciones acumuladas"},
                          title=f"Apariciones acumuladas del número {num_sel}")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════════
    # TAB 2: SETS Y APUESTAS
    # ════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("## 🎯 Generación de Sets y Apuestas")

        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("🎰 Generar 10 Sets de 21 Números", use_container_width=True, type="primary"):
                sets = generate_10_sets(df, n_recent=n_recent)
                st.session_state.generated_sets = sets

        with col2:
            n_bets_per_set = st.number_input("Apuestas por set:", 1, 10, 3)

        if "generated_sets" in st.session_state:
            sets = st.session_state.generated_sets

            for i, (name, set_nums) in enumerate(sets):
                with st.expander(f"📦 Set {i + 1}: {name} → {set_nums}", expanded=(i == 0)):
                    st.write(f"**Números ({len(set_nums)}):** {set_nums}")

                    # Equilibrio del set
                    n_even = sum(1 for n in set_nums if n % 2 == 0)
                    n_low = sum(1 for n in set_nums if n <= 25)
                    decades = Counter((n - 1) // 10 for n in set_nums)
                    st.write(f"Pares: {n_even} | Impares: {21 - n_even} | "
                             f"Bajos: {n_low} | Altos: {21 - n_low}")
                    st.write(f"Por década: {dict(decades)}")

                    # Generar apuestas
                    bets = generate_bets_from_set(set_nums, df, n_bets=n_bets_per_set, n_recent=n_recent)
                    st.markdown("**Apuestas generadas:**")
                    for j, bet in enumerate(bets):
                        balls_html = render_balls(bet["nums"], bet["stars"])
                        st.markdown(f"Apuesta {j + 1}: {balls_html}", unsafe_allow_html=True)

                    # Guardar para backtesting
                    if st.button(f"💾 Usar en Backtesting", key=f"bt_set_{i}"):
                        st.session_state.backtest_bets = bets
                        st.success("✅ Apuestas guardadas para backtesting (pestaña Backtesting)")

            # Sistema reducido
            st.markdown("---")
            st.markdown("### 📐 Sistema Reducido")
            st.write("Genera combinaciones con garantía de cobertura sobre un set de 21 números.")

            set_idx = st.selectbox("Selecciona set:", range(len(sets)),
                                   format_func=lambda i: f"Set {i + 1}: {sets[i][0]}")
            guarantee = st.selectbox("Garantía:", ["3si5", "4si5"],
                                     format_func=lambda g: "3 aciertos si 5 están en set" if g == "3si5"
                                     else "4 aciertos si 5 están en set")

            if st.button("🔢 Calcular Sistema Reducido"):
                with st.spinner("Calculando cobertura... (puede tardar)"):
                    _, sel_nums = sets[set_idx]
                    combos = calc_reduced_system(sel_nums, guarantee=guarantee)
                    st.write(f"**Apuestas necesarias:** {len(combos)}")
                    st.write(f"**Coste total:** {len(combos) * TICKET_COST:.2f} €")
                    combos_df = pd.DataFrame(combos, columns=["N1", "N2", "N3", "N4", "N5"])
                    st.dataframe(combos_df.head(50), hide_index=True, use_container_width=True)
                    if len(combos) > 50:
                        st.info(f"Mostrando primeras 50 de {len(combos)} combinaciones.")

    # ════════════════════════════════════════════════════════════
    # TAB 3: ALGORITMO GENÉTICO
    # ════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("## 🧬 Optimizador con Algoritmo Genético")
        st.markdown("""
        Evoluciona combinaciones de 5+2 maximizando un fitness que combina:
        frecuencia histórica (30%), coincidencias con sorteos pasados (40%),
        equilibrio pares/impares y bajos/altos (20%), y cobertura de décadas (10%).
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            ga_pop = st.number_input("Población:", 20, 200, 50)
        with col2:
            ga_gen = st.number_input("Generaciones:", 10, 100, 30)
        with col3:
            ga_mut = st.slider("Prob. Mutación:", 0.01, 0.3, 0.1)

        if st.button("🧬 Evolucionar!", type="primary", use_container_width=True):
            progress = st.progress(0, "Iniciando evolución...")
            with st.spinner("Ejecutando algoritmo genético..."):
                results, fitness_df = run_genetic_algorithm(
                    df, pop_size=ga_pop, n_gen=ga_gen, mut_prob=ga_mut,
                    n_recent=n_recent, progress_bar=progress,
                )

            if results:
                st.markdown("### 🏆 Top 10 Combinaciones Optimizadas")
                for i, r in enumerate(results):
                    balls_html = render_balls(r["nums"], r["stars"])
                    st.markdown(f"**#{i + 1}** (fitness: {r['fitness']:.4f}): {balls_html}",
                                unsafe_allow_html=True)

                # Gráfico de evolución
                st.markdown("### 📈 Evolución del Fitness")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fitness_df["gen"], y=fitness_df["max"],
                                         mode="lines", name="Máximo"))
                fig.add_trace(go.Scatter(x=fitness_df["gen"], y=fitness_df["avg"],
                                         mode="lines", name="Promedio"))
                fig.update_layout(title="Fitness por generación",
                                  xaxis_title="Generación", yaxis_title="Fitness",
                                  template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

                # Guardar para backtesting
                ga_bets = [{"nums": r["nums"], "stars": r["stars"]} for r in results[:5]]
                if st.button("💾 Usar top 5 en Backtesting"):
                    st.session_state.backtest_bets = ga_bets
                    st.success("✅ Guardadas para backtesting")

    # ════════════════════════════════════════════════════════════
    # TAB 4: ML PREDICTOR
    # ════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("## 🤖 Predictor Machine Learning")
        st.markdown("""
        Modelos entrenados con datos históricos para identificar patrones estadísticos.
        **Recuerda:** Los sorteos son aleatorios; estos modelos buscan optimizar, no predecir con certeza.
        """)

        ml_model = st.radio(
            "Modelo:", ["Random Forest", "XGBoost", "K-Means Clustering"],
            horizontal=True,
        )

        ml_window = st.slider("Ventana temporal (sorteos):", 5, 30, 10)

        if st.button("🧠 Entrenar y Predecir", type="primary", use_container_width=True):
            with st.spinner(f"Entrenando {ml_model}..."):
                if ml_model == "Random Forest":
                    probas, score = train_random_forest(df, window=ml_window, n_recent=min(300, len(df)))

                    if probas is not None:
                        st.metric("Accuracy (multi-output)", f"{score:.4f}")

                        # Top 5 números más probables
                        num_probas = list(zip(range(1, 51), probas))
                        num_probas.sort(key=lambda x: x[1], reverse=True)

                        st.markdown("### 🎯 Top 10 Números Más Probables")
                        top_df = pd.DataFrame(num_probas[:10], columns=["Número", "Probabilidad"])
                        st.dataframe(top_df, hide_index=True)

                        suggested_nums = [n for n, _ in num_probas[:5]]
                        suggested_stars = generate_stars(df, n=2, n_recent=n_recent)
                        st.markdown("### Apuesta Sugerida:")
                        st.markdown(render_balls(sorted(suggested_nums), suggested_stars),
                                    unsafe_allow_html=True)

                        # Gráfico de probabilidades
                        fig = px.bar(x=[n for n, _ in num_probas],
                                     y=[p for _, p in num_probas],
                                     labels={"x": "Número", "y": "Probabilidad"},
                                     title="Probabilidad por número (Random Forest)")
                        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                          plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ Datos insuficientes para entrenar.")

                elif ml_model == "XGBoost":
                    probas, score = train_xgboost(df, window=ml_window, n_recent=min(300, len(df)))

                    if probas is not None:
                        st.metric("Accuracy (multi-output)", f"{score:.4f}")

                        num_probas = list(zip(range(1, 51), probas))
                        num_probas.sort(key=lambda x: x[1], reverse=True)

                        st.markdown("### 🎯 Top 10 Números Más Probables")
                        top_df = pd.DataFrame(num_probas[:10], columns=["Número", "Probabilidad"])
                        st.dataframe(top_df, hide_index=True)

                        suggested_nums = [n for n, _ in num_probas[:5]]
                        suggested_stars = generate_stars(df, n=2, n_recent=n_recent)
                        st.markdown("### Apuesta Sugerida:")
                        st.markdown(render_balls(sorted(suggested_nums), suggested_stars),
                                    unsafe_allow_html=True)

                        fig = px.bar(x=[n for n, _ in num_probas],
                                     y=[p for _, p in num_probas],
                                     labels={"x": "Número", "y": "Probabilidad"},
                                     title="Probabilidad por número (XGBoost)")
                        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                          plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ Datos insuficientes o XGBoost no instalado.")

                elif ml_model == "K-Means Clustering":
                    n_clusters = st.slider("Número de clusters:", 3, 10, 5, key="kmeans_k")
                    cluster_results = train_clustering(df, n_clusters=n_clusters, n_recent=min(300, len(df)))

                    if cluster_results:
                        st.write(f"**Cluster caliente (más frecuente reciente):** #{cluster_results['hot_cluster']}")
                        st.write(f"**Tamaños de clusters:** {cluster_results['cluster_sizes']}")

                        st.markdown("### Apuesta Sugerida (centroide del cluster caliente):")
                        st.markdown(
                            render_balls(cluster_results["suggested_nums"],
                                         cluster_results["suggested_stars"]),
                            unsafe_allow_html=True,
                        )

                        # Visualización de clusters
                        from sklearn.decomposition import PCA
                        subset = df.tail(min(300, len(df)))
                        features = []
                        for _, row in subset.iterrows():
                            nums = [int(row[f"Num{i}"]) for i in range(1, 6)]
                            stars = [int(row["Estrella1"]), int(row["Estrella2"])]
                            features.append(nums + stars)
                        X = np.array(features)
                        pca = PCA(n_components=2)
                        X_2d = pca.fit_transform(X)

                        fig = px.scatter(x=X_2d[:, 0], y=X_2d[:, 1],
                                         color=[str(l) for l in cluster_results["labels"]],
                                         labels={"x": "PC1", "y": "PC2", "color": "Cluster"},
                                         title="Clusters de sorteos (PCA 2D)")
                        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                          plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════════
    # TAB 5: BACKTESTING
    # ════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown("## 📈 Backtesting Retrospectivo")
        st.markdown("Comprueba cómo habrían rendido las apuestas en sorteos pasados.")

        n_bt_draws = st.slider("Nº sorteos para backtest:", 5, min(100, len(df) - 10), 20)

        # Opción para introducir apuesta manual
        st.markdown("### Apuesta a testear")
        bet_source = st.radio("Fuente:", ["Manual", "Desde Sets/GA"], horizontal=True)

        bets_to_test = []
        if bet_source == "Manual":
            col1, col2 = st.columns(2)
            with col1:
                manual_nums = st.text_input("Números (5, separados por coma):", "7,15,23,35,44")
            with col2:
                manual_stars = st.text_input("Estrellas (2, separadas por coma):", "3,9")
            try:
                nums = sorted([int(n.strip()) for n in manual_nums.split(",")])
                stars = sorted([int(s.strip()) for s in manual_stars.split(",")])
                if len(nums) == 5 and len(stars) == 2:
                    bets_to_test = [{"nums": nums, "stars": stars}]
                    st.markdown("Apuesta: " + render_balls(nums, stars), unsafe_allow_html=True)
            except Exception:
                st.warning("Formato inválido. Usa números separados por comas.")
        else:
            if "backtest_bets" in st.session_state:
                bets_to_test = st.session_state.backtest_bets
                st.write(f"**{len(bets_to_test)} apuestas** cargadas desde Sets/GA")
                for i, b in enumerate(bets_to_test):
                    st.markdown(f"#{i + 1}: " + render_balls(b["nums"], b["stars"]),
                                unsafe_allow_html=True)
            else:
                st.info("No hay apuestas guardadas. Genera apuestas en Sets o GA primero.")

        if bets_to_test and st.button("▶️ Ejecutar Backtesting", type="primary", use_container_width=True):
            with st.spinner("Analizando sorteos históricos..."):
                results_df, summary = backtest_bets(bets_to_test, df, n_draws=n_bt_draws)

            # Resumen
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tickets", f"{summary['total_tickets']:,}")
            with col2:
                st.metric("Coste Total", f"{summary['total_cost']:,.2f} €")
            with col3:
                color = "normal" if summary["net_balance"] >= 0 else "inverse"
                st.metric("Balance Neto", f"{summary['net_balance']:,.2f} €", delta_color=color)
            with col4:
                st.metric("ROI", f"{summary['roi_pct']:.2f}%")

            # Premios
            if summary["prize_counts"]:
                st.markdown("### 🏆 Premios Obtenidos")
                prizes_df = pd.DataFrame(list(summary["prize_counts"].items()),
                                         columns=["Categoría", "Cantidad"])
                st.dataframe(prizes_df, hide_index=True, use_container_width=True)
            else:
                st.info("No se obtuvieron premios en el período seleccionado.")

            # Detalle
            with st.expander("📋 Detalle completo"):
                st.dataframe(results_df, use_container_width=True, hide_index=True)

            # Comparación vs aleatorio
            st.markdown("### 🎲 Comparación vs. Apuestas Aleatorias")
            random_bets = [{"nums": sorted(random.sample(range(1, 51), 5)),
                            "stars": sorted(random.sample(range(1, 13), 2))}
                           for _ in range(len(bets_to_test))]
            _, random_summary = backtest_bets(random_bets, df, n_draws=n_bt_draws)

            comp_df = pd.DataFrame({
                "Métrica": ["Coste", "Premios", "Balance", "ROI"],
                "Tu Estrategia": [
                    f"{summary['total_cost']:.2f}€",
                    f"{summary['total_prizes']:.2f}€",
                    f"{summary['net_balance']:.2f}€",
                    f"{summary['roi_pct']:.2f}%",
                ],
                "Aleatorio": [
                    f"{random_summary['total_cost']:.2f}€",
                    f"{random_summary['total_prizes']:.2f}€",
                    f"{random_summary['net_balance']:.2f}€",
                    f"{random_summary['roi_pct']:.2f}%",
                ],
            })
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # ════════════════════════════════════════════════════════════
    # TAB 6: SIMULACIONES
    # ════════════════════════════════════════════════════════════
    with tabs[6]:
        st.markdown("## 🎲 Simulaciones Monte Carlo")
        st.markdown("Simula miles de sorteos aleatorios para calcular probabilidades empíricas.")

        col1, col2 = st.columns(2)
        with col1:
            mc_nums = st.text_input("Números para simular:", "7,15,23,35,44", key="mc_nums")
        with col2:
            mc_stars = st.text_input("Estrellas para simular:", "3,9", key="mc_stars")

        n_iters = st.select_slider("Iteraciones:", options=[1000, 5000, 10000, 50000, 100000], value=10000)

        try:
            sim_nums = sorted([int(n.strip()) for n in mc_nums.split(",")])
            sim_stars = sorted([int(s.strip()) for s in mc_stars.split(",")])
            valid_input = len(sim_nums) == 5 and len(sim_stars) == 2
        except Exception:
            valid_input = False

        if valid_input and st.button("🎲 Ejecutar Simulación", type="primary", use_container_width=True):
            progress = st.progress(0, "Simulando...")
            with st.spinner(f"Simulando {n_iters:,} sorteos..."):
                results, total_ev, match_dist = monte_carlo_simulation(
                    sim_nums, sim_stars, n_iterations=n_iters, progress_bar=progress,
                )

            st.markdown(f"### Resultados de {n_iters:,} simulaciones")
            st.markdown(f"**Valor Esperado por ticket:** {total_ev:.4f} € (coste: {TICKET_COST} €)")
            st.markdown(f"**Retorno teórico:** {total_ev / TICKET_COST * 100:.2f}%")

            # Tabla de probabilidades
            rows = []
            for name, data in sorted(results.items(), key=lambda x: x[1]["prize"], reverse=True):
                rows.append({
                    "Categoría": name,
                    "Aciertos": data["count"],
                    "Probabilidad": f"{data['prob']:.6f}",
                    "Odds": data["odds"],
                    "Premio €": f"{data['prize']:,}",
                    "EV €": f"{data['expected_value']:.4f}",
                })
            mc_df = pd.DataFrame(rows)
            st.dataframe(mc_df, hide_index=True, use_container_width=True)

            # Distribución de coincidencias
            st.markdown("### Distribución de Coincidencias")
            match_data = []
            for (n, s), count in sorted(match_dist.items()):
                match_data.append({"Nums": n, "Stars": s, "Frecuencia": count,
                                   "Prob%": count / n_iters * 100})
            match_df = pd.DataFrame(match_data)

            fig = px.scatter(match_df, x="Nums", y="Stars", size="Frecuencia",
                             color="Prob%", title="Distribución de coincidencias",
                             labels={"Nums": "Números acertados", "Stars": "Estrellas acertadas"},
                             color_continuous_scale="YlOrRd")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        elif not valid_input:
            st.warning("Introduce 5 números (1-50) y 2 estrellas (1-12) separados por comas.")

    # ── Footer ──
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        🎰 EuroMillones Analyzer Pro v1.0 | Experimento matemático y educativo<br>
        ⚠️ Los sorteos son aleatorios. Esta app NO garantiza predicciones ni fomenta el juego.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
