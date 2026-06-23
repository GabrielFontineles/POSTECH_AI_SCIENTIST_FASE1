"""
utils.py
--------
Funções auxiliares reutilizáveis para o projeto NPS E-commerce.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Constantes de negócio ───────────────────────────────────────────────────

NPS_BINS = {
    "Detrator": (0, 6.9),
    "Neutro": (7, 8.9),
    "Promotor": (9, 10),
}

PALETTE = {
    "Detrator": "#E74C3C",
    "Neutro": "#F39C12",
    "Promotor": "#2ECC71",
}

FIGSIZE_DEFAULT = (12, 6)
FIGSIZE_SQUARE = (8, 8)

# ── Caminhos padrão ─────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "reports" / "figures"


# ── Funções de carregamento ─────────────────────────────────────────────────

def load_raw(filename: str = "desafio_nps_fase_1.csv") -> pd.DataFrame:
    """Carrega o CSV bruto e retorna um DataFrame."""
    path = DATA_RAW / filename
    df = pd.read_csv(path)
    print(f"[load_raw] {df.shape[0]:,} linhas × {df.shape[1]} colunas carregadas de '{path.name}'")
    return df


# ── Feature engineering ─────────────────────────────────────────────────────

def classify_nps(score: float) -> str:
    """Classifica um nps_score como Detrator, Neutro ou Promotor."""
    if score <= 6:
        return "Detrator"
    elif score <= 8:
        return "Neutro"
    else:
        return "Promotor"


def add_nps_category(df: pd.DataFrame, col: str = "nps_score") -> pd.DataFrame:
    """Adiciona coluna 'nps_category' ao DataFrame."""
    df = df.copy()
    df["nps_category"] = df[col].apply(classify_nps)
    return df


def compute_nps(df: pd.DataFrame, col: str = "nps_category") -> float:
    """
    Calcula o NPS agregado:
        NPS = %Promotores − %Detratores
    """
    counts = df[col].value_counts(normalize=True) * 100
    promotores = counts.get("Promotor", 0)
    detratores = counts.get("Detrator", 0)
    return round(promotores - detratores, 2)


def remove_leakage_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove colunas que causariam data leakage ao modelar nps_score.
    Essas variáveis só existem APÓS o evento de compra/avaliação.

    Atenção: ajuste esta lista conforme o entendimento do negócio.
    """
    leakage_cols = [
        "nps_score",       # próprio target
        "csat_internal_score",  # coletado junto ao NPS
        "repeat_purchase_30d",  # ocorre após a compra
    ]
    cols_to_drop = [c for c in leakage_cols if c in df.columns]
    print(f"[remove_leakage_cols] Removidas: {cols_to_drop}")
    return df.drop(columns=cols_to_drop)


# ── Visualização ────────────────────────────────────────────────────────────

def set_style() -> None:
    """Aplica estilo visual padrão ao projeto."""
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
    })


def save_fig(fig: plt.Figure, filename: str, tight: bool = True) -> None:
    """Salva figura em reports/figures/."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / filename
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"[save_fig] Salvo em '{path}'")


def plot_nps_distribution(df: pd.DataFrame, col: str = "nps_score") -> plt.Figure:
    """Histograma da distribuição do NPS Score."""
    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    ax.hist(df[col], bins=30, color="#3498DB", edgecolor="white", alpha=0.85)
    ax.axvline(6, color=PALETTE["Detrator"], linestyle="--", label="Corte Detrator (≤6)")
    ax.axvline(8, color=PALETTE["Neutro"], linestyle="--", label="Corte Neutro (≤8)")
    ax.set_title("Distribuição do NPS Score")
    ax.set_xlabel("NPS Score")
    ax.set_ylabel("Frequência")
    ax.legend()
    return fig


def plot_category_share(df: pd.DataFrame, col: str = "nps_category") -> plt.Figure:
    """Pizza com proporção de Detratores, Neutros e Promotores."""
    counts = df[col].value_counts()
    colors = [PALETTE[c] for c in counts.index]
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    ax.pie(
        counts,
        labels=counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    ax.set_title("Proporção de Categorias NPS")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap de correlação das variáveis numéricas."""
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Matriz de Correlação")
    return fig


def plot_delay_vs_nps(df: pd.DataFrame) -> plt.Figure:
    """Média do NPS por dias de atraso na entrega."""
    grouped = df.groupby("delivery_delay_days")["nps_score"].mean().reset_index()
    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    ax.plot(
        grouped["delivery_delay_days"],
        grouped["nps_score"],
        marker="o",
        color="#E74C3C",
        linewidth=2,
    )
    ax.set_title("Média do NPS Score por Dias de Atraso")
    ax.set_xlabel("Dias de Atraso (delivery_delay_days)")
    ax.set_ylabel("NPS Score Médio")
    ax.grid(True, alpha=0.4)
    return fig


# ── Estatísticas descritivas ────────────────────────────────────────────────

def nps_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Retorna média do NPS Score e NPS calculado por grupo.

    Exemplo:
        nps_by_group(df, 'customer_region')
    """
    df = add_nps_category(df)
    result = (
        df.groupby(group_col)
        .apply(lambda g: pd.Series({
            "nps_score_medio": round(g["nps_score"].mean(), 2),
            "nps_calculado": compute_nps(g),
            "n": len(g),
        }))
        .reset_index()
        .sort_values("nps_calculado", ascending=False)
    )
    return result
