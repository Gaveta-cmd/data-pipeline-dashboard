import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Limite maximo aceitavel para salario — valores acima disso sao tratados
# como erro de digitacao e definidos como nulos antes da analise estatistica.
SALARY_CAP = 50_000


def load_data(filepath: str) -> pd.DataFrame:
    """Carrega um CSV detectando separador automaticamente (virgula ou ponto-e-virgula)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo nao encontrado: {filepath}")

    for sep in (",", ";"):
        try:
            df = pd.read_csv(filepath, sep=sep, encoding="utf-8-sig")
            if df.shape[1] > 1:
                log.info("Arquivo carregado: %d linhas x %d colunas", *df.shape)
                return df
        except Exception:
            continue

    raise ValueError(f"Nao foi possivel ler o arquivo: {filepath}")


def clean_data(df: pd.DataFrame) -> tuple:
    """Remove linhas vazias, duplicatas e faz coercao de tipos.

    Retorna (DataFrame limpo, dict com contagens do que foi removido).
    """
    report: dict = {}
    n_original = len(df)
    log.info("Iniciando limpeza -- %d linhas", n_original)

    df = df.dropna(how="all")
    report["linhas_completamente_vazias"] = n_original - len(df)

    dupes = df.duplicated()
    report["duplicatas"] = int(dupes.sum())
    df = df[~dupes].reset_index(drop=True)

    df = _coerce_types(df)
    df = _validate_ranges(df)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        all_null = df[num_cols].isnull().all(axis=1)
        report["linhas_sem_numericos"] = int(all_null.sum())
        df = df[~all_null].reset_index(drop=True)
    else:
        report["linhas_sem_numericos"] = 0

    report["linhas_antes"] = n_original
    report["linhas_depois"] = len(df)
    report["total_removidas"] = n_original - len(df)

    log.info("Limpeza concluida -- %d linhas restantes", len(df))
    return df, report


def _validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica regras de negocio para invalidar valores fora de range esperado.

    Valores invalidos viram NaN — a linha nao e removida, apenas o campo.
    Isso preserva o registro e deixa claro que houve um problema naquele campo.
    """
    if "salario" in df.columns and pd.api.types.is_numeric_dtype(df["salario"]):
        invalid = df["salario"] > SALARY_CAP
        if invalid.any():
            log.warning(
                "%d salario(s) acima de %s substituidos por NaN",
                invalid.sum(), f"R$ {SALARY_CAP:,.0f}"
            )
            df.loc[invalid, "salario"] = np.nan

    if "nota_avaliacao" in df.columns and pd.api.types.is_numeric_dtype(df["nota_avaliacao"]):
        out_of_range = (df["nota_avaliacao"] < 0) | (df["nota_avaliacao"] > 10)
        if out_of_range.any():
            log.warning("%d nota(s) fora do range [0, 10] substituidas por NaN", out_of_range.sum())
            df.loc[out_of_range, "nota_avaliacao"] = np.nan

    return df


def _is_text_col(series: pd.Series) -> bool:
    return (
        not pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_datetime64_any_dtype(series)
    )


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Tenta converter colunas de texto para numerico ou datetime.

    Testa numerico primeiro para evitar que anos (ex: 4500) sejam
    interpretados como datas pelo dateutil.
    """
    for col in df.columns:
        if not _is_text_col(df[col]):
            continue

        series = df[col].astype(str).str.strip()
        n_valid = df[col].notna().sum()

        # --- numerico ---
        # Suporta formatos brasileiros: "R$ 1.234,56" e "1.234,56"
        cleaned = (
            series
            .str.replace(r"[R$\s%]", "", regex=True)
            .str.replace(r"\.(?=\d{3}(?:[,\.]|$))", "", regex=True)
            .str.replace(",", ".", regex=False)
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().sum() / max(n_valid, 1) > 0.5:
            if (numeric.dropna() % 1 == 0).all():
                df[col] = numeric.astype("Int64")
            else:
                df[col] = numeric.astype(float)
            log.info("Coluna '%s' -> numerico", col)
            continue

        # --- datetime ---
        # So tenta se os valores contem separadores de data (- / :)
        has_sep = series.str.contains(r"[\-/:]", regex=True)
        if has_sep.sum() / max(n_valid, 1) > 0.5:
            converted = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
            if converted.notna().sum() / max(n_valid, 1) > 0.5:
                df[col] = converted
                log.info("Coluna '%s' -> datetime", col)

    return df


def analyze_data(df: pd.DataFrame) -> dict:
    """Calcula estatisticas descritivas e detecta outliers via IQR (metodo Tukey)."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        log.warning("Nenhuma coluna numerica encontrada.")
        return {}

    stats: dict = {}
    outlier_mask = pd.Series(False, index=df.index)

    for col in num_cols:
        series = df[col].dropna().astype(float)
        if series.empty:
            continue

        arr = series.to_numpy()
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        col_out = df[col].notna() & ((df[col] < lower) | (df[col] > upper))
        outlier_mask |= col_out

        stats[col] = {
            "media":           round(float(np.mean(arr)), 4),
            "mediana":         round(float(np.median(arr)), 4),
            "desvio_padrao":   round(float(np.std(arr, ddof=1)), 4),
            "minimo":          round(float(np.min(arr)), 4),
            "maximo":          round(float(np.max(arr)), 4),
            "q1":              round(float(q1), 4),
            "q3":              round(float(q3), 4),
            "iqr":             round(float(iqr), 4),
            "limite_inferior": round(float(lower), 4),
            "limite_superior": round(float(upper), 4),
            "outliers":        int(col_out.sum()),
            "nulos":           int(df[col].isnull().sum()),
        }
        log.info("'%s': media=%.2f | DP=%.2f | outliers=%d",
                 col, stats[col]["media"], stats[col]["desvio_padrao"], stats[col]["outliers"])

    stats["_outlier_indices"] = df.index[outlier_mask].tolist()
    return stats


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """Salva o DataFrame processado em CSV com BOM UTF-8 (compativel com Excel)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    log.info("Salvo em: %s", output_path)


def print_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame,
                 clean_report: dict, stats: dict) -> None:
    sep  = "=" * 60
    dash = "-" * 60

    print("\n" + sep)
    print("  RELATORIO DE ANALISE")
    print(sep + "\n")

    print(f"{'LIMPEZA':^60}")
    print(dash)
    print(f"  Linhas brutas              : {clean_report['linhas_antes']:>8,}")
    print(f"  Linhas vazias              : {clean_report['linhas_completamente_vazias']:>8,}")
    print(f"  Duplicatas                 : {clean_report['duplicatas']:>8,}")
    print(f"  Sem campos numericos       : {clean_report['linhas_sem_numericos']:>8,}")
    print(f"  Total removidas            : {clean_report['total_removidas']:>8,}")
    print(f"  Linhas finais              : {clean_report['linhas_depois']:>8,}")
    print()

    print(f"{'TIPOS DE DADOS':^60}")
    print(dash)
    for col, dtype in df_clean.dtypes.items():
        nulos = int(df_clean[col].isnull().sum())
        print(f"  {col:<28} {str(dtype):<14} nulos: {nulos}")
    print()

    num_stats = {k: v for k, v in stats.items() if not k.startswith("_")}
    if num_stats:
        print(f"{'ESTATISTICAS':^60}")
        print(dash)
        for col, s in num_stats.items():
            print(f"\n  [{col}]")
            print(f"    Media         : {s['media']:>12,.4f}")
            print(f"    Mediana       : {s['mediana']:>12,.4f}")
            print(f"    Desvio padrao : {s['desvio_padrao']:>12,.4f}")
            print(f"    Min / Max     : {s['minimo']:>12,.4f} / {s['maximo']:,.4f}")
            print(f"    Q1 / Q3       : {s['q1']:>12,.4f} / {s['q3']:,.4f}")
            print(f"    Outliers      : {s['outliers']:>12,}")

    outlier_idx = stats.get("_outlier_indices", [])
    print(f"\n{'OUTLIERS':^60}")
    print(dash)
    print(f"  Linhas com outlier: {len(outlier_idx)}")
    if outlier_idx:
        print(df_clean.loc[outlier_idx[:5]].to_string(index=True))

    print(f"\n{'AMOSTRA (5 primeiras linhas)':^60}")
    print(dash)
    print(df_clean.head().to_string(index=False))
    print("\n" + sep + "\n")


if __name__ == "__main__":
    import sys

    input_file  = sys.argv[1] if len(sys.argv) > 1 else "sample_data.csv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output/dados_limpos.csv"

    df_raw = load_data(input_file)
    df_clean, clean_report = clean_data(df_raw.copy())
    stats = analyze_data(df_clean)
    save_data(df_clean, output_file)
    print_report(df_raw, df_clean, clean_report, stats)
