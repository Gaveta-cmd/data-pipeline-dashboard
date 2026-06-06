"""
Testes de validacao do data_analyzer.
Valida limpeza, estatisticas e persistencia sem dependencias externas de test runner.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data_analyzer import load_data, clean_data, analyze_data, save_data

PASS = "PASS"
FAIL = "FAIL"
_results: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    _results.append((name, condition, detail))
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_dirty_df() -> pd.DataFrame:
    """Cria um DataFrame sintetico com todos os problemas tipicos."""
    return pd.DataFrame({
        "nome":    ["Alice", "Bob", "Alice", None,    "Eve",  "Eve"],
        "idade":   [30,      25,    30,      None,    22,     22],
        "salario": [4500.0,  3200,  4500.0,  None,    2100,   2100],
        "nota":    [8.5,     7.0,   8.5,     None,    6.5,    6.5],
    })


# ---------------------------------------------------------------------------
# Testes de limpeza
# ---------------------------------------------------------------------------

def test_remove_duplicates():
    df, report = clean_data(make_dirty_df())
    check("Duplicatas removidas", report["duplicatas"] >= 2,
          f"encontradas: {report['duplicatas']}")
    check("Sem duplicatas no resultado", not df.duplicated().any())


def test_remove_empty_rows():
    df, report = clean_data(make_dirty_df())
    check("Linha totalmente vazia removida",
          report["linhas_completamente_vazias"] >= 1,
          f"removidas: {report['linhas_completamente_vazias']}")


def test_type_coercion():
    raw = pd.DataFrame({
        "valor": ["R$ 1.500", "2000", "abc", "3200"],
        "data":  ["2021-01-01", "15/06/2018", "nao-data", "2022-12-31"],
    })
    df, _ = clean_data(raw)
    check("Coluna numerica convertida", pd.api.types.is_numeric_dtype(df["valor"]),
          str(df["valor"].dtype))


def test_report_totals():
    df_in = make_dirty_df()
    df_out, report = clean_data(df_in)
    calculated = report["linhas_antes"] - report["linhas_depois"]
    check("total_removidas consistente",
          report["total_removidas"] == calculated,
          f"{report['total_removidas']} == {calculated}")


# ---------------------------------------------------------------------------
# Testes estatisticos
# ---------------------------------------------------------------------------

def test_statistics_accuracy():
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    stats = analyze_data(df)
    check("Media correta",    abs(stats["v"]["media"]    - 3.0)   < 1e-9, str(stats["v"]["media"]))
    check("Mediana correta",  abs(stats["v"]["mediana"]  - 3.0)   < 1e-9, str(stats["v"]["mediana"]))
    check("DP correto",       abs(stats["v"]["desvio_padrao"] - np.std([1,2,3,4,5], ddof=1)) < 1e-4)
    check("Minimo correto",   stats["v"]["minimo"] == 1.0)
    check("Maximo correto",   stats["v"]["maximo"] == 5.0)


def test_outlier_detection():
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]})
    stats = analyze_data(df)
    check("Outlier detectado", stats["v"]["outliers"] >= 1,
          f"outliers={stats['v']['outliers']}")


def test_no_numeric_columns():
    df = pd.DataFrame({"nome": ["A", "B"], "cidade": ["SP", "RJ"]})
    stats = analyze_data(df)
    check("Retorna vazio sem numericos", stats == {})


# ---------------------------------------------------------------------------
# Testes de persistencia
# ---------------------------------------------------------------------------

def test_save_and_reload(tmp_path="output/_test_save.csv"):
    df_orig = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    save_data(df_orig, tmp_path)
    df_loaded = pd.read_csv(tmp_path, encoding="utf-8-sig")
    check("Arquivo salvo existe", os.path.exists(tmp_path))
    check("Dimensoes preservadas", df_loaded.shape == df_orig.shape,
          str(df_loaded.shape))
    try:
        os.remove(tmp_path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Teste de ponta a ponta com o CSV real
# ---------------------------------------------------------------------------

def test_salary_cap():
    from data_analyzer import _validate_ranges, SALARY_CAP
    df = pd.DataFrame({"salario": [3000.0, 5000.0, 999999.0, 4500.0]})
    df = _validate_ranges(df)
    check("Salario acima do cap vira NaN", pd.isna(df.loc[2, "salario"]),
          f"cap={SALARY_CAP}")
    check("Salarios validos preservados", df["salario"].notna().sum() == 3)


def test_nota_cap():
    from data_analyzer import _validate_ranges
    df = pd.DataFrame({"nota_avaliacao": [7.0, 10.5, 8.0, -1.0]})
    df = _validate_ranges(df)
    check("Nota acima de 10 vira NaN",  pd.isna(df.loc[1, "nota_avaliacao"]))
    check("Nota negativa vira NaN",     pd.isna(df.loc[3, "nota_avaliacao"]))
    check("Nota valida preservada",     df.loc[0, "nota_avaliacao"] == 7.0)


def test_end_to_end():
    csv = os.path.join(os.path.dirname(__file__), "sample_data.csv")
    if not os.path.exists(csv):
        check("E2E -- arquivo CSV encontrado", False, "sample_data.csv ausente")
        return

    df_raw = load_data(csv)
    df_clean, report = clean_data(df_raw.copy())
    stats = analyze_data(df_clean)

    check("E2E -- CSV carregado",        len(df_raw) > 0,  f"{len(df_raw)} linhas")
    check("E2E -- duplicatas removidas", report["duplicatas"] >= 2,
          f"{report['duplicatas']} dups")
    check("E2E -- linhas reduzidas",     len(df_clean) < len(df_raw),
          f"{len(df_raw)} -> {len(df_clean)}")
    check("E2E -- estatisticas geradas", bool(stats), str(list(stats.keys())[:3]))
    check("E2E -- dados limpos sem outliers extremos",
          all(s.get("outliers", 0) == 0 for s in stats.values() if isinstance(s, dict)),
          "apos correcao de ranges nao devem existir outliers")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    suites = [
        test_remove_duplicates,
        test_remove_empty_rows,
        test_type_coercion,
        test_report_totals,
        test_statistics_accuracy,
        test_outlier_detection,
        test_no_numeric_columns,
        test_salary_cap,
        test_nota_cap,
        test_save_and_reload,
        test_end_to_end,
    ]

    print("\n" + "=" * 50)
    print("  SUITE DE TESTES -- data_analyzer")
    print("=" * 50)

    for suite in suites:
        print(f"\n>> {suite.__name__}")
        try:
            suite()
        except Exception as exc:
            _results.append((suite.__name__, False, str(exc)))
            print(f"  [ERRO] {exc}")

    passed = sum(1 for _, ok, _ in _results if ok)
    total  = len(_results)
    failed = total - passed

    print("\n" + "=" * 50)
    print(f"  RESULTADO: {passed}/{total} passaram", end="")
    if failed:
        print(f"  |  {failed} FALHA(S)")
        for name, ok, detail in _results:
            if not ok:
                print(f"    x {name}" + (f": {detail}" if detail else ""))
    else:
        print("  -- Todos os testes passaram!")
    print("=" * 50 + "\n")

    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
