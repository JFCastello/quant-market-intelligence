from __future__ import annotations

import sys

from quant_platform.services import (
    build_symbol_model_comparison_summary,
    build_symbol_structural_break_summary,
    build_symbol_overview_snapshot,
    get_symbol_model_comparison_bundle,
    get_symbol_overview_bundle,
    get_symbol_structural_changes_bundle,
    list_available_symbols,
)


def main() -> None:
    """
    Ejecuta un smoke test de alto nivel sobre la services layer del proyecto.

    Propósito:
    Este script no valida en profundidad la lógica matemática ni el contenido
    exacto de cada artefacto. Su objetivo es más simple y muy importante:
    comprobar que las funciones públicas principales de la capa de servicios
    pueden ejecutarse para todos los símbolos disponibles sin romperse y que,
    además, devuelven estructuras con la forma mínima esperada.

    Qué verifica:
    1. Que la capa de servicios pueda descubrir símbolos.
    2. Que el overview por símbolo se pueda construir.
    3. Que la capa de model comparison funcione y devuelva bundles consistentes.
    4. Que la capa de structural changes funcione y devuelva bundles consistentes.
    5. Que algunos contratos básicos de salida se cumplan.

    Filosofía del script:
    Esto es un "smoke check", no una suite de tests exhaustiva.
    La idea es responder a la pregunta:
        "¿La capa de servicios está viva y usable?"
    """
    # Lista acumuladora de problemas encontrados durante la corrida.
    #
    # En vez de abortar al primer fallo, el script intenta recolectar tantos
    # issues como sea posible para entregar un diagnóstico más útil.
    issues: list[str] = []

    # Descubre los símbolos que la services layer dice tener disponibles.
    symbols = list_available_symbols()

    # Si no hay símbolos, eso ya es un problema de discovery.
    if not symbols:
        issues.append("[DISCOVERY] No symbols available from services layer.")
    else:
        # Si sí hay símbolos, los imprime para dejar trazabilidad en consola.
        print(f"symbols={symbols}")

    # Recorre cada símbolo y prueba las funciones principales de la services layer.
    for symbol in symbols:
        try:
            # Construye el snapshot compacto del overview.
            overview_snapshot = build_symbol_overview_snapshot(symbol)

            # Construye el bundle completo de overview, que debería incluir
            # snapshot + tablas subyacentes.
            overview_bundle = get_symbol_overview_bundle(symbol)

        except Exception as exc:
            # Si falla cualquier parte del bloque de overview, se registra
            # el error con contexto suficiente y se pasa al siguiente símbolo.
            issues.append(f"[OVERVIEW] {symbol}: {type(exc).__name__}: {exc}")
            continue

        try:
            # Construye el summary agregado de model comparison.
            comparison_summary_df = build_symbol_model_comparison_summary(symbol)

            # Construye el bundle completo de model comparison.
            comparison_bundle = get_symbol_model_comparison_bundle(symbol)

        except Exception as exc:
            # Si falla esta capa, se registra el issue y se pasa al siguiente símbolo.
            issues.append(f"[MODEL_COMPARISON] {symbol}: {type(exc).__name__}: {exc}")
            continue

        try:
            # Construye el resumen compacto de structural breaks / structural changes.
            structural_summary = build_symbol_structural_break_summary(symbol)

            # Construye el bundle completo asociado a structural changes.
            structural_bundle = get_symbol_structural_changes_bundle(symbol)

        except Exception as exc:
            # Si falla esta capa, también se registra y se continúa.
            issues.append(f"[STRUCTURAL_CHANGES] {symbol}: {type(exc).__name__}: {exc}")
            continue

        # ------------------------------------------------------------------
        # Validaciones ligeras del bloque OVERVIEW
        # ------------------------------------------------------------------

        # El símbolo reportado por el snapshot debería coincidir con el símbolo
        # que estamos procesando, normalizado en mayúsculas.
        if overview_snapshot["symbol"] != symbol.upper():
            issues.append(f"[OVERVIEW] {symbol}: snapshot symbol mismatch")

        # El bundle de overview debería contener timeseries_df y no debería venir vacío.
        if "timeseries_df" not in overview_bundle or overview_bundle["timeseries_df"].empty:
            issues.append(f"[OVERVIEW] {symbol}: missing or empty timeseries_df")

        # ------------------------------------------------------------------
        # Validaciones ligeras del bloque MODEL_COMPARISON
        # ------------------------------------------------------------------

        # El summary de model comparison no debería estar vacío si la capa
        # está funcionando correctamente para ese símbolo.
        if comparison_summary_df.empty:
            issues.append(f"[MODEL_COMPARISON] {symbol}: empty summary dataframe")

        # Se define explícitamente el conjunto de llaves esperadas en el bundle.
        # Esto comprueba no solo que el bundle exista, sino que su contrato
        # estructural sea exactamente el previsto.
        expected_comparison_bundle_keys = {"metrics_df", "pivot_df", "summary_df", "panel_df"}

        if set(comparison_bundle.keys()) != expected_comparison_bundle_keys:
            issues.append(
                f"[MODEL_COMPARISON] {symbol}: unexpected bundle keys {sorted(comparison_bundle.keys())}"
            )

        # ------------------------------------------------------------------
        # Validaciones ligeras del bloque STRUCTURAL_CHANGES
        # ------------------------------------------------------------------

        # El símbolo reportado por el summary estructural también debería
        # coincidir con el símbolo procesado en mayúsculas.
        if structural_summary["symbol"] != symbol.upper():
            issues.append(f"[STRUCTURAL_CHANGES] {symbol}: summary symbol mismatch")

        # Se verifica que el bundle estructural tenga exactamente las llaves esperadas.
        expected_structural_bundle_keys = {"summary", "signal_df", "events_df", "recent_events_df"}

        if set(structural_bundle.keys()) != expected_structural_bundle_keys:
            issues.append(
                f"[STRUCTURAL_CHANGES] {symbol}: unexpected bundle keys {sorted(structural_bundle.keys())}"
            )

        # Si el símbolo pasó por los tres bloques sin excepciones fatales,
        # imprime un resumen operativo útil:
        # - número de filas del overview
        # - número de filas del summary de model comparison
        # - número de eventos estructurales
        print(
            f"[OK] {symbol} | "
            f"overview_rows={len(overview_bundle['timeseries_df'])} | "
            f"comparison_rows={len(comparison_bundle['summary_df'])} | "
            f"break_events={structural_summary['event_count']}"
        )

    # Si se detectó al menos un problema, el smoke test falla:
    # - imprime FAIL
    # - lista todos los issues encontrados
    # - devuelve exit code 1
    #
    # Esto es especialmente útil si el script se usa en un runbook, pipeline
    # de validación o paso manual de sanity check.
    if issues:
        print("SERVICES LAYER SMOKE CHECKS: FAIL")
        for issue in issues:
            print(issue)
        sys.exit(1)

    # Si no hubo problemas, declara PASS.
    print("SERVICES LAYER SMOKE CHECKS: PASS")


if __name__ == "__main__":
    """
    Punto de entrada estándar del script.

    Significa:
    - si el archivo se ejecuta directamente, corre `main()`;
    - si se importa como módulo desde otro lugar, no ejecuta nada automáticamente.

    Esto permite reutilizar `main` o extender este archivo en el futuro sin
    disparar el smoke test al importarlo.
    """
    main()