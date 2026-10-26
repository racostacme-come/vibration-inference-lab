"""Generate LaTeX evidence tables directly from the measured integration report."""
def write_table(data, path):
    rows = data.get("rows")
    lines = [r"\begin{center}\small"]
    if data["experiment"] == "common-cantilever":
        lines += [r"\begin{tabular}{rrrrr}\toprule",
                  r"Elements & Load rel. error & Reduced rel. error & Full rel. error & Product\\\midrule"]
        for r in rows:
            lines.append(f'{r["elements"]} & {r["buckling_relative_error"]:.3e} & {r["reduced_compliance_relative_error"]:.3e} & {r["full_compliance_relative_error"]:.3e} & {r["dimensionless_product"]:.6f}' + r"\\")
        lines += [r"\bottomrule\end{tabular}",
                  f'\\par Reference product: {data["product_reference"]:.9f}.']
    elif rows is not None:
        lines += [r"\begin{tabular}{rlrrr}\toprule",
                  r"Steps & Method & Mass-norm error & Heat defect & Minimum $T$\\\midrule"]
        for r in rows:
            lines.append(f'{r["steps"]} & {r["method"]} & {r["mass_weighted_error"]:.3e} & {r["heat_balance_error"]:.3e} & {r["minimum_temperature"]:.4f}' + r"\\")
        lines += [r"\bottomrule\end{tabular}",
                  f'\\par Modal-reference maximum difference: {data["modal_reference_max_difference"]:.3e}.']
    else:
        lines += [r"\begin{tabular}{lr}\toprule", r"Diagnostic & Value\\\midrule"]
        labels = {"transition_max_difference":"Transition maximum difference",
                  "observed_measurement_rmse":"Observed measurement RMSE (m)",
                  "outage_filtered_rmse":"Outage filtered RMSE (m)",
                  "outage_smoothed_rmse":"Outage smoothed RMSE (m)",
                  "max_smoothed_minus_filtered_variance":"Maximum smoothed minus filtered variance"}
        for key, label in labels.items():
            lines.append(f"{label} & {data[key]:.4e}"+r"\\")
        lines += [r"\bottomrule\end{tabular}"]
    if "ensemble" in data:
        e = data['ensemble']
        lines += [r'\par\medskip', r'\begin{tabular}{lrrr}\toprule',
                  r'Prior-matched estimator & MSE (m$^2$) & MC standard error & Expected MSE\\\midrule']
        for method in ('filtered', 'smoothed'):
            lines.append(f'{method.capitalize()} & {e[method+"_mse"]:.4e} & {e[method+"_mse_standard_error"]:.3e} & {e[method+"_expected_mse"]:.4e}'+r'\\')
        lines += [r'\bottomrule\end{tabular}',
                  f'\\par Paired improvement standard error: {e["paired_improvement_standard_error"]:.3e} m$^2$.']
    lines += [r"\end{center}"]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8", newline="\n")
