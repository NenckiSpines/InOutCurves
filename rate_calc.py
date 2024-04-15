class RateCalculator:
    def __init__(self):
        self.results = {}

    def calculate_rates(self, p_values_no_diff, p_values_diff, alphas=(0.05, 0.01, 0.001)):
        for alpha in alphas:
            fpr_no_diff = sum(p <= alpha for p in p_values_no_diff) / len(p_values_no_diff)
            fnr_diff = sum(p > alpha for p in p_values_diff) / len(p_values_diff)
            tpr_diff = 1 - fnr_diff
            tnr_no_diff = 1 - fpr_no_diff
            precision_diff = tpr_diff / (tpr_diff + fpr_no_diff)
            specificity_no_diff = tnr_no_diff
            sensitivity_diff = tpr_diff
            accuracy_no_diff = (tnr_no_diff + tpr_diff) / 2
            accuracy_diff = (precision_diff + sensitivity_diff) / 2

            self.results[alpha] = {
                "FPR": fpr_no_diff,
                "FNR": fnr_diff,
                "Precision": precision_diff,
                "Specificity": specificity_no_diff,
                "Sensitivity": sensitivity_diff,
                "Accuracy (No Diff)": accuracy_no_diff,
                "Accuracy (Diff)": accuracy_diff
            }

    def print_rates(self):
        for alpha, result in self.results.items():
            print(f"Significance level: {alpha}")
            print("False Positive Rate (FPR): {:.4f}".format(result.get("FPR", 0)))
            print("False Negative Rate (FNR): {:.4f}".format(result.get("FNR", 0)))
            print("Precision: {:.4f}".format(result.get("Precision", 0)))
            print("Specificity: {:.4f}".format(result.get("Specificity", 0)))
            print("Sensitivity: {:.4f}".format(result.get("Sensitivity", 0)))
            print("Accuracy (No Diff): {:.4f}".format(result.get("Accuracy (No Diff)", 0)))
            print("Accuracy (Diff): {:.4f}".format(result.get("Accuracy (Diff)", 0)))
            print()

    def latex_table(self, alphas=(0.05, 0.01, 0.001)):
        table = """
\\begin{table}[h]
\\centering
\\begin{tabular}{|l|c|c|c|}
\\hline
\\textbf{Metric} & \\textbf{Alpha=%g} & \\textbf{Alpha=%g} & \\textbf{Alpha=%g} \\\\
\\hline
""" % alphas

        metric_names = [
            "FPR",
            "FNR",
            "Precision",
            "Specificity",
            "Sensitivity",
            "Accuracy (No Diff)",
            "Accuracy (Diff)"
        ]

        for metric_name in metric_names:
            table += metric_name
            for alpha in alphas:
                table += " & {:.4f}".format(self.results.get(alpha, {}).get(metric_name, 0))
            table += " \\\\\n"

        table += "\\hline\n"
        table += "\\end{tabular}\n"
        table += "\\caption{Rates calculated for two groups}\n"
        table += "\\label{tab:rates}\n"
        table += "\\end{table}\n"

        return table




if __name__ == "__main__":
    # Example usage:
    p_values_no_diff = [0.01, 0.03, 0.06, 0.02, 0.04]  # List of p-values for comparison of two groups that do not differ
    p_values_diff = [0.001, 0.006, 0.08, 0.15, 0.02]    # List of p-values for comparison of two groups that do differ

    rc = RateCalculator()
    rc.calculate_rates(p_values_no_diff, p_values_diff)
    rc.print_rates()
