import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway, norm

# Load the two Excel files
df1 = pd.read_excel("/mnt/data/deliveries.xlsx")
df2 = pd.read_excel("/mnt/data/matches (1) - Google Sheets.xlsx")

# -----------------------------
# Select Numeric Columns
# -----------------------------

# Example: deliveries total_runs per ball
x = df1["total_runs"].dropna()

# Example: matches win_by_runs per match
y = df2["win_by_runs"].dropna()

# -----------------------------
# 1️⃣ TWO-SAMPLE T-TEST
# -----------------------------
t_stat, t_pvalue = ttest_ind(x, y, equal_var=False)  # Welch t-test
print("T-TEST RESULTS:")
print("t-statistic:", t_stat)
print("p-value:", t_pvalue)
print("------------------------------------")

# -----------------------------
# 2️⃣ Z-TEST (Two-sample)
# -----------------------------
mean_x = np.mean(x)
mean_y = np.mean(y)

std_x = np.std(x, ddof=1)
std_y = np.std(y, ddof=1)

n1 = len(x)
n2 = len(y)

# Standard error
se = np.sqrt(std_x**2 / n1 + std_y**2 / n2)

# Z-statistic
z_stat = (mean_x - mean_y) / se
z_pvalue = 2 * (1 - norm.cdf(abs(z_stat)))

print("Z-TEST RESULTS:")
print("z-statistic:", z_stat)
print("p-value:", z_pvalue)
print("------------------------------------")

# -----------------------------
# 3️⃣ F-TEST (Variance Ratio Test)
# -----------------------------
var_x = np.var(x, ddof=1)
var_y = np.var(y, ddof=1)

f_stat = var_x / var_y

print("F-TEST RESULTS:")
print("F-statistic (var_x / var_y):", f_stat)
print("Note: p-values require scipy.stats.f distribution if needed.")
print("------------------------------------")
