import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("/Users/maxkubicek/PycharmProjects/Bachelor/data/crime_clean.csv")

# =========================
# 2. PREPARE BETA DATA
# =========================
df_start = df[df["year"] == 2013][["country", "crime_rate"]].rename(columns={"crime_rate": "initial"})
df_end = df[df["year"] == 2023][["country", "crime_rate"]].rename(columns={"crime_rate": "final"})

df_beta = pd.merge(df_start, df_end, on="country")

df_beta["growth"] = np.log(df_beta["final"]) - np.log(df_beta["initial"])

# =========================
# 3. REGRESSION
# =========================
X = sm.add_constant(df_beta["initial"])
y = df_beta["growth"]

model = sm.OLS(y, X).fit()
print(model.summary())

# vytvoř složku
os.makedirs("../outputs/beta", exist_ok=True)

# =========================
# 4. GRAPH 1 – BASIC
# =========================
plt.figure(figsize=(8,6))
plt.scatter(df_beta["initial"], df_beta["growth"])

x_vals = np.linspace(df_beta["initial"].min(), df_beta["initial"].max(), 100)
y_vals = model.params["const"] + model.params["initial"] * x_vals

plt.plot(x_vals, y_vals)

plt.xlabel("Initial crime level (2013)")
plt.ylabel("Growth (log change 2013–2023)")
plt.title("Beta Convergence of Crime in EU")

plt.grid()
plt.tight_layout()
plt.savefig("../outputs/beta/beta_basic.png", dpi=200)
plt.close()

# =========================
# 5. GRAPH 2 – WITH LABELS
# =========================
plt.figure(figsize=(10,7))
plt.scatter(df_beta["initial"], df_beta["growth"])

for i, row in df_beta.iterrows():
    plt.text(row["initial"], row["growth"], row["country"], fontsize=8)

plt.plot(x_vals, y_vals)

plt.xlabel("Initial crime level (2013)")
plt.ylabel("Growth (log change 2013–2023)")
plt.title("Beta Convergence with Country Labels")

plt.grid()
plt.tight_layout()
plt.savefig("../outputs/beta/beta_labeled.png", dpi=200)
plt.close()

# =========================
# 6. GRAPH 3 – RESIDUALS
# =========================
df_beta["residuals"] = model.resid

plt.figure(figsize=(8,6))
plt.scatter(df_beta["initial"], df_beta["residuals"])

plt.axhline(0)

plt.xlabel("Initial crime level")
plt.ylabel("Residuals")
plt.title("Residual Plot (Beta Convergence Model)")

plt.grid()
plt.tight_layout()
plt.savefig("../outputs/beta/beta_residuals.png", dpi=200)
plt.close()

print("Beta graphs created ✅")