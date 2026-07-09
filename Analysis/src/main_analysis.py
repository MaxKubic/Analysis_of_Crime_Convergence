import os
import matplotlib.pyplot as plt
from data_cleaning import clean_gdp, clean_crime

# =========================
# 1. LOAD & CLEAN DATA
# =========================

gdp = clean_gdp("../data/tec00114__custom_20353208_linear.csv")
crime = clean_crime("../data/crim_off_cat__custom_20354171_linear.csv")

# =========================
# 2. SANITY CHECKS
# =========================

print("GDP countries:", gdp["country"].nunique())
print("Crime countries:", crime["country"].nunique())
print("Crime offence types:", crime["offence"].nunique())

# =========================
# 3. SELECT OFFENCES
# =========================

selected_offences = [
    "Intentional homicide",
    "Robbery",
    "Burglary of private residential premises",
    "Unlawful acts involving controlled drugs or precursors"
]

# vytvoříme složku pro grafy
os.makedirs("../outputs/graphs", exist_ok=True)

# =========================
# 4. SIGMA-CONVERGENCE
# =========================

plt.figure(figsize=(9,6))

for off in selected_offences:
    df = crime[crime["offence"] == off].copy()
    panel = df.merge(gdp, on=["country", "year"], how="inner")

    sigma = panel.groupby("year")["crime_rate"].std()

    plt.plot(sigma.index, sigma.values, marker="o", label=off)

    # uložit individuální graf
    plt.figure(figsize=(8,5))
    plt.plot(sigma.index, sigma.values, marker="o")
    plt.title(f"Sigma-convergence: {off} (EU-27, 2013–2023)")
    plt.xlabel("Year")
    plt.ylabel("Std. deviation across countries")
    plt.grid(True)
    plt.tight_layout()

    safe_name = off.lower().replace(" ", "_")
    plt.savefig(f"../outputs/graphs/sigma_{safe_name}.png", dpi=200)
    plt.close()

# společný graf
plt.title("Sigma-convergence across crime types (EU-27, 2013–2023)")
plt.xlabel("Year")
plt.ylabel("Std. deviation across countries")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../outputs/graphs/sigma_all_selected.png", dpi=200)
plt.show()