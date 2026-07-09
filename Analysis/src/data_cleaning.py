import pandas as pd
import numpy as np

EU27 = [
    "Austria","Belgium","Bulgaria","Croatia","Cyprus","Czechia","Denmark",
    "Estonia","Finland","France","Germany","Greece","Hungary","Ireland",
    "Italy","Latvia","Lithuania","Luxembourg","Malta","Netherlands",
    "Poland","Portugal","Romania","Slovakia","Slovenia","Spain","Sweden"
]


def clean_gdp(path):
    gdp_raw = pd.read_csv(path)

    gdp = gdp_raw.rename(columns={
        "geo": "country",
        "TIME_PERIOD": "year",
        "OBS_VALUE": "gdp_index"
    })

    gdp["year"] = pd.to_numeric(gdp["year"], errors="coerce").astype(int)
    gdp["gdp_index"] = pd.to_numeric(gdp["gdp_index"], errors="coerce")

    gdp = gdp[(gdp["year"] >= 2013) & (gdp["year"] <= 2023)]
    gdp = gdp[gdp["country"].isin(EU27)]

    gdp = gdp[["country", "year", "gdp_index"]].dropna()

    # log transformace
    gdp["log_gdp"] = np.log(gdp["gdp_index"])

    return gdp


def clean_crime(path):
    crime_raw = pd.read_csv(path)

    crime = crime_raw[crime_raw["unit"] == "Per hundred thousand inhabitants"].copy()

    crime = crime.rename(columns={
        "geo": "country",
        "TIME_PERIOD": "year",
        "OBS_VALUE": "crime_rate",
        "iccs": "offence"
    })

    crime["year"] = pd.to_numeric(crime["year"], errors="coerce").astype(int)
    crime["crime_rate"] = pd.to_numeric(crime["crime_rate"], errors="coerce")

    crime = crime[(crime["year"] >= 2013) & (crime["year"] <= 2023)]
    crime = crime[crime["country"].isin(EU27)]

    crime = crime[["country", "year", "offence", "crime_rate"]].dropna()

    # log transformace (malá konstanta kvůli 0 hodnotám)
    crime["log_crime_rate"] = np.log(crime["crime_rate"] + 1)

    return crime
def prepare_total_crime(crime_df):
    total = crime_df.groupby(["country", "year"])["crime_rate"].sum().reset_index()
    return total