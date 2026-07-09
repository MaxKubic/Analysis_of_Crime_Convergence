from data_cleaning import clean_crime, prepare_total_crime
import os

print(os.listdir("/Users/maxkubicek/PycharmProjects/Bachelor/data"))

# načtení a čištění
crime = clean_crime("/Users/maxkubicek/PycharmProjects/Bachelor/data/crim_off_cat__custom_20354171_linear.csv")

# agregace
crime_total = prepare_total_crime(crime)

# uložení
crime_total.to_csv("../data/crime_clean.csv", index=False)

print("crime_clean.csv created ✅")