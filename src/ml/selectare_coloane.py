

import pandas as pd

# 1. Citim fisierul CSV
df = pd.read_csv("data/telecomunicatii.csv")

# 2. Alegem doar coloanele importante pentru proiect
coloane_importante = [
    "account_length",
    "international_plan",
    "voice_mail_plan",
    "number_vmail_messages",
    "total_day_minutes",
    "total_day_calls",
    "total_eve_minutes",
    "total_eve_calls",
    "total_night_minutes",
    "total_night_calls",
    "total_intl_minutes",
    "total_intl_calls",
    "number_customer_service_calls",
    "payment_delay"
]

# 3. Cream un dataframe doar cu aceste coloane
df_selectat = df[coloane_importante]

# 4. Afisam primele randuri ca sa verificam
print(df_selectat.head())

# 5. Salvam intr-un fisier CSV nou
df_selectat.to_csv("data/telecomunicatii_coloane_importante.csv", index=False)

print("Fisierul cu coloanele importante a fost creat cu succes.")