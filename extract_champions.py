# extract real name from train
import pandas as pd

print("Loading dataset to extract champion names")
try:
    df = pd.read_csv('lol_dataset_challenger_grandmaster_clean.csv', encoding='latin-1', nrows=100000)
except:
    df = pd.read_csv('lol_dataset_challenger_grandmaster_clean.csv', encoding='iso-8859-1', nrows=100000)


champs = sorted(df['championName'].dropna().unique())
print(f"\nActual champions from dataset ({len(champs)}):")
for c in champs:
    print(f'  "{c}",')
