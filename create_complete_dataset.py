import pandas as pd

# Load all datasets
eui = pd.read_csv("data/global_eui.csv")
gpr = pd.read_csv("data/gpr_data.csv")
oil = pd.read_csv("data/oil_price.csv")
cpu = pd.read_csv("data/cpu_index.csv")

# --- Make sure 'date' is a datetime column ---
for df in [eui, gpr, oil, cpu]:
    df['date'] = pd.to_datetime(df['date'])

# --- Optional: rename value columns to avoid confusion ---
# (Change the right side of rename() to the actual variable column name in your file)
eui = eui.rename(columns={'value': 'EUI'})
gpr = gpr.rename(columns={'value': 'GPR'})
oil = oil.rename(columns={'value': 'OIL_PRICE'})
cpu = cpu.rename(columns={'value': 'CPU_INDEX'})

# --- Merge them all on the 'date' column (outer join keeps all dates) ---
merged = eui.merge(gpr, on='date', how='outer') \
             .merge(oil, on='date', how='outer') \
             .merge(cpu, on='date', how='outer')

# --- Sort by date ---
merged = merged.sort_values('date').reset_index(drop=True)

# --- Optionally: handle duplicates (if same date appears multiple times) ---
merged = merged.groupby('date', as_index=False).mean()

# --- Save final unified dataset ---
merged.to_csv("merged_energy_uncertainty_dataset.csv", index=False)

print("✅ Unified dataset saved as 'merged_energy_uncertainty_dataset.csv'")
