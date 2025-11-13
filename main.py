from __future__ import annotations

from preprocessing import preprocess_all


def main():

    # Preprocess data

    artifacts = preprocess_all(
    data_path="data/complete_dataset.csv",
    date_col="date",
    start="1996-01-01",
    end="2023-11-30",
    select_cols=[
        "date",
        "global_EUI_GDP_weighted", "GPR", "cpu_index",
        "Europe Brent Spot Price FOB (Dollars per Barrel)"
    ],
    rename_map={
        "global_EUI_GDP_weighted": "eui",
        "GPR": "gpr",
        "cpu_index": "cpu",
        "Europe Brent Spot Price FOB (Dollars per Barrel)": "oil_price",
    },
    log_cols=["eui", "gpr", "cpu", "oil_price"],   # keeps log_<col>
    diff_cols=["eui", "cpu", "oil_price"],         # makes d_<col> and dlog_<col>
    add_event_flags=True,
    train_ratio=0.8,
    save=False,
    save_path=None,
    visualize_flags={
        "variables": ("eui", "gpr", "cpu", "oil_price", "log_eui", "log_gpr", "log_cpu",
                      "log_oil_price", "d_eui", "d_cpu", "d_oil_price", "dlog_eui",
                      "dlog_cpu", "dlog_oil_price",
        ),
        "lags": 36,
        "time_series": True,
        "acf": True,
        "pacf": True,
        "pacf_method": "ywm",
    },
)


    # ✅ Quick structure check
    train_df = artifacts["train_df"]
    test_df = artifacts["test_df"]

    print("\n--- TRAIN SET ---")
    print(train_df.shape)
    print(train_df)

    print("\n--- TEST SET ---")
    print(test_df.shape)
    print(test_df)

if __name__ == "__main__":
    main()



