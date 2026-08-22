import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "log_amount",
    "log_city_population",
    "distance_km",
    "transaction_hour",
    "transaction_weekday",
    "transaction_month",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "log_hours_since_previous",
    "amount_to_prior_mean",
]


def build_feature_tables(train_df, test_df):
    """Build aligned development and test tables from two raw DataFrames."""

    # Work on copies and keep track of where each transaction came from.
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["source"] = "train"
    test_df["source"] = "test"

    # Combine and sort the files so the history features follow time order.
    transactions = pd.concat([train_df, test_df], ignore_index=True)
    transactions["trans_date_trans_time"] = pd.to_datetime(
        transactions["trans_date_trans_time"]
    )
    transactions = transactions.sort_values(
        ["trans_date_trans_time", "trans_num"],
        kind="mergesort",
    ).reset_index(drop=True)

    # Reduce the long right tails in amount and city population.
    transactions["log_amount"] = np.log1p(transactions["amt"])
    transactions["log_city_population"] = np.log1p(transactions["city_pop"])

    # Keep the original time values and add their cyclical versions.
    hour = transactions["trans_date_trans_time"].dt.hour
    weekday = transactions["trans_date_trans_time"].dt.dayofweek
    month_index = transactions["trans_date_trans_time"].dt.month - 1

    transactions["transaction_hour"] = hour
    transactions["transaction_weekday"] = weekday
    transactions["transaction_month"] = month_index + 1
    transactions["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    transactions["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    transactions["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    transactions["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    transactions["month_sin"] = np.sin(2 * np.pi * month_index / 12)
    transactions["month_cos"] = np.cos(2 * np.pi * month_index / 12)

    # Calculate the distance between the customer and merchant.
    customer_latitude = np.radians(transactions["lat"])
    customer_longitude = np.radians(transactions["long"])
    merchant_latitude = np.radians(transactions["merch_lat"])
    merchant_longitude = np.radians(transactions["merch_long"])

    latitude_difference = merchant_latitude - customer_latitude
    longitude_difference = merchant_longitude - customer_longitude
    haversine_value = (
        np.sin(latitude_difference / 2) ** 2
        + np.cos(customer_latitude)
        * np.cos(merchant_latitude)
        * np.sin(longitude_difference / 2) ** 2
    )
    transactions["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(haversine_value))

    # Build the two cardholder-history features from earlier transactions only.
    cardholder_time = (
        transactions.groupby(["cc_num", "trans_date_trans_time"], as_index=False)
        .agg(Amount_at_time=("amt", "sum"), Transactions_at_time=("amt", "size"))
        .sort_values(["cc_num", "trans_date_trans_time"], kind="mergesort")
    )
    by_cardholder = cardholder_time.groupby("cc_num", sort=False)

    cardholder_time["prior_transaction_count"] = (
        by_cardholder["Transactions_at_time"].cumsum()
        - cardholder_time["Transactions_at_time"]
    )
    cardholder_time["prior_amount_sum"] = (
        by_cardholder["Amount_at_time"].cumsum()
        - cardholder_time["Amount_at_time"]
    )
    cardholder_time["prior_mean_amount"] = (
        cardholder_time["prior_amount_sum"]
        / cardholder_time["prior_transaction_count"].replace(0, np.nan)
    )
    cardholder_time["previous_transaction_time"] = (
        by_cardholder["trans_date_trans_time"].shift()
    )

    transactions = transactions.merge(
        cardholder_time[[
            "cc_num",
            "trans_date_trans_time",
            "prior_transaction_count",
            "prior_mean_amount",
            "previous_transaction_time",
        ]],
        on=["cc_num", "trans_date_trans_time"],
        how="left",
        validate="many_to_one",
    )

    hours_since_previous = (
        transactions["trans_date_trans_time"]
        - transactions["previous_transaction_time"]
    ).dt.total_seconds() / 3600

    transactions["log_hours_since_previous"] = np.log1p(
        hours_since_previous.clip(lower=0)
    ).fillna(0)
    transactions["amount_to_prior_mean"] = (
        transactions["amt"] / transactions["prior_mean_amount"]
    ).clip(upper=20).fillna(1)

    # Keep only the variables chosen after the EDA.
    feature_table = transactions[
        ["source", "trans_date_trans_time", "is_fraud"]
        + NUMERIC_FEATURES
        + ["category"]
    ].copy()

    train_feature_table = feature_table[
        feature_table["source"] == "train"
    ].drop(columns="source")
    test_feature_table = feature_table[
        feature_table["source"] == "test"
    ].drop(columns="source")

    development_data = pd.get_dummies(
        train_feature_table,
        columns=["category"],
        dtype="int8",
    ).reset_index(drop=True)
    final_test_data = pd.get_dummies(
        test_feature_table,
        columns=["category"],
        dtype="int8",
    ).reset_index(drop=True)
    final_test_data = final_test_data.reindex(
        columns=development_data.columns,
        fill_value=0,
    )

    model_features = [
        column
        for column in development_data.columns
        if column not in ["trans_date_trans_time", "is_fraud"]
    ]
    history_rows = transactions["previous_transaction_time"].notna()

    if len(development_data) != len(train_df) or len(final_test_data) != len(test_df):
        raise ValueError("The feature tables do not contain one row per transaction.")
    if not development_data.columns.equals(final_test_data.columns):
        raise ValueError("The development and test columns do not match.")
    if (
        development_data[model_features].isna().any().any()
        or final_test_data[model_features].isna().any().any()
    ):
        raise ValueError("The model features contain missing values.")
    if not (
        np.isfinite(development_data[model_features].to_numpy()).all()
        and np.isfinite(final_test_data[model_features].to_numpy()).all()
    ):
        raise ValueError("The model features contain infinite values.")
    if not (
        transactions.loc[history_rows, "previous_transaction_time"]
        < transactions.loc[history_rows, "trans_date_trans_time"]
    ).all():
        raise ValueError("A history feature contains a current or future transaction.")

    return transactions, development_data, final_test_data
