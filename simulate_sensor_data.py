import pandas as pd
import numpy as np


def generate_timestamps(start_date: str, days: int, freq: str = "1H") -> pd.DatetimeIndex:
    """
    Generate a sequence of timestamps.

    Args:
        start_date: Start date in YYYY-MM-DD format.
        days: Number of days to simulate.
        freq: Frequency of timestamps (e.g., "1H" = hourly).

    Returns:
        A pandas DatetimeIndex representing the timeline.
    """

    # Total number of timestamps (hours)
    periods = days * 24

    # Generate evenly spaced timestamps starting from start_date
    return pd.date_range(start=start_date, periods=periods, freq=freq.lower())


def simulate_truck_data(
    truck_id: str,
    timestamps: pd.DatetimeIndex,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Simulate normal (healthy) haul truck sensor data.

    Each sensor is generated using a normal distribution with some noise.
    This represents typical operating conditions.

    Args:
        truck_id: Truck identifier (e.g., HT-01)
        timestamps: Time range for the simulation
        rng: Random number generator for reproducibility

    Returns:
        DataFrame containing simulated sensor data
    """

    n = len(timestamps)

    # ---------------------------------------------------------------------
    # STEP 1: Generate baseline sensor values
    # ---------------------------------------------------------------------
    # These represent "normal" operating conditions with some randomness.

    engine_temperature = rng.normal(loc=195, scale=5, size=n)
    hydraulic_pressure = rng.normal(loc=3000, scale=100, size=n)
    vibration = rng.normal(loc=0.45, scale=0.10, size=n)
    speed = rng.normal(loc=25, scale=4, size=n)

    # ---------------------------------------------------------------------
    # STEP 2: Prevent unrealistic values
    # ---------------------------------------------------------------------
    # Ensure no impossible values (e.g., negative speed or vibration).

    vibration = np.clip(vibration, 0.05, None)
    speed = np.clip(speed, 0, None)

    # ---------------------------------------------------------------------
    # STEP 3: Create DataFrame
    # ---------------------------------------------------------------------
    df = pd.DataFrame({
        "timestamp": timestamps,
        "truck_id": truck_id,
        "engine_temperature": engine_temperature,
        "hydraulic_pressure": hydraulic_pressure,
        "vibration": vibration,
        "speed": speed,
    })

    # Initially, everything is considered normal
    df["status"] = "normal"

    return df


def inject_abnormal_event(
    df: pd.DataFrame,
    start_idx: int,
    duration: int,
    event_type: str
) -> pd.DataFrame:
    """
    Inject abnormal behavior into the sensor data.

    This simulates real-world failures by modifying sensor values over time.

    Args:
        df: DataFrame for a single truck
        start_idx: Starting index of the abnormal event
        duration: How long the event lasts (number of rows)
        event_type: Type of failure to simulate

    Returns:
        Updated DataFrame with abnormal patterns injected
    """

    # Ensure we do not go out of bounds
    end_idx = min(start_idx + duration, len(df))

    # ---------------------------------------------------------------------
    # EVENT TYPE 1: Overheating
    # ---------------------------------------------------------------------
    # Gradual increase in temperature + slight drop in speed
    if event_type == "overheating":
        df.loc[start_idx:end_idx - 1, "engine_temperature"] += np.linspace(15, 35, end_idx - start_idx)
        df.loc[start_idx:end_idx - 1, "speed"] -= np.linspace(2, 8, end_idx - start_idx)

        # Label as warning (not immediately critical)
        df.loc[start_idx:end_idx - 1, "status"] = "warning"

    # ---------------------------------------------------------------------
    # EVENT TYPE 2: Hydraulic issue
    # ---------------------------------------------------------------------
    # Pressure drops while vibration increases (more severe)
    elif event_type == "hydraulic_issue":
        df.loc[start_idx:end_idx - 1, "hydraulic_pressure"] -= np.linspace(250, 600, end_idx - start_idx)
        df.loc[start_idx:end_idx - 1, "vibration"] += np.linspace(0.3, 0.9, end_idx - start_idx)

        # Label as critical (more severe failure)
        df.loc[start_idx:end_idx - 1, "status"] = "critical"

    # ---------------------------------------------------------------------
    # EVENT TYPE 3: Mechanical vibration issue
    # ---------------------------------------------------------------------
    # Sudden increase in vibration + drop in speed
    elif event_type == "mechanical_vibration":
        df.loc[start_idx:end_idx - 1, "vibration"] += np.linspace(0.5, 1.2, end_idx - start_idx)
        df.loc[start_idx:end_idx - 1, "speed"] -= np.linspace(1, 6, end_idx - start_idx)

        # Label as warning
        df.loc[start_idx:end_idx - 1, "status"] = "warning"

    # ---------------------------------------------------------------------
    # STEP: Clean up values after injection
    # ---------------------------------------------------------------------
    # Prevent invalid values after adjustments
    df["vibration"] = df["vibration"].clip(lower=0.05)
    df["speed"] = df["speed"].clip(lower=0)

    return df


def main() -> None:
    """
    Main function to generate the full dataset.

    This:
    - creates timestamps
    - simulates data for multiple trucks
    - injects abnormal events
    - combines everything into one dataset
    - saves it to CSV
    """

    # Random number generator (fixed seed for reproducibility)
    rng = np.random.default_rng(seed=42)

    # ---------------------------------------------------------------------
    # STEP 1: Define simulation settings
    # ---------------------------------------------------------------------
    truck_ids = ["HT-01", "HT-02", "HT-03"]

    # Generate 14 days of hourly timestamps
    timestamps = generate_timestamps(start_date="2026-01-01", days=14)

    all_truck_data = []

    # ---------------------------------------------------------------------
    # STEP 2: Simulate each truck
    # ---------------------------------------------------------------------
    for truck_id in truck_ids:
        truck_df = simulate_truck_data(
            truck_id=truck_id,
            timestamps=timestamps,
            rng=rng
        )

        # -----------------------------------------------------------------
        # STEP 3: Inject abnormal events
        # -----------------------------------------------------------------
        # Each truck gets a different type of failure

        if truck_id == "HT-01":
            truck_df = inject_abnormal_event(
                df=truck_df,
                start_idx=220,
                duration=18,
                event_type="overheating"
            )

        elif truck_id == "HT-02":
            truck_df = inject_abnormal_event(
                df=truck_df,
                start_idx=260,
                duration=14,
                event_type="hydraulic_issue"
            )

        elif truck_id == "HT-03":
            truck_df = inject_abnormal_event(
                df=truck_df,
                start_idx=150,
                duration=10,
                event_type="mechanical_vibration"
            )

        all_truck_data.append(truck_df)

    # ---------------------------------------------------------------------
    # STEP 4: Combine all trucks into one dataset
    # ---------------------------------------------------------------------
    final_df = pd.concat(all_truck_data, ignore_index=True)

    # ---------------------------------------------------------------------
    # STEP 5: Clean up output
    # ---------------------------------------------------------------------
    # Round values for cleaner presentation
    numeric_cols = ["engine_temperature", "hydraulic_pressure", "vibration", "speed"]
    final_df[numeric_cols] = final_df[numeric_cols].round(2)

    # Save to CSV
    output_path = "haul_truck_sensor_data.csv"
    final_df.to_csv(output_path, index=False)

    # ---------------------------------------------------------------------
    # STEP 6: Print summary for quick inspection
    # ---------------------------------------------------------------------
    print(f"Simulated data saved to: {output_path}")

    print("\nPreview:")
    print(final_df.head())

    print("\nStatus distribution:")
    print(final_df["status"].value_counts())

    print("\nSample abnormal rows:")
    abnormal_df = final_df[final_df["status"] != "normal"]
    print(abnormal_df.head(20))


if __name__ == "__main__":
    main()
