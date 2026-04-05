import pandas as pd
import matplotlib.pyplot as plt


def plot_truck_sensors(df: pd.DataFrame, truck_id: str) -> None:
    """
    Plot sensor trends for a single haul truck.

    This function creates separate line plots for:
    - Engine temperature
    - Hydraulic pressure
    - Vibration
    - Speed

    Each plot shows how the sensor values change over time.
    """

    # ---------------------------------------------------------------------
    # STEP 1: Filter data for a single truck
    # ---------------------------------------------------------------------
    # We only want to plot data for one truck at a time.
    truck_df = df[df["truck_id"] == truck_id].copy()

    # ---------------------------------------------------------------------
    # STEP 2: Plot engine temperature
    # ---------------------------------------------------------------------
    # This helps identify overheating trends.
    plt.figure(figsize=(12, 4))
    plt.plot(truck_df["timestamp"], truck_df["engine_temperature"])
    plt.title(f"{truck_id} - Engine Temperature")
    plt.xlabel("Timestamp")
    plt.ylabel("Temperature")

    # Rotate timestamps so they are easier to read
    plt.xticks(rotation=45)

    # Adjust layout so labels do not overlap
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------------
    # STEP 3: Plot hydraulic pressure
    # ---------------------------------------------------------------------
    # Useful for detecting hydraulic system failures.
    plt.figure(figsize=(12, 4))
    plt.plot(truck_df["timestamp"], truck_df["hydraulic_pressure"])
    plt.title(f"{truck_id} - Hydraulic Pressure")
    plt.xlabel("Timestamp")
    plt.ylabel("Pressure")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------------
    # STEP 4: Plot vibration
    # ---------------------------------------------------------------------
    # High vibration can indicate mechanical issues.
    plt.figure(figsize=(12, 4))
    plt.plot(truck_df["timestamp"], truck_df["vibration"])
    plt.title(f"{truck_id} - Vibration")
    plt.xlabel("Timestamp")
    plt.ylabel("Vibration")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------------
    # STEP 5: Plot speed
    # ---------------------------------------------------------------------
    # Speed helps provide context for other sensor readings.
    # For example, low speed combined with other issues can indicate failure.
    plt.figure(figsize=(12, 4))
    plt.plot(truck_df["timestamp"], truck_df["speed"])
    plt.title(f"{truck_id} - Speed")
    plt.xlabel("Timestamp")
    plt.ylabel("Speed")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Load the dataset and generate plots for each haul truck.
    """

    # ---------------------------------------------------------------------
    # STEP 1: Load the sensor data
    # ---------------------------------------------------------------------
    df = pd.read_csv("haul_truck_sensor_data.csv")

    # Convert timestamp column from string to datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ---------------------------------------------------------------------
    # STEP 2: Loop through each truck and generate plots
    # ---------------------------------------------------------------------
    # unique() gives us a list of all truck IDs (e.g., HT-01, HT-02, HT-03)
    for truck_id in df["truck_id"].unique():
        plot_truck_sensors(df, truck_id)


if __name__ == "__main__":
    main()