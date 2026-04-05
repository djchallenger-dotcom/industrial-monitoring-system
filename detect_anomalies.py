import pandas as pd
import numpy as np


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect abnormal haul truck conditions and assign alert levels.

    This function looks for unusual sensor behavior such as:
    - high engine temperature
    - low hydraulic pressure
    - high vibration
    - low speed
    - temperature that is unusual compared to recent history

    It then combines these signals into:
    - abnormal_count: how many problems are active at once
    - health_score: a score from 0 to 100
    - alert_level: normal, warning, or critical
    - alert_message: readable explanation of the issue(s)
    """

    # Make a copy so we do not accidentally overwrite the original DataFrame.
    result = df.copy()

    # ---------------------------------------------------------------------
    # STEP 1: Create simple rule-based flags using fixed thresholds
    # ---------------------------------------------------------------------
    # Each of these columns becomes True or False depending on whether
    # the sensor value looks abnormal.

    # Flag temperatures above 220 as high.
    result["high_temp"] = result["engine_temperature"] > 220

    # Flag hydraulic pressure below 2600 as low.
    result["low_pressure"] = result["hydraulic_pressure"] < 2600

    # Flag vibration above 1.2 as high.
    result["high_vibration"] = result["vibration"] > 1.2

    # Flag speed below 15 as low.
    # A low speed by itself is not always a failure, but it can support
    # other warning signs.
    result["low_speed"] = result["speed"] < 15

    # ---------------------------------------------------------------------
    # STEP 2: Detect temperature anomalies using recent history
    # ---------------------------------------------------------------------
    # Fixed thresholds are useful, but they do not always catch gradual
    # changes. For example, temperature may still be below 220 but be rising
    # unusually fast compared to the truck's recent behavior.
    #
    # To handle that, we calculate a rolling mean and rolling standard
    # deviation for each truck separately.

    # Use a 6-hour rolling window.
    window = 6

    # Rolling mean of engine temperature for each truck.
    # min_periods=window means we require a full window before calculating
    # the rolling value. This avoids noisy results at the beginning.
    result["temp_mean"] = result.groupby("truck_id")["engine_temperature"].transform(
        lambda x: x.rolling(window, min_periods=window).mean()
    )

    # Rolling standard deviation of engine temperature for each truck.
    result["temp_std"] = result.groupby("truck_id")["engine_temperature"].transform(
        lambda x: x.rolling(window, min_periods=window).std()
    )

    # Temperature anomaly flag:
    # This becomes True when:
    # 1. rolling standard deviation exists
    # 2. we are far enough into the truck's timeline to trust the rolling stats
    # 3. the current temperature is more than 2.5 standard deviations away
    #    from the recent rolling mean
    #
    # cumcount() counts rows within each truck group starting from 0.
    # Requiring >= 12 helps avoid early false alarms.
    result["temp_anomaly"] = (
        result["temp_std"].notna() &
        (result.groupby("truck_id").cumcount() >= 12) &
        ((result["engine_temperature"] - result["temp_mean"]).abs() > 2.5 * result["temp_std"])
    )

    # ---------------------------------------------------------------------
    # STEP 3: Count how many abnormal conditions are active
    # ---------------------------------------------------------------------
    # This gives a simple measure of how many warning signals are happening
    # at the same time.
    abnormal_cols = [
        "high_temp",
        "low_pressure",
        "high_vibration",
        "low_speed",
        "temp_anomaly"
    ]
    result["abnormal_count"] = result[abnormal_cols].sum(axis=1)

    # ---------------------------------------------------------------------
    # STEP 4: Create a health score
    # ---------------------------------------------------------------------
    # Start each row with a perfect health score of 100.
    # Then subtract points for each abnormal condition.
    #
    # Larger penalties are used for more serious issues.
    result["health_score"] = 100

    # High temperature is important, so subtract 25 points.
    result.loc[result["high_temp"], "health_score"] -= 25

    # Low hydraulic pressure is serious, so subtract 30 points.
    result.loc[result["low_pressure"], "health_score"] -= 30

    # High vibration can indicate mechanical issues, so subtract 30 points.
    result.loc[result["high_vibration"], "health_score"] -= 30

    # Low speed is less severe on its own, so subtract 10 points.
    result.loc[result["low_speed"], "health_score"] -= 10

    # Temperature anomaly from rolling history gets a moderate penalty.
    result.loc[result["temp_anomaly"], "health_score"] -= 15

    # Prevent the score from going below 0.
    result["health_score"] = result["health_score"].clip(lower=0)

    # ---------------------------------------------------------------------
    # STEP 5: Convert health score into alert levels
    # ---------------------------------------------------------------------
    # Lower health score = worse condition.
    #
    # - below 70  -> critical
    # - below 90  -> warning
    # - otherwise -> normal
    result["alert_level"] = np.select(
        [
            result["health_score"] < 70,
            result["health_score"] < 90
        ],
        [
            "critical",
            "warning"
        ],
        default="normal"
    )

    # ---------------------------------------------------------------------
    # STEP 6: Create a human-readable alert message
    # ---------------------------------------------------------------------
    # This turns the True/False flags into a readable explanation.
    def build_alert_message(row):
        issues = []

        if row["high_temp"]:
            issues.append("engine temperature high")
        if row["low_pressure"]:
            issues.append("hydraulic pressure low")
        if row["high_vibration"]:
            issues.append("vibration high")
        if row["low_speed"]:
            issues.append("speed low")
        if row["temp_anomaly"]:
            issues.append("temperature anomaly vs recent trend")

        # If no issues were found, label it as no alert.
        if not issues:
            return "no alert"

        # Join all detected issues into one readable string.
        return ", ".join(issues)

    result["alert_message"] = result.apply(build_alert_message, axis=1)

    # Return the updated DataFrame with all added columns.
    return result


def main() -> None:
    """
    Load sensor data, detect anomalies, save results, and print a summary.
    """

    # Read the simulated haul truck sensor data.
    df = pd.read_csv("haul_truck_sensor_data.csv")

    # Convert timestamp column from text into datetime format.
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Run the anomaly detection logic.
    alerts_df = detect_anomalies(df)

    # Save the results to a new CSV file.
    alerts_df.to_csv("haul_truck_alerts.csv", index=False)

    # Print a short summary so we can inspect the results quickly.
    print("Alerts saved to: haul_truck_alerts.csv")
    print("\nAlert level counts:")
    print(alerts_df["alert_level"].value_counts())

    print("\nSample alerts:")
    print(
        alerts_df.loc[
            alerts_df["alert_level"] != "normal",
            ["timestamp", "truck_id", "alert_level", "health_score", "alert_message"]
        ].head(20)
    )


if __name__ == "__main__":
    main()
