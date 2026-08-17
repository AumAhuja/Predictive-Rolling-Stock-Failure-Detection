import os
import argparse
import pandas as pd
from typing import List
from smart_train_ai.schema import FaultType, FaultLocation
from smart_train_ai.simulator.generator import generate_run_dataframe


def generate_full_dataset(
    output_dir: str = "data/synthetic",
    runs_per_class: int = 15,
    duration_seconds: float = 30.0,
):
    """Generates a complete multi-run synthetic train dataset across all fault classes and severities."""
    os.makedirs(output_dir, exist_ok=True)
    all_runs: List[pd.DataFrame] = []

    fault_classes = [
        FaultType.NORMAL,
        FaultType.WHEEL_FLAT,
        FaultType.AXLE_MISALIGNMENT,
        FaultType.BEARING_FAULT,
        FaultType.BRAKE_ABNORMAL,
        FaultType.SUSPENSION_FAULT,
        FaultType.MOTOR_FAULT,
        FaultType.BODY_DAMAGE,
    ]

    run_counter = 1

    for fault in fault_classes:
        for i in range(runs_per_class):
            run_id = f"RUN_{run_counter:03d}_{fault.value}"
            # Vary severity (0 for NORMAL, 1-3 for faults)
            severity = 0 if fault == FaultType.NORMAL else ((i % 3) + 1)
            speed = 100.0 + (i * 5.0) % 50.0
            load = 1.0 + (i * 0.2) % 1.5
            amb_temp = 22.0 + (i * 0.5) % 8.0

            df = generate_run_dataframe(
                fault_type=fault,
                run_id=run_id,
                duration_seconds=duration_seconds,
                speed_rpm=speed,
                load_kg=load,
                fault_severity=severity,
                ambient_temp=amb_temp,
                seed=42 + run_counter,
            )

            # Save individual run CSV file
            run_file = os.path.join(output_dir, f"{run_id}.csv")
            df.to_csv(run_file, index=False)
            all_runs.append(df)
            run_counter += 1

    # Combined master dataset
    master_df = pd.concat(all_runs, ignore_index=True)
    master_csv = os.path.join(output_dir, "master_synthetic_dataset.csv")
    master_df.to_csv(master_csv, index=False)
    print(f"Dataset generation complete: {len(all_runs)} runs ({len(master_df)} records) saved to {output_dir}")
    return master_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Train Synthetic Dataset Generator")
    parser.add_argument("--out_dir", type=str, default="data/synthetic", help="Output directory path")
    parser.add_argument("--runs_per_class", type=int, default=15, help="Number of runs per fault class")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration per run in seconds")
    args = parser.parse_args()

    generate_full_dataset(
        output_dir=args.out_dir,
        runs_per_class=args.runs_per_class,
        duration_seconds=args.duration,
    )
