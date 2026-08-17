import numpy as np
from typing import Dict, Any, List, Tuple
from smart_train_ai.schema import FaultType, FaultSeverity


class HealthEngine:
    """Computes train health score (0-100), operating status, maintenance priority, and human-readable evidence."""

    def __init__(
        self,
        warning_threshold: int = 75,
        critical_threshold: int = 50,
        weights: Dict[str, float] = None,
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.weights = weights or {
            "anomaly": 0.30,
            "vibration": 0.25,
            "thermal": 0.20,
            "electrical": 0.15,
            "fault_confidence": 0.10,
        }

    def compute_health(
        self,
        anomaly_score: float,
        fault_type: str,
        fault_severity: int,
        confidence: float,
        features: Dict[str, float],
    ) -> Tuple[int, str, str, List[str]]:
        """
        Computes composite health score, status, maintenance priority, and evidence.

        Returns:
        - health_score: int (0..100)
        - status: str ("HEALTHY", "WARNING", "CRITICAL")
        - maintenance_priority: str ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        - evidence: List[str]
        """
        penalties = {}
        evidence_list = []

        # 1. Anomaly Penalty
        anomaly_penalty = np.clip(anomaly_score * 100.0, 0.0, 100.0)
        penalties["anomaly"] = anomaly_penalty
        if anomaly_score > 0.40:
            evidence_list.append(f"Anomaly score elevated ({anomaly_score:.2f})")

        # 2. Vibration Penalty (IMU RMS & Kurtosis)
        imu_z_rms = features.get("imu_z_rms", 1.0)
        imu_mag_rms = features.get("imu_mag_rms", 1.0)
        imu_kurtosis = features.get("imu_z_kurtosis", 0.0)

        vib_penalty = 0.0
        if imu_mag_rms > 1.2:
            vib_penalty += np.clip((imu_mag_rms - 1.2) * 50.0, 0.0, 60.0)
            evidence_list.append(f"Vibration RMS increased ({imu_mag_rms:.2f}g)")
        if imu_kurtosis > 3.0:
            vib_penalty += np.clip((imu_kurtosis - 3.0) * 10.0, 0.0, 40.0)
            evidence_list.append(f"Vibration shock impulses detected (kurtosis {imu_kurtosis:.1f})")

        penalties["vibration"] = np.clip(vib_penalty, 0.0, 100.0)

        # 3. Thermal Penalty (Slope & Rise)
        bearing_slope = features.get("bearing_temp_slope", 0.0)
        bearing_rise = features.get("bearing_temp_rise", 0.0)
        motor_slope = features.get("motor_temp_slope", 0.0)

        thermal_penalty = 0.0
        if bearing_slope > 0.15:
            thermal_penalty += np.clip(bearing_slope * 200.0, 0.0, 70.0)
            evidence_list.append(f"Bearing temperature rising rapidly (+{bearing_slope:.2f}°C/s)")
        if bearing_rise > 4.0:
            thermal_penalty += np.clip(bearing_rise * 8.0, 0.0, 50.0)
            evidence_list.append(f"Bearing temperature rise delta detected ({bearing_rise:.1f}°C)")
        if motor_slope > 0.30:
            thermal_penalty += np.clip(motor_slope * 150.0, 0.0, 50.0)
            evidence_list.append(f"Motor thermal dissipation slope elevated (+{motor_slope:.2f}°C/s)")

        penalties["thermal"] = np.clip(thermal_penalty, 0.0, 100.0)

        # 4. Electrical Penalty (Current / RPM, Power variation)
        current_per_rpm = features.get("current_per_rpm", 0.0)
        current_std = features.get("current_std", 0.0)
        rpm_var = features.get("rpm_variation", 0.0)

        elec_penalty = 0.0
        if current_per_rpm > 0.003:
            elec_penalty += 35.0
            evidence_list.append("Electrical current to RPM ratio elevated (mechanical drag)")
        if current_std > 0.04:
            elec_penalty += 25.0
            evidence_list.append(f"Electrical current ripple detected (std {current_std:.3f}A)")
        if rpm_var > 3.0:
            elec_penalty += 30.0
            evidence_list.append(f"RPM speed instability detected (variation index {rpm_var:.1f})")

        penalties["electrical"] = np.clip(elec_penalty, 0.0, 100.0)

        # 5. Fault Severity & Confidence Penalty
        fault_penalty = 0.0
        if fault_type != FaultType.NORMAL.value:
            fault_penalty = np.clip((fault_severity + 1) * 25.0 * confidence, 0.0, 100.0)
            evidence_list.append(f"Classified fault '{fault_type}' with {confidence * 100:.0f}% confidence")

        penalties["fault_confidence"] = fault_penalty

        # Weighted Total Penalty
        total_penalty = sum(penalties[k] * self.weights.get(k, 0.20) for k in penalties)
        health_score = int(np.clip(100.0 - total_penalty, 0, 100))

        # Operating Status & Maintenance Priority
        if health_score >= self.warning_threshold:
            status = "HEALTHY"
            priority = "LOW"
        elif health_score >= self.critical_threshold:
            status = "WARNING"
            priority = "MEDIUM" if fault_severity <= 1 else "HIGH"
        else:
            status = "CRITICAL"
            priority = "CRITICAL" if fault_severity >= 2 else "HIGH"

        if not evidence_list:
            evidence_list.append("All physical parameters operating within normal parameters")

        return health_score, status, priority, evidence_list
