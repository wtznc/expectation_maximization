"""
Real-World Simulations of the Expectation-Maximization Algorithm

Each example generates realistic synthetic data, runs EM, and prints results
so you can see exactly how the algorithm discovers hidden structure.

Examples:
    1. Customer Segmentation  — find distinct spending groups in retail data
    2. Anomaly Detection      — separate normal network traffic from attacks
    3. Species Classification — identify plant sub-species from measurements
    4. Sensor Calibration     — unmix readings from overlapping sensors

Run:
    python real_world_examples.py
"""

import numpy as np
from em_algorithm import expectation_maximization, generate_mixture_data


# ── Helpers ────────────────────────────────────────────────────────────────── #

def print_header(title, description):
    print("\n" + "=" * 70)
    print(f"  EXAMPLE: {title}")
    print("=" * 70)
    print(f"\n  Scenario: {description}\n")


def print_comparison(true_params, learned_means, learned_stds, learned_weights, labels):
    """Print a side-by-side comparison of ground truth vs learned parameters."""
    order = np.argsort(learned_means)
    learned_means = learned_means[order]
    learned_stds = learned_stds[order]
    learned_weights = learned_weights[order]

    print(f"  {'Component':<14} {'True Mean':>10} {'Learned Mean':>13} "
          f"{'True Std':>10} {'Learned Std':>12} "
          f"{'True Weight':>12} {'Learned Wt':>11}")
    print("  " + "-" * 84)
    for i, (tm, ts, tw) in enumerate(true_params):
        print(f"  {'  #' + str(i+1):<14} {tm:>10.1f} {learned_means[i]:>13.2f} "
              f"{ts:>10.1f} {learned_stds[i]:>12.2f} "
              f"{tw:>12.2f} {learned_weights[i]:>11.2f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════ #
#  1. CUSTOMER SEGMENTATION                                                     #
# ══════════════════════════════════════════════════════════════════════════════ #

def example_customer_segmentation():
    print_header(
        "Customer Segmentation (Retail)",
        "An online store tracks monthly spending per customer.\n"
        "  There are three hidden groups: budget shoppers, regular shoppers,\n"
        "  and premium shoppers. The store doesn't know who belongs to which\n"
        "  group — it only sees the dollar amounts. EM discovers the groups."
    )

    # Ground truth (hidden from the algorithm)
    true_means = [25.0, 120.0, 450.0]   # avg monthly spend ($)
    true_stds = [10.0, 35.0, 80.0]
    true_weights = [0.50, 0.35, 0.15]   # 50% budget, 35% regular, 15% premium

    data, labels = generate_mixture_data(true_means, true_stds, true_weights,
                                         n_samples=3000, seed=10)

    weights, means, stds, lls = expectation_maximization(data, n_components=3)

    true_params = list(zip(true_means, true_stds, true_weights))
    print(f"  Customers simulated: {len(data)}")
    print(f"  EM iterations:       {len(lls)}\n")
    print("  Results (spending in $):\n")
    print_comparison(true_params, means, stds, weights, labels)

    print("  Insight: The store can now target each segment with tailored")
    print("  promotions — coupons for budget shoppers, loyalty rewards for")
    print("  regulars, and exclusive offers for premium customers.")


# ══════════════════════════════════════════════════════════════════════════════ #
#  2. ANOMALY DETECTION IN NETWORK TRAFFIC                                      #
# ══════════════════════════════════════════════════════════════════════════════ #

def example_anomaly_detection():
    print_header(
        "Anomaly Detection (Network Security)",
        "A server logs response times (ms) for every request.\n"
        "  Normal traffic has a typical response time, but a small fraction\n"
        "  of requests are malicious (DDoS probes, SQL injection attempts)\n"
        "  and cause unusually high latencies. EM separates the two groups\n"
        "  so the security team can set an automatic alert threshold."
    )

    true_means = [45.0, 320.0]         # normal vs suspicious latency (ms)
    true_stds = [12.0, 60.0]
    true_weights = [0.92, 0.08]        # 92% normal, 8% suspicious

    data, labels = generate_mixture_data(true_means, true_stds, true_weights,
                                         n_samples=5000, seed=21)
    # Clip negative latencies (latency can't be < 0)
    data = np.maximum(data, 1.0)

    weights, means, stds, lls = expectation_maximization(data, n_components=2)

    true_params = list(zip(true_means, true_stds, true_weights))
    print(f"  Requests simulated: {len(data)}")
    print(f"  EM iterations:      {len(lls)}\n")
    print("  Results (latency in ms):\n")
    print_comparison(true_params, means, stds, weights, labels)

    # Derive a practical threshold
    order = np.argsort(means)
    normal_idx = order[0]
    threshold = means[normal_idx] + 3 * stds[normal_idx]
    print(f"  Suggested alert threshold: {threshold:.0f} ms")
    print(f"  (3 standard deviations above normal traffic mean)")
    flagged = np.sum(data > threshold)
    print(f"  Requests that would be flagged: {flagged} / {len(data)} "
          f"({100*flagged/len(data):.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════ #
#  3. SPECIES CLASSIFICATION (BIOLOGY)                                          #
# ══════════════════════════════════════════════════════════════════════════════ #

def example_species_classification():
    print_header(
        "Species Classification (Biology)",
        "A botanist measures petal lengths (cm) of flowers collected in a\n"
        "  meadow. She suspects there are two sub-species mixed together\n"
        "  but can't tell them apart by eye. EM finds the two groups from\n"
        "  measurements alone — no labels needed."
    )

    true_means = [1.5, 4.8]      # petal length (cm) for species A vs B
    true_stds = [0.3, 0.6]
    true_weights = [0.45, 0.55]

    data, labels = generate_mixture_data(true_means, true_stds, true_weights,
                                         n_samples=500, seed=33)

    weights, means, stds, lls = expectation_maximization(data, n_components=2)

    true_params = list(zip(true_means, true_stds, true_weights))
    print(f"  Flowers measured: {len(data)}")
    print(f"  EM iterations:    {len(lls)}\n")
    print("  Results (petal length in cm):\n")
    print_comparison(true_params, means, stds, weights, labels)

    print("  Insight: The botanist can now assign each flower to a likely")
    print("  sub-species based on which Gaussian component explains its")
    print("  petal length best — a soft, probabilistic classification.")


# ══════════════════════════════════════════════════════════════════════════════ #
#  4. SENSOR CALIBRATION (IoT / Manufacturing)                                  #
# ══════════════════════════════════════════════════════════════════════════════ #

def example_sensor_calibration():
    print_header(
        "Sensor Calibration (Manufacturing / IoT)",
        "A factory has temperature sensors along a production line. Due to\n"
        "  a wiring mix-up, readings from two different zones (a cool zone\n"
        "  and a heated zone) are logged into the same data stream with no\n"
        "  zone labels. EM separates the two temperature distributions so\n"
        "  engineers can recalibrate each zone independently."
    )

    true_means = [22.0, 85.0]    # degrees Celsius: cool zone vs heated zone
    true_stds = [3.0, 5.0]
    true_weights = [0.60, 0.40]

    data, labels = generate_mixture_data(true_means, true_stds, true_weights,
                                         n_samples=2000, seed=44)

    weights, means, stds, lls = expectation_maximization(data, n_components=2)

    true_params = list(zip(true_means, true_stds, true_weights))
    print(f"  Readings simulated: {len(data)}")
    print(f"  EM iterations:      {len(lls)}\n")
    print("  Results (temperature in °C):\n")
    print_comparison(true_params, means, stds, weights, labels)

    print("  Insight: With the two distributions separated, engineers can")
    print("  detect if either zone drifts from its target temperature and")
    print("  trigger maintenance alerts automatically.")


# ══════════════════════════════════════════════════════════════════════════════ #
#  5. MEDICAL DIAGNOSTICS (Healthcare)                                          #
# ══════════════════════════════════════════════════════════════════════════════ #

def example_medical_diagnostics():
    print_header(
        "Medical Diagnostics (Healthcare)",
        "A hospital collects blood glucose readings (mg/dL) from a screening\n"
        "  program. The population contains healthy individuals and undiagnosed\n"
        "  pre-diabetic patients. EM separates these groups so doctors can\n"
        "  identify at-risk patients who need follow-up testing."
    )

    true_means = [90.0, 135.0]      # mg/dL: healthy vs pre-diabetic
    true_stds = [8.0, 15.0]
    true_weights = [0.75, 0.25]     # 75% healthy, 25% pre-diabetic

    data, labels = generate_mixture_data(true_means, true_stds, true_weights,
                                         n_samples=4000, seed=55)

    weights, means, stds, lls = expectation_maximization(data, n_components=2)

    true_params = list(zip(true_means, true_stds, true_weights))
    print(f"  Patients screened: {len(data)}")
    print(f"  EM iterations:     {len(lls)}\n")
    print("  Results (blood glucose in mg/dL):\n")
    print_comparison(true_params, means, stds, weights, labels)

    order = np.argsort(means)
    risk_mean = means[order[1]]
    risk_std = stds[order[1]]
    cutoff = risk_mean - 2 * risk_std
    at_risk = np.sum(data > cutoff)
    print(f"  Suggested screening cutoff: {cutoff:.0f} mg/dL")
    print(f"  Patients flagged for follow-up: {at_risk} / {len(data)} "
          f"({100*at_risk/len(data):.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════ #
#  Main                                                                         #
# ══════════════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#   REAL-WORLD EXAMPLES: Expectation-Maximization Algorithm" + " " * 9 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    example_customer_segmentation()
    example_anomaly_detection()
    example_species_classification()
    example_sensor_calibration()
    example_medical_diagnostics()

    print("\n" + "=" * 70)
    print("  All examples completed.")
    print("  In every case, EM recovered the hidden group structure from")
    print("  unlabelled data — no supervision required.")
    print("=" * 70 + "\n")
