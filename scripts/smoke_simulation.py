"""Generate two small trajectories and print diagnostic summaries."""

from hh_self_discovery import current_step, simulate_current_clamp, simulate_voltage_clamp, time_grid, voltage_step


def main() -> None:
    time = time_grid(80.0, 0.01)
    current = current_step(time, amplitude_uA_cm2=10.0, start_ms=10.0, stop_ms=50.0)
    current_trace = simulate_current_clamp(time, current)

    clamp_voltage = voltage_step(time, holding_mV=-80.0, step_mV=0.0, start_ms=10.0, stop_ms=40.0)
    voltage_trace = simulate_voltage_clamp(time, clamp_voltage, include_capacitive_current=False)

    print(
        "current clamp:",
        f"V=[{current_trace.voltage_mV.min():.3f}, {current_trace.voltage_mV.max():.3f}] mV",
        f"gate_range=[{min(current_trace.n.min(), current_trace.m.min(), current_trace.h.min()):.5f}, "
        f"{max(current_trace.n.max(), current_trace.m.max(), current_trace.h.max()):.5f}]",
    )
    print(
        "voltage clamp:",
        f"I=[{voltage_trace.current_uA_cm2.min():.3f}, {voltage_trace.current_uA_cm2.max():.3f}] uA/cm^2",
    )


if __name__ == "__main__":
    main()

