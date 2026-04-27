"""Print WASAPI audio device info so we can target the right loopback device."""
from __future__ import annotations

import sys


def main() -> int:
    import sounddevice as sd

    print("sounddevice version:", sd.__version__)
    hostapis = sd.query_hostapis()
    for i, h in enumerate(hostapis):
        print(f"\nhostapi[{i}] name={h['name']!r} default_in={h.get('default_input_device')} default_out={h.get('default_output_device')} keys={sorted(h.keys())}")

    print("\n=== devices ===")
    for i, d in enumerate(sd.query_devices()):
        print(
            f"[{i}] hostapi={d['hostapi']} name={d['name']!r} "
            f"max_in={d['max_input_channels']} max_out={d['max_output_channels']} "
            f"default_sr={d['default_samplerate']}"
        )

    wasapi_idx = next(
        (i for i, h in enumerate(hostapis) if "WASAPI" in h["name"].upper()), None
    )
    print("\nWASAPI hostapi index:", wasapi_idx)
    if wasapi_idx is not None:
        default_out = hostapis[wasapi_idx]["default_output_device"]
        print("WASAPI default output device index:", default_out)
        if default_out >= 0:
            print("WASAPI default output device info:", sd.query_devices(default_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
