import asyncio
import json

import websockets


async def run() -> None:
    async with websockets.connect("ws://127.0.0.1:8765/ws/session") as ws:
        events: list[dict] = []
        try:
            for _ in range(25):
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=25))
                events.append(msg)
        except asyncio.TimeoutError:
            pass

    counts: dict[str, int] = {}
    for e in events:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    print("event counts:", counts)

    summary_types = sorted(
        {e["payload"].get("summary_type") for e in events if e["type"] == "summary_snapshot"}
    )
    print("summary types observed:", summary_types)


if __name__ == "__main__":
    asyncio.run(run())
