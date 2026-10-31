import statistics
import sys
import time

import requests


def main() -> None:
    port = sys.argv[1] if len(sys.argv) > 1 else "8100"
    url = f"http://127.0.0.1:{port}/invocations"
    body = {
        "dataframe_records": [
            {"montant": 8500.0, "anciennete": 3, "freq_24h": 9, "nuit": 1}
        ]
    }
    response = requests.post(url, json=body, timeout=10)
    print("reponse :", response.status_code, response.json())
    for _ in range(10):
        requests.post(url, json=body, timeout=10)
    latencies = []
    for _ in range(200):
        started = time.perf_counter()
        requests.post(url, json=body, timeout=10)
        latencies.append((time.perf_counter() - started) * 1000)
    p95 = statistics.quantiles(latencies, n=20)[18]
    print(
        f"port {port} | mediane : {statistics.median(latencies):.1f} ms"
        f" | p95 : {p95:.1f} ms"
    )


if __name__ == "__main__":
    main()
