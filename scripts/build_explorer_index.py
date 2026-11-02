"""Build the recorded-data explorer index from local files (no download, no model training beyond the deterministic frozen refit).

    python -m scripts.build_explorer_index

Writes data/processed/explorer/ (gitignored). Stops without writing the manifest if the frozen threshold, the frozen Incident 2 smoothed
score or the frozen episode counts are not reproduced.
"""
from backend.app.explorer import INDEX, build_index


def main() -> None:
    m = build_index(INDEX)
    print(f"explorer index written to {INDEX}")
    print(f"threshold {m['threshold']:.2f}; bins {m['bins']}; raw rows {m['raw_rows']:,}; raw zip matches config: {m['raw_zip_matches_config']}")
    for k, (got, want) in m["reproduced"].items():
        print(f"  reproduced {k}: {got} (frozen {want})")


if __name__ == "__main__":
    main()
