"""Print a JSON audit of one generated dataset split."""

import argparse
import json

from hh_self_discovery.audit import audit_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/generated/pilot")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    args = parser.parse_args()
    print(json.dumps(audit_dataset(args.data, args.split), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

