import sys

from tensorboard.backend.event_processing import event_accumulator


def main() -> int:
    if len(sys.argv) < 3:
        return 1

    event_file = sys.argv[1]
    pairs = sys.argv[2:]

    acc = event_accumulator.EventAccumulator(event_file)
    acc.Reload()

    for pair in pairs:
        alias, tag = pair.split("=", 1)
        try:
            ev = acc.Scalars(tag)[-1]
        except Exception:
            continue
        print(f"{alias}\t{ev.step}\t{ev.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
