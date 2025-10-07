import os
import argparse
import numpy as np
import csv
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tabulate_events(dpath):
    summary_iterators = [
        EventAccumulator(os.path.join(dpath, dname)).Reload()
        for dname in os.listdir(dpath)
    ]

    tags = summary_iterators[0].Tags()["scalars"]
    for it in summary_iterators:
        assert it.Tags()["scalars"] == tags

    out = defaultdict(list)
    steps = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]
        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1
            out[tag].append([e.value for e in events])

    return out, steps


def write_csv(file_path, headers, steps, values):
    with open(file_path, "w", newline="\n") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["step"] + headers)
        for step, row in zip(steps, values):
            writer.writerow([step] + row)


def to_csv(dpath):
    dirs = os.listdir(dpath)
    d, steps = tabulate_events(dpath)

    for tag, tag_values in d.items():
        file_path = get_file_path(dpath, tag)
        # tag_values is already a list of lists of floats
        write_csv(file_path, dirs, steps, tag_values)


def get_file_path(dpath, tag):
    file_name = tag.replace("/", "_") + ".csv"
    folder_path = os.path.join(dpath, "csv")
    os.makedirs(folder_path, exist_ok=True)
    return os.path.join(folder_path, file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert TensorBoard summaries to CSV files."
    )
    parser.add_argument(
        "path", type=str, help="Path to the directory containing event files."
    )
    args = parser.parse_args()

    to_csv(args.path)
