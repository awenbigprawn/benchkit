#!/usr/bin/env python3
# Copyright (C) 2024 Vrije Universiteit Brussel. All rights reserved.
# SPDX-License-Identifier: MIT

from cgol_gpu import cgol_campaign

from benchkit.campaign import CampaignSuite
from benchkit.utils.dir import get_curdir


# Creates a campaign for a specific version of the code
def create_campaign_for_version(cgol_dir, version_nr):
    version_src_dir = cgol_dir / f"version-{version_nr}"
    campaign = cgol_campaign(
        src_dir=version_src_dir,
        build_dir=version_src_dir,
        bench_name=["time_based"],
        threads_per_block=[64],
        size=[200, 500, 1000, 2000, 4000, 6000, 8000],
        benchmark_duration_seconds=25,
        nb_runs=30,
        constants={"bench_version": f"version-{version_nr}"},
    )
    return campaign


def main() -> None:
    """Main function of the campaign script."""

    # Root directory where the Conway's Game of Life implementation is located
    cgol_dir = (get_curdir(__file__).parent / "deps/conway-game-of-life-parallel/").resolve()

    # Define the campaign for the CUDA version
    campaign_cuda = create_campaign_for_version(cgol_dir, "cuda")

    # Define the campaign suite and run the benchmarks in the suite
    campaigns = [campaign_cuda]
    suite = CampaignSuite(campaigns=campaigns)
    suite.print_durations()
    suite.run_suite()

    # Generate a graph with the results
    suite.generate_graph(
        plot_name="barplot",
        x="size",
        y="throughput",
        xlabel="Grid size (N x N)",
        ylabel="Throughput (cells updated/sec)",
        hue="bench_version",
        title="Throughput vs. grid size",
    )


if __name__ == "__main__":
    main()
