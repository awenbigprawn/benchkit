# Copyright (C) 2024 Vrije Universiteit Brussel. All rights reserved.
# SPDX-License-Identifier: MIT

import pathlib
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, SupportsIndex
import numpy as np

from pythainer.examples.runners import camera_runner, gui_runner, personal_runner, gpu_runner
from pythainer.runners import ConcreteDockerRunner, DockerRunner
from deps.cpp.docker.cppdemo_container import get_cppdemo_builder
from generate_plot import generate_plots_for_all_results

from benchkit.benchmark import Benchmark, CommandAttachment, PostRunHook, PreRunHook
from benchkit.campaign import CampaignCartesianProduct
from benchkit.commandwrappers import CommandWrapper
from benchkit.commandwrappers.perf import PerfStatWrap
from benchkit.commandwrappers.strace import StraceWrap
from benchkit.communication.docker import DockerCommLayer
from benchkit.platforms import Platform, get_current_platform
from benchkit.sharedlibs import SharedLib
from benchkit.utils.dir import caller_dir
from benchkit.utils.types import PathType

DOCKER = True
NB_RUNS = 1
NUM_CAMERAS = 5
DURATION_DEMO = 30

def edit_output(exec_times, print_name):
    if exec_times:
        average = np.sum(exec_times) / len(exec_times)
        maximum = np.max(exec_times)
        minimum = np.min(exec_times)
        std_div = np.std(exec_times)
        median = np.median(exec_times)
    else:
        average = 0
        maximum = 0
        minimum = 0
        std_div = 0
        median = 0

    if not print_name:
        print(f"warning: {print_name} is empty")

    return {
        f"{print_name}_avg": float(average),
        f"{print_name}_max": float(maximum),
        f"{print_name}_min": float(minimum),
        f"{print_name}_std": float(std_div),
        f"{print_name}_med": float(median),
    }


class SafebotBench(Benchmark):
    def __init__(
        self,
        src_dir: PathType,
        command_wrappers: Iterable[CommandWrapper] = (),
        command_attachments: Iterable[CommandAttachment] = (),
        shared_libs: Iterable[SharedLib] = (),
        pre_run_hooks: Iterable[PreRunHook] = (),
        post_run_hooks: Iterable[PostRunHook] = (),
        platform: Platform | None = None,
    ) -> None:
        super().__init__(
            command_wrappers=command_wrappers,
            command_attachments=command_attachments,
            shared_libs=shared_libs,
            pre_run_hooks=pre_run_hooks,
            post_run_hooks=post_run_hooks,
        )

        if platform is not None:
            self.platform = platform  # TODO Warning! overriding upper class platform

        # Use pass as a placeholder if method's body is not yet implemented
        self._src_dir = src_dir
        print(src_dir)
        self._bench_src_path = pathlib.Path(src_dir)

    def prebuild_bench(self, **kwargs):
        make_command = ["./build.sh", "--gpu"]
        self.platform.comm.shell(command=make_command, current_dir=self._src_dir)
        pass

    def build_bench(self, **kwargs) -> None:
        pass

    def single_run(
        self,
        duration_seconds: int = DURATION_DEMO,
        **kwargs,
    ) -> str:
        environment = self._preload_env(duration_seconds=duration_seconds, **kwargs)
        run_command = [
            # "./home_position",
            "./multicam_separate_pipeline",
            f"{duration_seconds}",
        ]
        wrapped_run_command, wrapped_environment = self._wrap_command(
            run_command=run_command,
            environment=environment,
            **kwargs,
        )

        output = self.run_bench_command(
            run_command=run_command,
            wrapped_run_command=wrapped_run_command,
            current_dir=(self._src_dir / "build-gpu").resolve(),
            environment=environment,
            wrapped_environment=wrapped_environment,
            print_output=False,
        )

        return output

    def parse_output_to_results(
        self,
        command_output: str,
        build_variables: Dict[str, Any],
        # run_variables: Dict[str, Any],
        benchmark_duration_seconds: int,
        record_data_dir: PathType,
        **kwargs,
    ):
        output = {}
        camera_relate_module_strings = [
            "driver_get_frame",
            "human_detection",
            "buffer_write",
            "perception_loop",
        ]
        other_module_strings = [
            "set_ssm_speed",
        ]
        for camera_module in camera_relate_module_strings:
            for camera_i in range(NUM_CAMERAS):
                matches_exec_time = re.findall(
                    rf"camera_{camera_i}_loop_(\d+)_{camera_module} takes: (\d+)", command_output)
                if not matches_exec_time:
                    continue
                loop_times = [int(item[0]) for item in matches_exec_time]
                exec_times = [int(item[1]) for item in matches_exec_time]
                if loop_times[-1] != len(exec_times):
                    print(f"warning: camera_{camera_i}_{camera_module}: exec_times[-1] = {loop_times[-1]} != {len(exec_times)} = len(exec_times) ")
                output.update(edit_output(exec_times, f"camera_{camera_i}_{camera_module}"))
        for module in other_module_strings:
            matches_exec_time = re.findall(
                rf"{module} takes: (\d+)", command_output)
            if not matches_exec_time:
                continue
            exec_times = [int(item) for item in matches_exec_time]
            output.update(edit_output(exec_times, module))

        return output

    def get_build_var_names(self) -> List[str]:
        return []

    def get_run_var_names(self) -> List[str]:
        return [
            "duration_seconds",
        ]

    @property
    def bench_src_path(self) -> pathlib.Path:
        return self._bench_src_path


def get_docker_platform(
    gpu_enabled: bool,
    host_cppdemo_dir: PathType,
    docker_cppdemo_path: PathType,
    image_name: str,
) -> Platform:
    docker_builder = get_cppdemo_builder(
        image_name=image_name,
        gpu_enabled=gpu_enabled,
    )
    docker_builder.build()

    docker_runner = ConcreteDockerRunner(
        image=image_name,
        name=image_name,
        environment_variables={},
        volumes={
            f"{host_cppdemo_dir}": f"{docker_cppdemo_path}",
        },
        devices=[],
        network="host",
        workdir=docker_cppdemo_path,
    )
    docker_runner |= camera_runner()
    docker_runner |= gui_runner()
    if gpu_enabled:
        docker_runner |= gpu_runner()
    docker_runner |= personal_runner()
    volume_run_user = {
        "/run/user/1000": "/run/user/1000",
        "/run/dbus/system_bus_socket": "/run/dbus/system_bus_socket",
        "/etc/localtime": "/etc/localtime:ro",
        "/etc/timezone": "/etc/timezone:ro",
        "safebotModelBuilds": "/home/user/model-builds",
    }
    docker_runner |= DockerRunner(
        volumes=volume_run_user
    )

    comm = DockerCommLayer(docker_runner=docker_runner)
    platform = Platform(comm_layer=comm)

    return platform


def main() -> None:
    command_wrappers = []

    safebot_bench_dir = Path(__file__).resolve().parent.resolve()
    host_cppdemo_dir = Path(safebot_bench_dir.resolve() / "deps/cpp")
    print(host_cppdemo_dir)
    docker_cppdemo_path = Path("/home/user/workspace/cppdemo/")
    image_name = "cppdemo"
    platform = get_docker_platform(
        gpu_enabled = True,
        host_cppdemo_dir = host_cppdemo_dir,
        docker_cppdemo_path = docker_cppdemo_path,
        image_name=image_name,
    )

    safebot_benchmark = SafebotBench(
        src_dir=docker_cppdemo_path,
        command_wrappers=command_wrappers,
        platform=platform,
    )

    campaign = CampaignCartesianProduct(
        name="safebot_campaign",
        benchmark=safebot_benchmark,
        nb_runs=NB_RUNS,
        variables={},
        constants=None,
        debug=False,
        gdb=False,
        enable_data_dir=True,
        benchmark_duration_seconds=DURATION_DEMO,
    )
    campaign.run()
    print(campaign.base_data_dir())

    results_dir = Path(__file__).resolve().parent.resolve() / "results"
    generate_plots_for_all_results(search_dir=results_dir, target_pattern="experiment_results.json")


if __name__ == "__main__":
    main()
