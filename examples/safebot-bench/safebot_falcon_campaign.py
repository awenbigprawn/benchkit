# Copyright (C) 2024 Vrije Universiteit Brussel. All rights reserved.
# SPDX-License-Identifier: MIT

import pathlib
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List

from pythainer.examples.runners import camera_runner, gui_runner, personal_runner, gpu_runner
from pythainer.runners import ConcreteDockerRunner, DockerRunner

from benchkit.benchmark import Benchmark, CommandAttachment, PostRunHook, PreRunHook
from benchkit.campaign import CampaignCartesianProduct
from benchkit.commandwrappers import CommandWrapper
from benchkit.commandwrappers.perf import PerfStatWrap
from benchkit.commandwrappers.strace import StraceWrap
from benchkit.communication.docker import DockerCommLayer
from benchkit.platforms import Platform, get_current_platform, get_remote_platform
from benchkit.sharedlibs import SharedLib
from benchkit.utils.dir import caller_dir
from benchkit.utils.types import PathType

DOCKER = True
NB_RUNS = 3


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

    def build_bench(self, **kwargs) -> None:
        pass

    def single_run(
        self,
        duration_seconds: int,
        **kwargs,
    ) -> str:
        environment = self._preload_env(duration_seconds=duration_seconds, **kwargs)
        run_command = [
            "./muilticam",
            f"{duration_seconds}",
        ]
        wrapped_run_command, wrapped_environment = self._wrap_command(
            run_command=run_command,
            environment=environment,
            duration_seconds=duration_seconds,
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
        run_variables: Dict[str, Any],
        benchmark_duration_seconds: int,
        record_data_dir: PathType,
        **kwargs,
    ):
        benchmark_duration_seconds = int(run_variables["benchmark duration seconds"])  # TODO?
        match = re.search(r"Final counter value: (\d+)", command_output)
        if match:
            # Extract the number from the matched group and convert it to an integer
            final_counter_value = int(match.group(1))

            # Calculate the result by dividing the final counter value by the duration
            result_per_second = final_counter_value / benchmark_duration_seconds
            output = {"throughput": result_per_second}
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
        "/etc/timezone": "/etc/timezone:ro"
    }
    docker_runner |= DockerRunner(
        volumes=volume_run_user
    )

    comm = DockerCommLayer(docker_runner=docker_runner)
    platform = Platform(comm_layer=comm)

    return platform


def main() -> None:
    command_wrappers = []
    if DOCKER:
        safebot_bench_dir = Path(__file__).resolve().parent.resolve()
        host_cppdemo_dir = safebot_bench_dir.parent.parent.parent.resolve()
        docker_cppdemo_path = Path("/home/user/workspace/cppdemo/")
        image_name = "cppdemo"
        platform = get_docker_platform(
            gpu_enabled = True,
            host_cppdemo_dir = host_cppdemo_dir,
            docker_cppdemo_path = docker_cppdemo_path,
            image_name=image_name,
        )
    else:
        docker_cppdemo_path = caller_dir()
        platform = get_current_platform()

    print(safebot_bench_dir)

    campaign = CampaignCartesianProduct(
        name="safebot_campaign",
        benchmark=SafebotBench(
            src_dir=docker_cppdemo_path,
            command_wrappers=command_wrappers,
            platform=platform,
        ),
        nb_runs=NB_RUNS,
        variables={},
        constants=None,
        debug=False,
        gdb=False,
        enable_data_dir=True,
        benchmark_duration_seconds=None,
    )


if __name__ == "__main__":
    main()
