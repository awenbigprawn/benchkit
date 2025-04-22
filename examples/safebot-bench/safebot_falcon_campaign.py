# Copyright (C) 2024 Vrije Universiteit Brussel. All rights reserved.
# SPDX-License-Identifier: MIT

import pathlib
from imp import source_from_cache
from pathlib import Path
import re
import glob, os, shutil
from typing import Any, Dict, Iterable, List, SupportsIndex
import pandas as pd


from pythainer.examples.runners import camera_runner, gui_runner, personal_runner, gpu_runner
from pythainer.runners import ConcreteDockerRunner, DockerRunner
from deps.cpp.docker.cppdemo_container import get_cppdemo_builder
from generate_plot import generate_dataframe_from_csv_file, generate_barplot_from_dataframe
from analyse_benchmark_data import (read_and_clean_data,
                                    calculate_pipeline_runtimes,
                                    calculate_statistical_measures,
                                    process_of_data,
                                    generate_boxplots_all_cams2)

from benchkit.benchmark import Benchmark, CommandAttachment, PostRunHook, PreRunHook, RecordResult, WriteRecordFileFunction
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
NUM_CAMERAS:SupportsIndex = 5
NUM_CPUS = 20
DURATION_DEMO = 30
RECORD_ALL_TIME = False
OLD_EXEC_TIME_ONLY = False

docker_cpp_folder_path = "/home/user/workspace/cppdemo"


class SafebotBench(Benchmark):
    def __init__(
        self,
        src_dir: PathType,
        host_cppdemo_dir: PathType,
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
        self._host_cppdemo_dir = host_cppdemo_dir
        self._bench_src_path = pathlib.Path(src_dir)

    def prebuild_bench(self, **kwargs):
        pass

    def build_bench(self, **kwargs) -> None:
        make_command = ["./build.sh", "--gpu"]
        self.platform.comm.shell(command=make_command, current_dir=self._src_dir)

        with_sched_ddl = kwargs.get("with_sched_ddl", False)

        if with_sched_ddl:
            setcap_command = ["sudo", "setcap", "cap_sys_nice+ep", "./multicam_separate_pipeline"]
            self.platform.comm.shell(command=setcap_command, current_dir=self._src_dir / "build-gpu")

    def single_run(
        self,
        duration_seconds: int = DURATION_DEMO,
        num_cameras: int = NUM_CAMERAS,
        synth_workers: int = 0,
        num_cpus: int = NUM_CPUS,
        # record_data_dir: PathType,
        # write_record_file_fun: WriteRecordFileFunction,
        **kwargs,
    ) -> str:
        environment = self._preload_env(duration_seconds=duration_seconds, **kwargs)
        run_command = [
            "taskset","-c",f"0-{num_cpus-1}",
            "./multicam_separate_pipeline",
            f"--runtime={duration_seconds}",
            f"--cameranum={num_cameras}",
            f"--synthworker={synth_workers}"
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
        run_variables: Dict[str, Any],
        benchmark_duration_seconds: int,
        record_data_dir: PathType,
        **kwargs,
    ):
        output = {}
        matches_cpp_output_path = re.findall(
            rf"Output path: (.*?)\n", command_output)

        #if matches_cpp_output_path is not empty
        if matches_cpp_output_path:
            docker_cpp_output_path = str(matches_cpp_output_path[0])
            docker_prefix = self._src_dir
            benchkit_prefix = self._host_cppdemo_dir.resolve()

            benchkit_cpp_output_path = docker_cpp_output_path.replace(str(docker_prefix), str(benchkit_prefix), 1)
            print(benchkit_cpp_output_path)
            # copy all csv files generated by cpp demo to record_data_dir
            csv_files = glob.glob(os.path.join(benchkit_cpp_output_path, '*.csv'))
            target_record_data_file = record_data_dir / 'result.csv'
            if csv_files:
                source_file = csv_files[0]
                shutil.copy(source_file, target_record_data_file)

            process_of_data(target_record_data_file)

            df = pd.read_csv(record_data_dir/'all_stats.csv', sep=",")
            types = [
                "ssm_total",
                "perception_iteration_from_noidle_till_writebuffer"
            ]
            box_plot = generate_boxplots_all_cams2(df, types_of_interest=types, save_path=record_data_dir/'perception-ssm-statistic.png')

            benchkit_output_types = [
                "pipeline_runtime",
                "pipeline_wait_in_buffer",
                "perception_iteration_from_noidle_till_writebuffer",
                "ssm_total",
            ]
            benchkit_output_statistics = [
                "num",
                "min",
                "max",
                "avg",
            ]
            for type in benchkit_output_types:
                for statistic in benchkit_output_statistics:
                    output.update(
                        {f"{type}_{statistic}": float(min(df.loc[df['type'] == type, statistic]))}
                    )
        return output

    def get_build_var_names(self) -> List[str]:
        return [
            "with_sched_ddl",
        ]

    def get_run_var_names(self) -> List[str]:
        return [
            "num_cpus",
            "duration_seconds",
            "num_cameras",
            "synth_workers",
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

def post_run_hook_barplot(
    experiment_results_lines: List[RecordResult],
    record_data_dir: PathType,
    write_record_file_fun: WriteRecordFileFunction,
):
    print(experiment_results_lines)
    df = pd.DataFrame(experiment_results_lines)
    # generate_barplot_from_dataframe(df=df, output_dir=record_data_dir/"figs")

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
        host_cppdemo_dir=host_cppdemo_dir,
        command_wrappers=command_wrappers,
        platform=platform,
        post_run_hooks=[post_run_hook_barplot],
    )

    campaign = CampaignCartesianProduct(
        name="safebot_campaign",
        benchmark=safebot_benchmark,
        nb_runs=NB_RUNS,
        variables={
            "num_cpus" : [8],
            "with_sched_ddl" : [False, True],
            "num_cameras" : [1],
            "synth_workers": [0,1,2,4,8], #,1,5,10,20],
            "runtime": [30],
        },
        constants=None,
        debug=False,
        gdb=False,
        enable_data_dir=True,
        benchmark_duration_seconds=DURATION_DEMO,
    )
    campaign.run()
    print(campaign.base_data_dir())

if __name__ == "__main__":
    main()
