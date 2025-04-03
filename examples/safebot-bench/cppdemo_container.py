#!/usr/bin/env python3

from pathlib import Path
from pythainer.examples.installs import rtde_lib_install_from_src, \
                                        realsense2_lib_install_from_src, \
                                        opencv_lib_install_from_src, \
                                        tensor_rt_lib_install_from_deb,\
                                        cudnn_lib_install_from_deb
from pythainer.builders import DockerBuilder, UbuntuDockerBuilder
from pythainer.examples.builders import get_user_gui_builder
from pythainer.examples.runners import camera_runner, gui_runner, gpu_runner, personal_runner
from pythainer.runners import ConcreteDockerRunner, DockerRunner
import sys


def onrobot_vgc10_dependencies(
    builder: DockerBuilder,
) -> None:
    """
    Adds dependencies required for the OnRobot VGC10.

    @param builder: The DockerBuilder instance to which the dependencies will be added.
    """
    builder.add_packages(packages=["libxmlrpc-c++8-dev"])


def install_packages_for_opencv(
    builder: DockerBuilder,
) -> None:
    """
    Installs additional packages required for building OpenCV.

    @param builder: The DockerBuilder instance to which the packages will be added.
    """
    builder.add_packages(packages=[
        "libtbb-dev",
        "libgtk2.0-dev",
        "libgtk-3-dev",
        "pkg-config",
    ])


def cicd_tools(
    builder: DockerBuilder,
    lib_dir: str,
) -> None:
    """
    Installs tools required for CI/CD processes, such as clang-format.

    @param builder: The DockerBuilder instance to which the tools will be added.
    @param lib_dir: The directory where additional libraries will be installed.
    """
    builder.desc("Dependency to run formatters & tests")
    # TODO Package not available on Ubuntu 22.04, so we need:
    llvm_version = 19
    builder.run_multiple(
        commands=[
            f"cd {lib_dir}",
            f"mkdir llvm-{llvm_version}",
            f"cd llvm-{llvm_version}",
            "wget https://apt.llvm.org/llvm.sh",
            "chmod +x llvm.sh",
            f"sudo ./llvm.sh {llvm_version}"
        ]
    )
    builder.root()
    builder.add_packages(packages=[
        f"clang-format-{llvm_version}",
    ])
    builder.run_multiple(
        commands=[
            f"sudo ln /usr/bin/clang-format-{llvm_version} /usr/bin/clang-format",
            f"sudo ln /usr/bin/git-clang-format-{llvm_version} /usr/bin/git-clang-format",
        ]
    )


def get_cppdemo_builder(
    image_name: str,
    gpu_enabled: bool,
) -> UbuntuDockerBuilder:
    """
    Creates and configures the Docker builder for the cppdemo container.

    @param image_name: The name of the Docker image to be built.
    @param gpu_enabled: Boolean indicating whether GPU support is enabled.
    @return: Configured UbuntuDockerBuilder instance.
    """
    base_image = "nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04" if gpu_enabled else "ubuntu:22.04"

    user_name = "user"
    work_dir = "/home/${USER_NAME}/workspace"
    lib_dir = f"{work_dir}/libraries"
    debug_mode = False

    docker_builder = get_user_gui_builder(
        image_name=image_name,
        base_ubuntu_image=base_image,
        user_name=user_name,
        lib_dir=lib_dir,
    )
    docker_builder.space()

    docker_builder.desc("Dependency for the image pipeline demo")
    docker_builder.add_packages(
        packages=[
            "freeglut3",
            "freeglut3-dev",
            "libpcl-dev",
            "libopencv-dev",
            "libopen3d-dev",
            "usbutils",
            "at-spi2-core",
            "libcanberra-gtk-module",
            "libcanberra-gtk3-module",
            "v4l-utils",
        ]
    )
    docker_builder.space()

    docker_builder.desc("Install gcc & g++")
    docker_builder.add_packages(
        packages=[
            "gcc-12",
            "g++-12"
        ]
    )
    docker_builder.space()

    if gpu_enabled:
        docker_builder.desc("Install TensorRT 10 from wget .deb")
        tensor_rt_lib_install_from_deb(builder=docker_builder)
        docker_builder.space()

        docker_builder.desc("Install cudnn from .deb")
        cudnn_lib_install_from_deb(builder=docker_builder)

    install_packages_for_opencv(builder=docker_builder)
    docker_builder.space()

    docker_builder.desc("OnRobot VGC10 dependencies")
    onrobot_vgc10_dependencies(
        builder=docker_builder,
    )
    docker_builder.space()

    rtde_lib_install_from_src(
        builder=docker_builder,
        workdir=lib_dir,
        debug=debug_mode,
    )
    docker_builder.space()

    realsense2_lib_install_from_src(
        builder=docker_builder,
        workdir=lib_dir,
        debug=debug_mode,
    )
    docker_builder.space()

    docker_builder.root()
    docker_builder.add_packages(packages=[
        "python3", "python3-venv", "python3-pip", "python3-wheel",
        "python3-numpy",
    ])
    docker_builder.user()
    docker_builder.space()

    opencv_cmake_options = {
        "BUILD_TESTS": "OFF",
        "BUILD_EXAMPLES": "OFF",
        "WITH_NVCUVID": "OFF",
        "WITH_NVCUVENC": "OFF",
    }
    if gpu_enabled:
        opencv_cmake_options |= {
            "CUDA_ARCH_BIN": "8.6",
            "WITH_CUDNN": "ON",
            "OPENCV_DNN_CUDA": "ON",
            "BUILD_opencv_cudaarithm": "ON",
        }
    opencv_lib_install_from_src(
        builder=docker_builder,
        workdir=work_dir,
        debug=False,
        commit_main="4.10.0",
        commit_contrib="4.10.0",
        extra_cmake_options=opencv_cmake_options,
    )
    docker_builder.space()

    cicd_tools(
        builder=docker_builder,
        lib_dir=lib_dir,
    )
    docker_builder.space()

    docker_builder.user()
    docker_builder.workdir(path=work_dir)

    return docker_builder


def buildrun(
    batch: bool,
    gpu_enabled: bool,
) -> None:
    """
    Builds and runs the Docker container.

    @param batch: Boolean indicating whether to run in batch mode.
    @param gpu_enabled: Boolean indicating whether GPU support is enabled.
    """
    print("BuildRunning:")
    print(f"  batch={batch}")
    print(f"  gpu_enabled={gpu_enabled}")

    docker_script_dir = Path(__file__).resolve().parent.resolve()
    host_cppdemo_dir = docker_script_dir.parent.resolve()
    docker_cppdemo_path = Path("/home/user/workspace/cppdemo/")

    image_name = "cppdemo"
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
    }
    docker_runner |= DockerRunner(
        volumes=volume_run_user
    )

    cmd = docker_runner.get_command()
    print(" ".join(cmd))

    docker_runner.generate_script()

    if not batch:
        docker_runner.run()


def main() -> None:
    batch = False
    gpu = True
    argv = sys.argv
    if len(argv) > 1:
        batch = any("batch" in arg for arg in argv)
        gpu = any("--gpu" in arg for arg in argv)
        if not gpu:
            gpu = not any("--no-gpu" in arg for arg in argv)
    buildrun(
        batch=batch,
        gpu_enabled=gpu,
    )


if __name__ == "__main__":
    main()
