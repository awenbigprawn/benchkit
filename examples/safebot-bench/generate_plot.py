import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import pathlib
import glob
from pathlib import Path
from benchkit.utils.types import PathType

def _generate_timestamp() -> str:
    result = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return result

def generate_barplot_from_json_file(json_file_path: PathType, output_dir: PathType = "/tmp/figs") -> None:
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    df = pd.DataFrame(json_data)

    columns_to_plot = [
        "camera_0_driver_get_frame_avg",
        "camera_0_driver_get_frame_std",
        "camera_0_driver_get_frame_min",
        "camera_0_driver_get_frame_max",
        "camera_0_human_detection_avg",
        "camera_0_human_detection_std",
        "camera_0_human_detection_min",
        "camera_0_human_detection_max",
        "camera_0_buffer_write_avg",
        "camera_0_buffer_write_std",
        "camera_0_buffer_write_min",
        "camera_0_buffer_write_max",
        "camera_0_perception_loop_avg",
        "camera_0_perception_loop_std",
        "camera_0_perception_loop_min",
        "camera_0_perception_loop_max",
        "camera_1_driver_get_frame_avg",
        "camera_1_driver_get_frame_std",
        "camera_1_driver_get_frame_min",
        "camera_1_driver_get_frame_max",
        "camera_1_human_detection_avg",
        "camera_1_human_detection_std",
        "camera_1_human_detection_min",
        "camera_1_human_detection_max",
        "camera_1_buffer_write_avg",
        "camera_1_buffer_write_std",
        "camera_1_buffer_write_min",
        "camera_1_buffer_write_max",
        "camera_1_perception_loop_avg",
        "camera_1_perception_loop_std",
        "camera_1_perception_loop_min",
        "camera_1_perception_loop_max"
    ]

    selected_df = df[columns_to_plot]

    long_df = pd.melt(selected_df, var_name='Metric', value_name='Value')

    long_df['Camera'] = long_df['Metric'].apply(lambda x: x.split('_')[0])
    long_df['Module'] = long_df['Metric'].apply(lambda x: '_'.join(x.split('_')[1:-1]))
    long_df['Stat'] = long_df['Metric'].apply(lambda x: x.split('_')[-1])

    stats_df = long_df.pivot_table(
        index=['Module', 'Camera'],
        columns='Stat',
        values='Value'
    ).reset_index()

    stats_df.columns = ['Module', 'Camera', 'Avg', 'Max', 'Min', 'Std']

    avg_df = stats_df[['Module', 'Camera', 'Avg', 'Min', 'Max', 'Std']]

    plt.figure(figsize=(16, 10))
    sns.barplot(x='Module', y='Avg', hue='Camera', data=avg_df, ci=None)

    for i, row in avg_df.iterrows():
        plt.errorbar(
            x=i,
            y=row['Avg'],
            yerr=[[row['Avg'] - row['Min']], [row['Max'] - row['Avg']]],
            fmt='o',
            color='black',
            capsize=5,
            label='Min/Max' if i == 0 else ""
        )

    for i, row in avg_df.iterrows():
        plt.errorbar(
            x=i,
            y=row['Avg'],
            yerr=row['Std'],
            fmt='o',
            color='red',
            capsize=5,
            linewidth=2,
            label='Std' if i == 0 else ""
        )

    plt.title('Comparison of Camera Modules by Statistic')
    plt.xlabel('Module')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    timestamp = _generate_timestamp()
    output_path = pathlib.Path(output_dir)
    fig_path_png = output_path / f"barplot_{timestamp}.png"
    fig_path_pdf = output_path / f"barplot_{timestamp}.pdf"

    plt.savefig(fig_path_png, transparent=False)
    plt.savefig(fig_path_pdf, transparent=False)

    plt.show()
    plt.close()

def generate_plots_for_all_results(search_dir, target_pattern) -> None:
    json_files = [str(file) for file in search_dir.rglob(target_pattern)]

    for json_file_path in json_files:
        json_file_path = Path(json_file_path)
        output_dir = json_file_path.parent / "figs"
        generate_barplot_from_json_file(json_file_path, output_dir)

base_dir = Path(__file__).resolve().parent.resolve()
results_dir = base_dir / "results"
generate_plots_for_all_results(search_dir=results_dir, target_pattern="experiment_results.json")
