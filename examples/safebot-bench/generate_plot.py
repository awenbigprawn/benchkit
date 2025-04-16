import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import pathlib
from pathlib import Path
from benchkit.utils.types import PathType

def _generate_timestamp() -> str:
    result = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return result

def generate_barplot_from_json_file(json_file_path: PathType, output_dir: PathType = "/tmp/figs") -> None:
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    df = pd.DataFrame(json_data)
    generate_barplot_from_dataframe(df=df, output_dir=output_dir)

def generate_barplot_from_dataframe(df: pd.DataFrame, output_dir: PathType = "/tmp/figs") -> None:
    camera_num = 0
    camera_name_list = []
    for col in df.columns:
        if col.startswith('camera'):
            parts = col.split('_')
            if len(parts) > 1 and parts[0] not in camera_name_list:
                camera_name_list.append(parts[0])
                camera_num += 1

    print(f"Camera Number: {camera_num}")
    print(f"Camera Name List: {camera_name_list}")

    camera_relate_module_strings = [
        "idle_waitframe",
        "preprocess_frame",
        "human_detect",
        "bufferwrite",
    ]
    other_module_strings = [
        "set_ssm_speed",
    ]
    statistic_strings = [
        "avg",
        "min",
        "max"
    ]

    columns_to_plot = []
    for camera_i in camera_name_list:
        for camera_relate_module_string in camera_relate_module_strings:
            for statistic_string in statistic_strings:
                columns_to_plot.append(f"{camera_i}_{camera_relate_module_string}_{statistic_string}")
    other_columns_to_plot = []
    for other_module_string in other_module_strings:
        for statistic_string in statistic_strings:
            other_columns_to_plot.append(f"{other_module_string}_{statistic_string}")


    camera_related_df = df[columns_to_plot]
    other_selected_df = df[other_columns_to_plot]

    long_df = pd.melt(camera_related_df, var_name='Metric', value_name='Value')
    long_df['Camera'] = long_df['Metric'].apply(lambda x: x.split('_')[0])
    long_df['Module'] = long_df['Metric'].apply(lambda x: '_'.join(x.split('_')[1:-1]))
    long_df['Stat'] = long_df['Metric'].apply(lambda x: x.split('_')[-1])

    stats_df = long_df.pivot_table(
        index=['Module', 'Camera'],
        columns='Stat',
        values='Value'
    ).reset_index()

    stats_df.columns = ['Module', 'Camera', 'avg', 'max', 'min']

    stats_df['yerr_lower'] = stats_df['avg'] - stats_df['min']
    stats_df['yerr_upper'] = stats_df['max'] - stats_df['avg']

    # Create a separate DataFrame for the other modules
    long_other_df = pd.melt(other_selected_df, var_name='Metric', value_name='Value')
    long_other_df['Module'] = long_other_df['Metric'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    long_other_df['Stat'] = long_other_df['Metric'].apply(lambda x: x.split('_')[-1])

    other_stats_df = long_other_df.pivot_table(
        index=['Module'],
        columns='Stat',
        values='Value'
    ).reset_index()

    other_stats_df.columns = ['Module', 'avg', 'max', 'min']

    other_stats_df['yerr_lower'] = other_stats_df['avg'] - other_stats_df['min']
    other_stats_df['yerr_upper'] = other_stats_df['max'] - other_stats_df['avg']

    plt.figure(figsize=(16, 10))
    barplot = sns.barplot(
        data=stats_df,
        x='Module',
        y='avg',
        hue='Camera',
        order=camera_relate_module_strings,
        ci=None
    )

    x_coords = []
    for patch in barplot.patches:
        x_coords.append(patch.get_x() + patch.get_width() / 2)

    sorted_data = stats_df.set_index(['Module', 'Camera']).loc[
        [(module, camera) for camera in camera_name_list for module in camera_relate_module_strings]
    ].reset_index()

    for i, (_, row) in enumerate(sorted_data.iterrows()):
        barplot.errorbar(
            x=x_coords[i],
            y=row['avg'],
            yerr=[[row['yerr_lower']], [row['yerr_upper']]],
            fmt='none',
            ecolor='black',
            capsize=5
        )

    # Add bars for other modules
    num_bars_per_camera = len(camera_relate_module_strings)
    positions = [num_bars_per_camera + i for i in range(len(other_module_strings))]

    for idx, pos in enumerate(positions):
        module_row = other_stats_df.iloc[idx]
        barplot.bar(pos, module_row['avg'], color='gray', label=f"{module_row['Module']}")
        barplot.errorbar(
            x=pos,
            y=module_row['avg'],
            yerr=[[module_row['yerr_lower']], [module_row['yerr_upper']]],
            fmt='none',
            ecolor='black',
            capsize=5
        )

    # Update x-ticks to include other modules
    new_xticks = list(barplot.get_xticklabels())
    other_labels = [f"{mod}" for mod in other_module_strings]
    all_xticks = new_xticks + other_labels
    barplot.set_xticks(range(len(all_xticks)))
    barplot.set_xticklabels(all_xticks, rotation=45)

    plt.title('Comparison of Modules Execution Time')
    plt.xlabel('Module')
    plt.ylabel('Exec time (us)')
    plt.legend(title="camera", loc='upper right')
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

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.resolve()
    results_dir = base_dir / "results/benchmark_SAFEBOT_cppdemo_safebot_campaign_20250412_200506_721668/run-1"
    generate_plots_for_all_results(search_dir=results_dir, target_pattern="experiment_results.json")



