# in this script output from benchmarks is processed into statistics and images for evaluation purposes
# the script is quite dirty because it is currently unclear if this will be the final viz framework

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# SETTINGS
# list of subfolder names, leave empty list to process most recent folder only
folders_to_process = []
# print all pandas columns
pd.set_option('display.max_columns', None)
num_ignored_frames = 30

# FUNCTIONS
def determine_folder_to_process():
    all_subfolders_sorted = sorted([f for f in os.listdir() if os.path.isdir(f)])
    if len(all_subfolders_sorted) == 0:
        return None
    if len(folders_to_process) == 0:
        # if nothing was listed, take the latest folder
        return [all_subfolders_sorted[-1]]
    else:
        # if folders were listed only process the existing ones
        folders_to_process_existing = []
        for folder in folders_to_process:
            if folder in all_subfolders_sorted:
                folders_to_process_existing.append(folder)
            else:
                print(f"Folder {folder} not found, skipping.")
        return folders_to_process_existing

def determine_files_to_process():
    folders = determine_folder_to_process()
    if folders is None:
        raise("Error: No folders found in the current directory.")

    # loop through the folders and find all csv files to process
    files_to_process = []
    for folder in folders:
        files = os.listdir(folder)
        for file in files:
            if file.endswith(".csv"):
                # verify if the file has a yyyymmdd-hhmmss format to be sure it is a benchmark result
                if len(file) == 19 and file[8] == "-" and file[15] == ".":
                    if file[0:8].isdigit() and file[9:15].isdigit():
                        files_to_process.append(os.path.join(folder, file))

    return files_to_process

def read_and_clean_data(file_to_process):
    # read csv
    raw_df = pd.read_csv(file_to_process, sep=",")

    # Create a copy of the raw data to avoid modifying the original DataFrame
    df = raw_df.copy()

    # add a string column camera id,  if the description contains ...cam_id_NUMBER... store NUMBER in that column
    # e.g. perception_writebuffer-cam_id_1-frame_id_26,1744882792782243,1744882792782247,4
    df["camera_id"] = df["description"].str.findall(r'cam_id_(\d+)').apply(lambda x: tuple(int(cid) for cid in x) if x else (-1,))
    df["frame_id"] = df["description"].str.findall(r'frame_id_(\d+)').apply(lambda x: tuple(int(fid) for fid in x) if x else (-1,))

    # Filter out rows where any frame_ids are less than num_ignored_frames
    df = df[df["frame_id"].apply(lambda x: all(fid >= num_ignored_frames for fid in x))]

    # add a timer_type column that is desription where only the substring before '-cam_id' is kept
    # e.g. perception_writebuffer-cam_id_1-frame_id_26,1744882792782243,1744882792782247,4
    df["timer_type"] = df["description"].str.split("cam_id").str[0]

    # if timer_type ends in '-' '-_' or '_' remove that suffix
    df["timer_type"] = df["timer_type"].str.rstrip("-_")

    return df

def write_list_to_disk(list_to_write, output_folder, filename):
    # write the list to a file
    with open(os.path.join(output_folder, filename), "w") as f:
        for item in list_to_write:
            f.write("%s\n" % item)

def generate_histogram_per_type(df, a_type, output_folder):
    print(f"Generating histogram for type: {a_type}")
    # make histogram for this type
    df_type_filtered = df[df["timer_type"] == a_type]
    plot = create_duration_histogram(df_type_filtered["duration"].tolist(), a_type, "all'")
    save_plot_to_disk(plot, output_folder, a_type, "all")

    # additionally, for the perception timers, make a histogram for this type for every possible camera id
    if "perception" in a_type:
        distinct_camera_ids = list(df_type_filtered["camera_id"].unique())
        for cam_id in distinct_camera_ids:
            df_type_and_camera_filtered = df_type_filtered[df_type_filtered["camera_id"] == cam_id]
            plot = create_duration_histogram(df_type_and_camera_filtered["duration"].tolist(), a_type, cam_id)
            save_plot_to_disk(plot, output_folder, a_type, cam_id)

def create_duration_histogram(values, a_type, cam_id):
    values = [value/1000 for value in values]
    # create the histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(values, bins=50, kde=True)
    plt.title(f"Histogram of {a_type} for camera id {str(cam_id)}")
    plt.xlabel("Duration (ms)")
    plt.ylabel("Frequency")
    return plt

def save_plot_to_disk(plot, output_folder, a_type, cam_id):
    plt.savefig(os.path.join(output_folder, f"histogram_{a_type}_cam_id_{cam_id}.png"))
    plt.close()

def calculate_pipeline_runtimes(df):
     # Create both smaller lookup DataFrame
    camread_df = df[df["timer_type"] == "perception_iteration_from_noidle_till_writebuffer"].copy()
    end_of_pipeline_df = df[df["timer_type"] == "ssm_iteration_action"].copy()


    # Step 2: Loop through ssm_iteration_action rows and find earliest matching start_times
    pipeline_runtimes = []
    in_buffer_wait_times = []

    for idx, row in end_of_pipeline_df.iterrows():
        cam_ids = row["camera_id"]
        frame_ids = row["frame_id"]
        start_times = []
        end_times = []
        for cam_id, frame_id in zip(cam_ids, frame_ids):
            match = camread_df[
                (camread_df["camera_id"] == (cam_id,)) &
                (camread_df["frame_id"] == (frame_id,))
                ]
            if not match.empty:
                start_times.append(match["start_time"].min())
                end_times.append(match["end_time"].max())
            else:
                print(f"there was a problem finding a start match for this camera id: {cam_id} and frame id: {frame_id}")
        oldest_start_time = min(start_times)
        start_wait_in_buffer = max(end_times)

        # Calculate the runtime
        run_time = row["end_time"] - oldest_start_time
        # Append the result to the list
        pipeline_runtimes.append(run_time)
        in_buffer_wait_times.append(row["start_time"] - start_wait_in_buffer)
    return pipeline_runtimes, in_buffer_wait_times

def create_pipeline_duration_histogram(pipeline_runtimes, title:str=f"Pipeline runtimes"):
    values = [value/1000 for value in pipeline_runtimes]
    # create the histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(values, bins=50, kde=True)
    plt.title(title)
    plt.xlabel("Duration (ms)")
    plt.ylabel("Frequency")
    return plt

def collect_statistical_measures_in_dataframe(df, a_type):
    """return a dataframe with a column for timer_type for each statistical measure"""
    print(f"Generating stats for type: {a_type}")
    # make histogram for this type
    df_type_filtered = df[df["timer_type"] == a_type]
    stats = calculate_statistical_measures(df_type_filtered.duration.tolist()) # dict
    dfstats = pd.DataFrame([stats])
    # add camera_id column to dfstats with value "all"
    dfstats["camera_id"] = "all"
    dfstats["type"] = a_type

    # additionally, for the perception timers, make a  histogram for this type for every possible camera id
    substats = []
    if "perception" in a_type:
        distinct_camera_ids = list(df_type_filtered["camera_id"].unique())
        for cam_id in distinct_camera_ids:
            df_type_and_camera_filtered = df_type_filtered[df_type_filtered["camera_id"] == cam_id]
            stats = calculate_statistical_measures(df_type_and_camera_filtered.duration.tolist()) # dict
            subdfstats = pd.DataFrame([stats])
            subdfstats["camera_id"] = cam_id
            subdfstats["type"] = a_type
            substats.append(subdfstats)

    # combine dfstatts and all substats into one dataframe
    if len(substats) > 0:
        subdfstats = pd.concat(substats, ignore_index=True)
        dfstats = pd.concat([dfstats, subdfstats], ignore_index=True)

    return dfstats



def calculate_statistical_measures(values):
    """return min, p10 p25 p50 p75 p90 max, average of values in list"""
    num_value = len(values)
    min_value = min(values)
    max_value = max(values)
    avg_value = sum(values) / len(values)
    sorted_values = sorted(values)
    p10_value = sorted_values[int(num_value * 0.1)]
    p25_value = sorted_values[int(num_value * 0.25)]
    p50_value = sorted_values[int(num_value * 0.5)]
    p75_value = sorted_values[int(num_value * 0.75)]
    p90_value = sorted_values[int(num_value * 0.9)]

    return {
        "num": num_value,
        "min": min_value,
        "p10": p10_value,
        "p25": p25_value,
        "p50": p50_value,
        "p75": p75_value,
        "p90": p90_value,
        "max": max_value,
        "avg": avg_value
    }

def generate_boxplots_all_cams(dfstats):
    # Generate horizontal boxplots for camera_id == "all"

    types_of_interest = [
        "perception_camread_noidle",
        "perception_humandetection",
        "perception_writebuffer",
        "perception_iteration_from_noidle_till_writebuffer",
        "pipeline_wait_in_buffer",
        "ssm_total",
        "pipeline_runtime"
    ]

    filtered = dfstats[
        dfstats["type"].isin(types_of_interest) &
        (dfstats["camera_id"] == "all")
        ].copy()

    # Convert duration from us to ms
    filtered["min"] = filtered["min"] / 1000
    filtered["p25"] = filtered["p25"] / 1000
    filtered["p50"] = filtered["p50"] / 1000
    filtered["p75"] = filtered["p75"] / 1000
    filtered["max"] = filtered["max"] / 1000

    box_data = []
    labels = []

    for _, row in filtered.iterrows():
        p25, p50, p75 = row["p25"], row["p50"], row["p75"]
        min_val = row["min"]
        max_val = row["max"]

        box_data.append({
            'whislo': min_val,
            'q1': p25,
            'med': p50,
            'q3': p75,
            'whishi': max_val,
            'fliers': []
        })
        labels.append(row["type"])

    # Plotting
    fig, ax = plt.subplots(figsize=(10, len(box_data) * 1.2))  # Adjust height per boxplot
    ax.bxp(box_data, showfliers=False, vert=False)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Duration (ms)")
    ax.set_title("Boxplot per type (camera_id == 'all')")

    # Add text annotations for min, max, and median
    # Add text annotations for min, max, and median
    for i, box in enumerate(box_data):
        y_position = i + 1  # Corresponds to the boxplot's y-axis position

        # Num annotation
        ax.text(
            box['whishi'], y_position + 0.2, f"Num: {filtered.iloc[i]['num']}",
            va='top', ha='center', fontsize=8, color='black'
        )
        # Min annotation
        ax.text(
            box['whislo'], y_position - 0.32, f"Min: {box['whislo']:.2f}",
            va='top', ha='center', fontsize=8, color='blue'
        )
        # Max annotation
        ax.text(
            box['whishi'], y_position - 0.2, f"Max: {box['whishi']:.2f}",
            va='top', ha='center', fontsize=8, color='red'
        )
        # Median annotation
        ax.text(
            box['med'], y_position - 0.44, f"Med: {box['med']:.2f}",
            va='top', ha='center', fontsize=8, color='green'
        )
    plt.tight_layout()
    return plt


def generate_boxplots_all_cams2(
        dfstats,
        types_of_interest=None,
        save_path=None,):
    # generate a composite graph where we show multiple boxplots one next to the other,

    # Filter to only relevant types

    if types_of_interest is None:
        types_of_interest = [
            "perception_camread_noidle",
            "perception_humandetection",
            "perception_writebuffer",
            "perception_iteration_from_noidle_till_writebuffer",
            "ssm_iteration_calculation",
            "pipeline_runtime"
        ]
    filtered = dfstats[dfstats["type"].isin(types_of_interest) &
                       (dfstats["camera_id"] == "all")].copy()

    # Convert duration from us to ms
    filtered["min"] = filtered["min"] / 1000
    filtered["p25"] = filtered["p25"] / 1000
    filtered["p50"] = filtered["p50"] / 1000
    filtered["p75"] = filtered["p75"] / 1000
    filtered["max"] = filtered["max"] / 1000

    # Prepare boxplot data from p25, p50, p75, min (approximated as p25 - IQR), max, etc.
    box_data = []
    labels = []

    for _, row in filtered.iterrows():
        # Construct an artificial boxplot distribution from percentiles
        p25, p50, p75 = row["p25"], row["p50"], row["p75"]
        min_val = row["min"]
        max_val = row["max"]

        # Build a box dictionary compatible with matplotlib
        box_data.append({
            'whislo': min_val,
            'q1': p25,
            'med': p50,
            'q3': p75,
            'whishi': max_val,
            'fliers': []
        })
        labels.append(f"{row['type']}")

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    boxplot_dict = ax.bxp(box_data, showfliers=False)
    ax.set_xticklabels(labels)
    ax.set_title("Boxplot per type")
    ax.set_ylabel("Duration (ms)")

    # Add text annotations at the right side of each box
    for i, (box_patch, box) in enumerate(zip(boxplot_dict['boxes'], box_data)):
        path = box_patch.get_path().vertices
        x_pos = path[:, 0].mean()
        x_offset = 0.1

        ax.text(x_pos + x_offset, box['whislo'], f"Min: {box['whislo']:.2f}", va='center', ha='left', fontsize=8, color='blue')
        ax.text(x_pos + x_offset, box['whishi'], f"Max: {box['whishi']:.2f}", va='center', ha='left', fontsize=8, color='red')
        ax.text(x_pos + x_offset, box['med'], f"Med: {box['med']:.2f}", va='center', ha='left', fontsize=8, color='green')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path))
        plt.close()
    return plt

def process_of_data(result_file_to_process : str):
    output_folder = os.path.dirname(result_file_to_process)

    df = read_and_clean_data(result_file_to_process)
    df.to_csv(os.path.join(output_folder, "cleaned_data.csv"), index=False)

    distinct_types = list(df["timer_type"].unique())
    write_list_to_disk(distinct_types, output_folder, "distinct_timer_types.txt")

    for a_type in distinct_types:
        generate_histogram_per_type(df, a_type, output_folder)

    print("Calculating total pipeline runtime...")
    runtimes, wait_in_buffer = calculate_pipeline_runtimes(df)
    plot = create_pipeline_duration_histogram(runtimes, f"Pipeline runtimes")
    save_plot_to_disk(plot, output_folder, "pipeline_runtime", "all")
    plot = create_pipeline_duration_histogram(wait_in_buffer, f"Pipeline wait in buffer")
    save_plot_to_disk(plot, output_folder, "pipeline_wait_in_buffer", "all")

    print("Calculating statistics on all types of measurements...")
    list_of_stat_dfs = []
    for a_type in distinct_types:
        new_df = collect_statistical_measures_in_dataframe(df, a_type)
        list_of_stat_dfs.append(new_df)
    # add runtime stats too
    runtime_stats = calculate_statistical_measures(runtimes)
    runtime_stats = pd.DataFrame([runtime_stats])
    # add camera_id column to dfstats with value "all"
    runtime_stats["camera_id"] = "all"
    runtime_stats["type"] = "pipeline_runtime"
    list_of_stat_dfs.append(runtime_stats)

    wait_in_buffer_stats = calculate_statistical_measures(wait_in_buffer)
    wait_in_buffer_stats = pd.DataFrame([wait_in_buffer_stats])
    wait_in_buffer_stats["camera_id"] = "all"
    wait_in_buffer_stats["type"] = "pipeline_wait_in_buffer"
    list_of_stat_dfs.append(wait_in_buffer_stats)
    dfstats = pd.DataFrame()
    # combine all dfstats into one bigdataframe:
    if len(list_of_stat_dfs) > 0:
        dfstats = pd.concat(list_of_stat_dfs, ignore_index=True)
    # save this to disk
    dfstats.to_csv(os.path.join(output_folder, "all_stats.csv"), index=False)

    # create boxplots of different pipeline component runtimes
    generate_boxplots_all_cams(dfstats)
    plt.savefig(os.path.join(output_folder, f"boxplot_pipeline_components.png"))
    plt.close()

# MAIN
if __name__ == "__main__":
    # change the working directory for the execution to the location of this file
    os.chdir(os.path.dirname(__file__))

    files_to_process = determine_files_to_process()
    print("files_to_process:")
    print(files_to_process)

    for file_to_process in files_to_process:
        process_of_data(result_file_to_process=file_to_process)













