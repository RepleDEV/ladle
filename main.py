"""
Ladle, a noise data analysis program.
Copyright (C) 2026 Ragazzo Chaesa

This file is part of Ladle.

Ladle is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

Ladle is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Ladle. If not, see <https://www.gnu.org/licenses/>.
"""

import pandas as pd
import numpy as np

import functions.sound as soundf
import functions.kmeans as kmeans

import argparse

DEFAULT_DATA_DIRECTORY = "./data/";
DEFAULT_FILENAME_REGEX_PATTERN = r'.*pt([0-9]+)(?:eve)?\.csv'

def dfFromCSV(filepath: str, drop_duplicates = True):
    if (not filepath.endswith(".csv")):
        raise ValueError("File path doesn't end with .csv")
    # TODO: Add csv error checking with file contents. This assumes first line has correct column headers

    # Separator regex fixes whitespaces after comma delimiter in csv file
    df = pd.read_csv(filepath, sep=r'\s*,\s*', engine="python");
    if drop_duplicates:
        df = df.drop_duplicates(subset=["timestamp"])

    return df

def getAnalysisValues(df: pd.DataFrame):
    timestamps = pd.to_datetime(df["timestamp"]).to_list()
    timestamp_deltas = [0]

    for i in range(1, len(timestamps)):
        now = timestamps[i]
        prev = timestamps[i - 1]
        delta_ms = (now - prev).microseconds // 1000

        timestamp_deltas.append(delta_ms)

    # timestamp_mean = np.mean(np.array(timestamp_deltas))

    readings = df["reading"].to_numpy()
    delta_cumsum = np.cumsum(np.array(timestamp_deltas))
    T_ms = 5 * 60 * 1000
    total_indexes = np.argmax(delta_cumsum > T_ms) + 1

    readings = readings[:total_indexes]
    deltas = timestamp_deltas[:total_indexes]

    result = {}

    result["L_Aeq"] = soundf.getL_Aeq(readings, deltas);

    result["L_10"] = np.percentile(readings, 90)
    result["L_90"] = np.percentile(readings, 10)
    result["L_min"] = np.min(readings)
    result["L_max"] = np.max(readings)

    return result

import os
import re 
def getPointsFilepaths(ddir: str = ""):
    if not ddir:
        ddir = DEFAULT_DATA_DIRECTORY
    filenames = os.listdir(ddir)

    # Filter with regex
    p = re.compile(DEFAULT_FILENAME_REGEX_PATTERN)
    filepaths = [ ddir + f for f in filenames if p.match(f) ]
    filepaths.sort()
    return filepaths

def getPointNumber(filepath: str):
    search = re.search(DEFAULT_FILENAME_REGEX_PATTERN, filepath)
    if search:
        return int(search.group(1))

    return -1

def main():
    # Setup parser
    parser = argparse.ArgumentParser()
    
    # Setup 
    parser.add_argument("--ddir", help="set data directory for default setup (defaults to ./data/)", default=DEFAULT_DATA_DIRECTORY)
    # parser.add_argument("-p", "--point", help="point number for default setup", type=int)
    # parser.add_argument("-f", "--filepath", type=str, help="filepath to read")
    # parser.add_argument("-o", "--output", type=str, help="file output")

    parser.add_argument("--kmeans", action="store_true", help="run k means analysis")

    args = parser.parse_args()

    defaultSetup = True

    filepaths = []

    if args.filepath:
        filepaths.append(args.filepath)
        defaultSetup = False
    else:
        print("Running default setup.")
        filepaths = getPointsFilepaths(args.ddir)

        print(f"Found {len(filepaths)} files in {args.ddir}.")
    
    if len(filepaths) == 0:
        print("No filepath provided/found. Exiting.")
        return

    print("Running analysis.")

    analysis_columns = []
    day_night_column = []
    points_column = []
    analyses = []
    for fp in filepaths:
        df = dfFromCSV(fp)
        analysis = getAnalysisValues(df)

        analysis_values = list(analysis.values()) 
        data = [fp] + analysis_values

        if not len(analysis_columns):
            analysis_columns = list(analysis.keys())


        if defaultSetup:
            point_number = getPointNumber(fp)
            points_column.append(point_number)
            day_night_column.append("eve" if "eve" in fp else "day")

        analyses.append(data)

    print("Analysis finished.")

    columns = ["filepath"] + analysis_columns
    anal_df = pd.DataFrame(analyses, columns=columns)

    if len(points_column):
        anal_df = anal_df.assign(point=pd.Series(points_column).values)
    if len(day_night_column):
        anal_df = anal_df.assign(time=pd.Series(day_night_column).values)

    if (args.kmeans and defaultSetup):
        print("Proceeding to k-means analysis.")
        kmeans.run_kmeans(anal_df)

    
if __name__ == "__main__":
    main()
