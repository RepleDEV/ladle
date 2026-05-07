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
import functions.sk.kmeans as sk_kmeans
import functions.argparser as ap

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

from pathlib import Path
from datetime import datetime
def writeProcessedValues(processed_df: pd.DataFrame, wholeday_df: pd.DataFrame, outdir: str):
    # Create dir if doesn't exist yet
    Path(outdir).mkdir(parents=True, exist_ok=True)

    now = datetime.today()
    isoformat = now.isoformat()
    
    if not outdir.endswith("/"):
        outdir = outdir + "/"

    processed_df.to_csv(outdir + f"main_{isoformat}.csv", index=False)
    wholeday_df.to_csv(outdir + f"sec_{isoformat}.csv", index=False)

from typing import List
def getProcessedDataPointsDF(filepaths: List):
    processed_values_columns = []
    day_night_column = []
    points_column = []
    data_list = []
    for fp in filepaths:
        df = dfFromCSV(fp)
        processedValues = soundf.processValues(df)

        analysis_values = list(processedValues.values()) 
        data = [fp] + analysis_values

        if not len(processed_values_columns):
            processed_values_columns = list(processedValues.keys())

        point_number = getPointNumber(fp)
        points_column.append(point_number)
        day_night_column.append("eve" if "eve" in fp else "day")

        data_list.append(data)

    columns = ["filepath"] + processed_values_columns
    proc_df = pd.DataFrame(data_list, columns=columns)

    columns = ["filepath"] + processed_values_columns
    proc_df = pd.DataFrame(data_list, columns=columns)

    proc_df = proc_df.assign(point=pd.Series(points_column).values)
    proc_df = proc_df.assign(time=pd.Series(day_night_column).values)

    proc_df = proc_df.sort_values(by=["point"])

    return proc_df

def main():
    args = ap.parseArgs()

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
    df = getProcessedDataPointsDF(filepaths)
    print("Analysis finished.")

    print(df)

    day_df = df[df["time"] == "day"]
    eve_df = df[df["time"] == "eve"]

    L_den = soundf.getL_den_from_df(day_df, eve_df)
    L_TNI = soundf.getL_TNI_from_df(day_df, eve_df)

    points_list = day_df["point"].to_list()
    wholeday_df = pd.DataFrame({
        "point": points_list,
        "L_den": L_den,
        "L_TNI": L_TNI,
    })

    print(wholeday_df)

    if (args.outdir):
        print(f"Outputting analysis to {args.outdir}")
        writeProcessedValues(df.drop(["filepath"], axis=1), wholeday_df, args.outdir)
        print("Done")

    if defaultSetup:
        match args.analysis:
            case "kmeans":
                kmeans.run_kmeans(L_den, L_TNI, wholeday_df["point"].to_list())
                print("Proceeding to k-means analysis.")
            case "sk.kmeans":
                sk_kmeans.run(np.array([L_den, L_TNI]).transpose(), points_list)

if __name__ == "__main__":
    main()
