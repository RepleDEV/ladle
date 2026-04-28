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

    analysis_columns = []
    day_night_column = []
    points_column = []
    analyses = []
    for fp in filepaths:
        df = dfFromCSV(fp)
        analysis = soundf.processValues(df)

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

    anal_df = anal_df.sort_values(by=["point"])

    day_df = anal_df[anal_df["time"] == "day"]
    eve_df = anal_df[anal_df["time"] == "eve"]

    L_Aeq_day = day_df["L_Aeq"].to_numpy()
    L_Aeq_eve = eve_df["L_Aeq"].to_numpy()

    L_den = soundf.getL_den(L_Aeq_day, L_Aeq_eve)

    L_90_day = day_df["L_90"].to_numpy()
    L_90_eve = eve_df["L_90"].to_numpy()
    L_10_day = day_df["L_10"].to_numpy()
    L_10_eve = eve_df["L_10"].to_numpy()

    L_TNI = soundf.getL_TNI(L_90_day, L_90_eve, L_10_day, L_10_eve)

    if (args.kmeans and defaultSetup):
        print("Proceeding to k-means analysis.")
        kmeans.run_kmeans(L_den, L_TNI)

if __name__ == "__main__":
    main()
