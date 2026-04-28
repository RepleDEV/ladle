import argparse

DEFAULT_DATA_DIRECTORY = "./data/";
def parseArgs():
    # Setup parser
    parser = argparse.ArgumentParser()
    
    # Setup 
    parser.add_argument("--ddir", help="set data directory for default setup (defaults to ./data/)", default=DEFAULT_DATA_DIRECTORY)
    # parser.add_argument("-p", "--point", help="point number for default setup", type=int)
    parser.add_argument("-f", "--filepath", type=str, help="filepath to read")
    # parser.add_argument("-o", "--output", type=str, help="file output")

    parser.add_argument("--kmeans", action="store_true", help="run k means analysis")

    args = parser.parse_args()

    return args
