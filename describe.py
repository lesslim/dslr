import argparse

import bears as br


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Describe a dataset")
    argparser.add_argument("filename")
    args = argparser.parse_args()

    try:
        df = br.DataFrame.read_csv(args.filename)[1:]
        if isinstance(df, br.DataFrame):
            df.describe()
        else:
            print("Invalid number of columns")
    except Exception as e:
        print(f"Oof, the following error happened: {e.args}")
