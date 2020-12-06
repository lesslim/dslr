import bears as br
import argparse
import matplotlib.pyplot as plt  # type: ignore


def faculty_grades(df, faculty, course):
    """
    Receives grades from one faculty on one course.
    """
    return df[df["Hogwarts House"] == faculty][course].dropna()


def historgam(df, course):
    plt.figure(figsize=(12, 9))
    plt.title(f"Histogram of {course} grades among houses.")
    plt.xlabel("Grades")
    plt.ylabel("Percentage of students")
    for house in ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]:
        plt.hist(
            faculty_grades(df, house, course),
            alpha=0.3, label=house, density=True
        )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    try:
        df = br.read_csv("datasets/dataset_train.csv")
    except Exception as e:
        print("Something wrong with train file:", e.args)
        exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--one",
        "-o",
        default=False,
        action="store_true",
        help="Display right correlation.",
    )
    parser.add_argument(
        "--all",
        "-a",
        default=False,
        action="store_true",
        help="Display all correlations.",
    )

    flags = parser.parse_args()

    try:
        if flags.one or not flags.all:
            historgam(df, "Care of Magical Creatures")
        if flags.all:
            for i in df.columns[6:]:
                historgam(df, i)
    except Exception as e:
        print("Something wrong with train:", e.args)
        exit(1)
