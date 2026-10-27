import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class FitnessTracker:

    def __init__(self, filename):
        self.filename = filename

        try:
            self.data = pd.read_csv(self.filename)

            self.data["Date"] = pd.to_datetime(
                self.data["Date"], errors="coerce"
            )

            self.data["Duration (Minutes)"] = pd.to_numeric(
                self.data["Duration (Minutes)"], errors="coerce"
            )

            self.data["Calories Burned"] = pd.to_numeric(
                self.data["Calories Burned"], errors="coerce"
            )

            self.data["Activity Type"] = (
                self.data["Activity Type"].astype("string").str.strip()
            )

            self.data = self.data.dropna(
                subset=[
                    "Date",
                    "Activity Type",
                    "Duration (Minutes)",
                    "Calories Burned"
                ]
            )

            self.data = self.data[
                (self.data["Activity Type"] != "") &
                (self.data["Duration (Minutes)"] > 0) &
                (self.data["Calories Burned"] >= 0)
            ]

            self.save_data()

        except FileNotFoundError:
            print("Dataset not found. Please create fitness_activities.csv.")
            self.data = pd.DataFrame(columns=[
                "Date",
                "Activity Type",
                "Duration (Minutes)",
                "Calories Burned"
            ])

    def save_data(self):
        self.data.to_csv(self.filename, index=False)

    def log_activity(self):
        print("\n--- Log Fitness Activity ---")

        date = input("Enter date (YYYY-MM-DD): ").strip()
        activity = input("Enter activity type: ").strip().title()

        try:
            date = pd.to_datetime(date, format="%Y-%m-%d", errors="raise")
            duration = float(input("Enter duration in minutes: "))
            calories = float(input("Enter calories burned: "))

            if not activity:
                print("Activity type cannot be empty.")
                return

            if duration <= 0 or calories < 0:
                print("Duration must be positive and calories cannot be negative.")
                return

            new_activity = pd.DataFrame([{
                "Date": date,
                "Activity Type": activity,
                "Duration (Minutes)": duration,
                "Calories Burned": calories
            }])

            self.data = pd.concat(
                [self.data, new_activity], ignore_index=True
            )

            self.save_data()
            print("Activity added successfully!")

        except ValueError:
            print("Invalid input! Check the date and enter valid numbers.")

    def calculate_metrics(self):
        if self.data.empty:
            print("No fitness data available.")
            return

        durations = self.data["Duration (Minutes)"].to_numpy()
        calories = self.data["Calories Burned"].to_numpy()

        total_duration = np.sum(durations)
        total_calories = np.sum(calories)
        average_duration = np.mean(durations)
        average_calories = np.mean(calories)

        print("\n--- Fitness Metrics ---")
        print("Total activities:", len(self.data))
        print("Total duration:", round(total_duration, 2), "minutes")
        print("Total calories burned:", round(total_calories, 2))
        print("Average duration:", round(average_duration, 2), "minutes")
        print("Average calories per activity:", round(average_calories, 2))

        self.data["Calories per Minute"] = (
            self.data["Calories Burned"] /
            self.data["Duration (Minutes)"]
        )

        print(
            "Average calories per minute:",
            round(self.data["Calories per Minute"].mean(), 2)
        )

        print("\nActivity Frequency:")
        print(self.data["Activity Type"].value_counts())

        print("\nTotal Time by Activity:")
        print(
            self.data.groupby("Activity Type")["Duration (Minutes)"]
            .sum()
        )

        daily_calories = self.data.groupby("Date")[
            "Calories Burned"
        ].sum().to_numpy()

        print(
            "\nAverage daily calories burned:",
            round(np.mean(daily_calories), 2)
        )

        if len(daily_calories) >= 2:
            first_day = daily_calories[0]
            last_day = daily_calories[-1]

            if first_day != 0:
                improvement = (
                    (last_day - first_day) / first_day
                ) * 100

                print(
                    "Change in daily calories from first to last recorded day:",
                    round(improvement, 2), "%"
                )
            else:
                print("Percentage change cannot be calculated from a zero baseline.")

    def filter_activities(self):
        if self.data.empty:
            print("No fitness data available.")
            return

        print("\n1. Filter by activity type")
        print("2. Filter by date range")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            activity = input("Enter activity type: ").strip()

            result = self.data[
                self.data["Activity Type"].str.lower()
                == activity.lower()
            ]

        elif choice == "2":
            try:
                start_date = pd.to_datetime(
                    input("Enter start date (YYYY-MM-DD): "),
                    format="%Y-%m-%d"
                )

                end_date = pd.to_datetime(
                    input("Enter end date (YYYY-MM-DD): "),
                    format="%Y-%m-%d"
                )

                if start_date > end_date:
                    print("Start date cannot be after end date.")
                    return

                result = self.data[
                    (self.data["Date"] >= start_date) &
                    (self.data["Date"] <= end_date)
                ]

            except ValueError:
                print("Invalid date format.")
                return

        else:
            print("Invalid choice.")
            return

        if result.empty:
            print("No matching activities found.")
        else:
            print("\nFiltered Activities:")
            print(result.to_string(index=False))

    def generate_report(self):
        if self.data.empty:
            print("No fitness data available.")
            return

        print("\n========== FITNESS REPORT ==========")
        print("Total activities:", len(self.data))
        print(
            "Total calories burned:",
            round(self.data["Calories Burned"].sum(), 2)
        )
        print(
            "Total exercise time:",
            round(self.data["Duration (Minutes)"].sum(), 2),
            "minutes"
        )
        print(
            "Most frequent activity:",
            self.data["Activity Type"].mode().iloc[0]
        )

        print("\nActivity Summary:")
        summary = self.data.groupby("Activity Type").agg(
            Total_Duration=("Duration (Minutes)", "sum"),
            Total_Calories=("Calories Burned", "sum"),
            Number_of_Activities=("Activity Type", "count")
        )

        print(summary)

        summary.to_csv("fitness_report.csv")
        print("\nReport saved as fitness_report.csv")

    def visualize_data(self):
        if self.data.empty:
            print("No fitness data available.")
            return

        print("\n1. Bar Chart")
        print("2. Line Graph")
        print("3. Pie Chart")
        print("4. KDE plot")
        print("5. Display All Graphs")

        choice = input("Choose a graph: ").strip()

        sns.set_theme(style="whitegrid")

        if choice == "1" or choice == "5":
            activity_time = self.data.groupby("Activity Type")[
                "Duration (Minutes)"
            ].sum().reset_index()

            plt.figure(figsize=(8, 5))

            sns.barplot(
                data=activity_time,
                x="Activity Type",
                y="Duration (Minutes)",
                hue="Activity Type",
                palette="viridis",
                legend=False
            )

            plt.title("Total Time Spent on Each Activity")
            plt.xlabel("Activity Type")
            plt.ylabel("Duration (Minutes)")
            plt.tight_layout()
            plt.show()

        if choice == "2" or choice == "5":
            daily_data = self.data.groupby("Date")[
                "Calories Burned"
            ].sum().reset_index()

            plt.figure(figsize=(9, 5))

            sns.lineplot(
                data=daily_data,
                x="Date",
                y="Calories Burned",
                marker="o",
                color="darkorange"
            )

            plt.title("Calories Burned Over Time")
            plt.xlabel("Date")
            plt.ylabel("Calories Burned")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        if choice == "3" or choice == "5":
            activity_count = self.data["Activity Type"].value_counts()

            plt.figure(figsize=(7, 7))

            plt.pie(
                activity_count,
                labels=activity_count.index,
                autopct="%1.1f%%",
                startangle=90
            )

            plt.title("Percentage Distribution of Activities")
            plt.tight_layout()
            plt.show()

        if choice == "4" or choice == "5":

            plt.figure(figsize=(8, 5))

            sns.kdeplot(
                data=self.data,
                x="Duration (Minutes)",
                hue="Activity Type",
                fill=True,
                alpha=0.3,
                linewidth=2
            )

            plt.title("Distribution of Exercise Duration")
            plt.xlabel("Duration (Minutes)")
            plt.ylabel("Density")

            plt.tight_layout()
            plt.show()

        if choice not in ["1", "2", "3", "4", "5"]:
            print("Invalid graph choice.")


def main():
    tracker = FitnessTracker("fitness_activities.csv")

    while True:
        print("\n===== PERSONAL FITNESS TRACKER =====")
        print("1. Log Fitness Activity")
        print("2. Calculate Fitness Metrics")
        print("3. Filter Activities")
        print("4. Generate Fitness Report")
        print("5. Visualize Fitness Data")
        print("6. Display All Activities")
        print("7. Exit")

        choice = input("Enter your choice: ").strip()

        if choice == "1":
            tracker.log_activity()

        elif choice == "2":
            tracker.calculate_metrics()

        elif choice == "3":
            tracker.filter_activities()

        elif choice == "4":
            tracker.generate_report()

        elif choice == "5":
            tracker.visualize_data()

        elif choice == "6":
            if tracker.data.empty:
                print("No fitness activities available.")
            else:
                print(tracker.data.to_string(index=False))

        elif choice == "7":
            print("Thank you for using the Fitness Tracker!")
            break

        else:
            print("Invalid choice! Please enter a number from 1 to 7.")


if __name__ == "__main__":
    main()