import random
import pandas as pd

TOTAL_STUDENTS = 1000

data = []

for _ in range(TOTAL_STUDENTS):

    study_hours = random.randint(1, 10)
    sleep_hours = random.randint(3, 9)
    attendance = random.randint(50, 100)
    assignment_load = random.randint(0, 8)
    social_media_hours = random.randint(0, 10)
    exercise_hours = random.randint(0, 3)
    stress_level = random.randint(1, 10)

    score = 0

    if study_hours > 8:
        score += 2
    elif study_hours > 6:
        score += 1

    if sleep_hours < 5:
        score += 2
    elif sleep_hours < 7:
        score += 1

    if attendance < 70:
        score += 1

    if assignment_load > 5:
        score += 2
    elif assignment_load > 3:
        score += 1

    if social_media_hours > 6:
        score += 2
    elif social_media_hours > 4:
        score += 1

    if exercise_hours == 0:
        score += 2
    elif exercise_hours == 1:
        score += 1

    if stress_level > 8:
        score += 2
    elif stress_level > 5:
        score += 1

    if score <= 3:
        burnout_level = "Low"

    elif score <= 6:
        burnout_level = "Medium"

    else:
        burnout_level = "High"    

    data.append([
        study_hours,
        sleep_hours,
        attendance,
        assignment_load,
        social_media_hours,
        exercise_hours,
        stress_level,
        burnout_level
    ])

columns = [
    "Study_Hours",
    "Sleep_Hours",
    "Attendance",
    "Assignment_Load",
    "Social_Media_Hours",
    "Exercise_Hours",
    "Stress_Level",
    "Burnout_Level"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("dataset/burnout_dataset.csv", index=False)

print("Dataset created successfully!")
print(df.head())