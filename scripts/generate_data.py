import random
import csv
from datetime import datetime, timedelta
from collections import defaultdict

random.seed()

# ---------- CONFIG ----------

schools = [
    "VSST", "TSM", "VSOL",
    "VSOD", "JAGSoM", "Junior College"
]

venues = [
    ("Activity Room", 50),
    ("Agni 1", 30), ("Agni 2", 30), ("Agni 3", 30),
    ("Agni 4", 30), ("Agni 5", 30), ("Agni 6", 30),
    ("Agni 7", 30), ("Agni 8", 30),
    ("Arts A", 30), ("Arts B", 30), ("Arts C", 50),
    ("Computer Lab", 50),
    ("Music Lab 1", 30), ("Music Lab 2", 30)
]

# random time generator
base_hours = [9, 11, 13, 15]
minutes = [0, 15, 30, 45]

school_activity_prob = {
    "VSST": 0.8,
    "TSM": 0.6,
    "VSOL": 0.7,
    "VSOD": 0.7,
    "JAGSoM": 0.75,
    "Junior College": 0.9
}

start_date = datetime(2026, 1, 1)
end_date = datetime(2026, 1, 31)

history = defaultdict(list)

# ---------- ATTENDANCE MODEL ----------

def generate_attendance(enrolled, school, exam_week, hour, prev):
    base = random.uniform(0.6, 0.95)

    if exam_week:
        base -= 0.2

    if hour == 9:
        base -= 0.1

    if school == "TSM":
        base += 0.05

    if prev:
        base += (prev / max(enrolled,1)) * 0.1

    rate = max(0.3, min(base, 1.0))
    attendance = int(enrolled * rate)

    if random.random() < 0.05:
        attendance += random.randint(1, 10)

    return max(0, attendance)

# ---------- SIMULATION ----------

rows = []
current_date = start_date
semester_day = 1

while current_date <= end_date:

    weekday = current_date.strftime("%A")

    # internal holiday logic (not stored)
    holiday_roll = random.random()
    if holiday_roll < 0.15:
        active_schools = []
    elif holiday_roll < 0.35:
        active_schools = random.sample(schools, random.randint(1, 3))
    else:
        active_schools = schools[:]

    exam_week = semester_day > 20

    used_rooms = set()

    for school in active_schools:

        if random.random() > school_activity_prob[school]:
            continue

        sessions_today = random.randint(1, 4)

        for _ in range(sessions_today):

            available_rooms = [v for v in venues if v[0] not in used_rooms]
            if not available_rooms:
                break

            venue, capacity = random.choice(available_rooms)
            used_rooms.add(venue)

            hour = random.choice(base_hours)
            minute = random.choice(minutes)
            time_str = f"{hour:02d}:{minute:02d}"

            enrolled = random.randint(int(capacity * 0.6), capacity + 15)

            key = (school, venue)
            prev = history[key][-1] if history[key] else None

            attendance = generate_attendance(enrolled, school, exam_week, hour, prev)

            history[key].append(attendance)

            rolling3 = sum(history[key][-3:]) / min(3, len(history[key]))
            rolling5 = sum(history[key][-5:]) / min(5, len(history[key]))
            trend = attendance - prev if prev else 0

            rows.append([
                current_date.strftime("%Y-%m-%d"),
                time_str,
                school,
                venue,
                capacity,
                weekday,
                semester_day,
                exam_week,
                enrolled,
                prev if prev else 0,
                round(rolling3, 2),
                round(rolling5, 2),
                trend,
                attendance
            ])

    current_date += timedelta(days=1)
    semester_day += 1

# ---------- SAVE CSV ----------

header = [
    "date",
    "time",
    "school",
    "venue",
    "venue_capacity",
    "weekday",
    "semester_day",
    "exam_week",
    "enrolled_students",
    "prev_attendance",
    "rolling_avg_3",
    "rolling_avg_5",
    "trend_delta",
    "actual_attendance"
]

filename = "campus_occupancy_dataset.csv"

with open(filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"✅ Dataset created: {filename}")
print(f"Rows generated: {len(rows)}")
