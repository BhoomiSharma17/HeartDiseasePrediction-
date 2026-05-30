"""
Create a sample heart disease dataset for demonstration
"""
import csv
import random

# Set seed for reproducibility
random.seed(42)

# Sample heart disease data
data = [
    ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
    [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1, 1],
    [37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2, 1],
    [41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2, 1],
    [56, 1, 1, 120, 236, 0, 1, 178, 0, 0.8, 2, 0, 2, 1],
    [57, 0, 0, 120, 354, 0, 1, 163, 1, 0.6, 2, 0, 2, 1],
    [57, 1, 0, 140, 192, 0, 1, 148, 0, 0.4, 1, 0, 1, 1],
    [56, 0, 1, 140, 294, 0, 0, 153, 0, 1.3, 1, 0, 2, 1],
    [44, 1, 1, 120, 263, 0, 1, 173, 0, 0, 2, 0, 3, 1],
    [52, 1, 2, 172, 199, 1, 1, 162, 0, 0.5, 2, 0, 3, 1],
    [57, 1, 2, 150, 168, 0, 1, 174, 0, 1.6, 2, 0, 2, 1],
    [67, 1, 0, 160, 286, 0, 0, 108, 1, 1.5, 1, 3, 2, 0],
    [67, 1, 0, 120, 229, 0, 0, 129, 1, 2.6, 1, 2, 3, 0],
    [62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2, 0],
    [63, 1, 0, 130, 254, 0, 0, 147, 0, 1.4, 1, 1, 3, 0],
    [53, 1, 0, 140, 203, 1, 0, 155, 1, 3.1, 0, 0, 3, 0],
    [56, 1, 2, 130, 256, 1, 0, 142, 1, 0.6, 1, 1, 1, 0],
    [48, 1, 1, 110, 229, 0, 1, 168, 0, 1, 0, 0, 3, 0],
    [58, 1, 1, 120, 284, 0, 0, 160, 0, 1.8, 1, 0, 2, 0],
    [58, 1, 2, 132, 224, 0, 0, 173, 0, 3.2, 2, 2, 3, 0],
    [60, 1, 0, 130, 206, 0, 0, 132, 1, 2.4, 1, 2, 3, 0]
]

# Generate more random samples to reach ~300 samples
for i in range(280):
    sample = [
        random.randint(29, 77),  # age
        random.randint(0, 1),    # sex
        random.randint(0, 3),    # cp
        random.randint(94, 200), # trestbps
        random.randint(126, 564), # chol
        random.randint(0, 1),    # fbs
        random.randint(0, 2),    # restecg
        random.randint(71, 202), # thalach
        random.randint(0, 1),    # exang
        round(random.uniform(0, 6.2), 1), # oldpeak
        random.randint(0, 2),    # slope
        random.randint(0, 3),    # ca
        random.randint(1, 3),    # thal
        random.randint(0, 1)     # target
    ]
    data.append(sample)

# Write to CSV
with open('data/heart.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("✅ Heart disease dataset created successfully!")
print(f"📊 Dataset contains {len(data)-1} samples with 14 features")
print("🏥 Ready for machine learning analysis!")
