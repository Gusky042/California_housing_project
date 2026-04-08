import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

print("🎈 LET'S LEARN ABOUT CORRELATION - THE FRIENDSHIP METER!")
print("=" * 50)

# Load our house data
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

print("🏠 We have data about houses in California")
print(f"We have {df.shape[0]} houses and {df.shape[1]} things we know about each house")
print("\nHere are the first 3 houses:")
print(df.head(3))
print("\n🏠 REAL EXAMPLE: ANALYZING CALIFORNIA HOUSES")
print("=" * 50)

# Load the data
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

print("We're going to create a 2x2 dashboard to understand house prices!")

# Create a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('California Housing Analysis Dashboard 🏠', fontsize=16, fontweight='bold')

# Box 1: Income vs Price (Most important relationship!)
axes[0, 0].scatter(df['MedInc'], df['MedHouseVal'], alpha=0.3, s=10, c='green')
axes[0, 0].set_xlabel('Median Income ($100,000s)', fontsize=11)
axes[0, 0].set_ylabel('House Price ($100,000s)', fontsize=11)
axes[0, 0].set_title('Income vs Price\n(BEST FRIENDS!)', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(df['MedInc'], df['MedHouseVal'], 1)
p = np.poly1d(z)
axes[0, 0].plot(df['MedInc'].sort_values(), p(df['MedInc'].sort_values()),
               'r--', linewidth=2, label='Trend')
axes[0, 0].legend()

# Box 2: House Age vs Price
axes[0, 1].scatter(df['HouseAge'], df['MedHouseVal'], alpha=0.3, s=10, c='blue')
axes[0, 1].set_xlabel('House Age (years)', fontsize=11)
axes[0, 1].set_ylabel('House Price ($100,000s)', fontsize=11)
axes[0, 1].set_title('Age vs Price\n(Weak Friends)', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# Box 3: Latitude vs Price (North vs South)
axes[1, 0].scatter(df['Latitude'], df['MedHouseVal'], alpha=0.3, s=10, c='red')
axes[1, 0].set_xlabel('Latitude (North vs South)', fontsize=11)
axes[1, 0].set_ylabel('House Price ($100,000s)', fontsize=11)
axes[1, 0].set_title('Location vs Price\n(Rivals!)', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

# Box 4: Rooms vs Price
axes[1, 1].scatter(df['AveRooms'], df['MedHouseVal'], alpha=0.3, s=10, c='orange')
axes[1, 1].set_xlabel('Average Rooms', fontsize=11)
axes[1, 1].set_ylabel('House Price ($100,000s)', fontsize=11)
axes[1, 1].set_title('Rooms vs Price\n(Weak Positive)', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n💡 WHAT WE LEARNED FROM OUR DASHBOARD:")
print("  📈 Box 1: Income is BEST FRIEND with price!")
print("  📈 Box 2: Age is WEAK friend with price")
print("  📉 Box 3: Latitude is RIVAL (north = cheaper)")
print("  📈 Box 4: More rooms = slightly higher price")