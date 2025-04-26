#!/usr/bin/env python3
"""
Run and Visualize initialization of beliefs
--------------------------------------------
This script creates visualization for random initialization of the beliefs 

NOTE: This header and parts of the code were written with the assistance of generative AI.

Results are saved to a PNG file with a timestamp in the filename.

Authors: rootma21, VivekRkay24, saswata0502, DhanyaSNaik
Last updated: April 18th, 2025
"""

import numpy as np
import matplotlib.pyplot as plt

# Number of samples for each belief parameter
n_samples = 10000

# Generate distributions based on the code parameters:
# 1. Fear of COVID: Normal(mean=7, std=2), truncated to [0,10]
fear_covid = np.clip(np.random.normal(7, 2, n_samples), 0, 10).astype(int)

# 2. Mask Annoyance Factor: Normal(mean=6, std=2), truncated to [0,10]
mask_annoyance = np.clip(np.random.normal(6, 2, n_samples), 0, 10).astype(int)

# 3. Loneliness Factor: Normal(mean=5, std=2), truncated to [0,10]
loneliness = np.clip(np.random.normal(5, 2, n_samples), 0, 10).astype(int)

# 4. Compliance for Vaccine: Normal(mean=6, std=2), truncated to [0,10]
compliance_vaccine = np.clip(np.random.normal(6, 2, n_samples), 0, 10).astype(int)

# 5. Compliance for Mask: Normal(mean=6, std=2), truncated to [0,10]
compliance_mask = np.clip(np.random.normal(6, 2, n_samples), 0, 10).astype(int)

# 6. Fear of Vaccine: Normal(mean=3, std=1.5), truncated to [0,10]
fear_vaccine = np.clip(np.random.normal(3, 1.5, n_samples), 0, 10).astype(int)

# 7. Family Lockdown Compliance: Normal(mean=5, std=2), truncated to [0,10]
family_lockdown = np.clip(np.random.normal(5, 2, n_samples), 0, 10).astype(int)

# 8. Age Factor: Uniform distribution between 0 and 1
age_factor = np.random.uniform(0, 1, n_samples)

# 9. Family Anti-Vax: Bernoulli-like; 0 with p=0.8, 1 with p=0.2
family_anti_vax = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

# Prepare the figure with 3x3 grid of subplots
fig, axs = plt.subplots(3, 3, figsize=(18, 14))
axs = axs.flatten()

# Define bins for integer-distributions (0 to 10)
bins_int = np.arange(-0.5, 11, 1)

# Plot each distribution with clear titles and labels
axs[0].hist(fear_covid, bins=bins_int, edgecolor='black', color='skyblue')
axs[0].set_title('Fear of COVID (mean=7, std=2)')
axs[0].set_xlabel("Value")
axs[0].set_ylabel("Frequency")
axs[0].set_xticks(range(0, 11))

axs[1].hist(mask_annoyance, bins=bins_int, edgecolor='black', color='salmon')
axs[1].set_title('Mask Annoyance (mean=6, std=2)')
axs[1].set_xlabel("Value")
axs[1].set_xticks(range(0, 11))

axs[2].hist(loneliness, bins=bins_int, edgecolor='black', color='lightgreen')
axs[2].set_title('Loneliness Factor (mean=5, std=2)')
axs[2].set_xlabel("Value")
axs[2].set_xticks(range(0, 11))

axs[3].hist(compliance_vaccine, bins=bins_int, edgecolor='black', color='orchid')
axs[3].set_title('Compliance for Vaccine (mean=6, std=2)')
axs[3].set_xlabel("Value")
axs[3].set_ylabel("Frequency")
axs[3].set_xticks(range(0, 11))

axs[4].hist(compliance_mask, bins=bins_int, edgecolor='black', color='gold')
axs[4].set_title('Compliance for Mask (mean=6, std=2)')
axs[4].set_xlabel("Value")
axs[4].set_xticks(range(0, 11))

axs[5].hist(fear_vaccine, bins=bins_int, edgecolor='black', color='lightcoral')
axs[5].set_title('Fear of Vaccine (mean=3, std=1.5)')
axs[5].set_xlabel("Value")
axs[5].set_xticks(range(0, 11))

axs[6].hist(family_lockdown, bins=bins_int, edgecolor='black', color='plum')
axs[6].set_title('Family Lockdown Compliance (mean=5, std=2)')
axs[6].set_xlabel("Value")
axs[6].set_ylabel("Frequency")
axs[6].set_xticks(range(0, 11))

# Age Factor: Uniform distribution; using 20 bins between 0 and 1
axs[7].hist(age_factor, bins=20, edgecolor='black', color='lightblue')
axs[7].set_title('Age Factor (Uniform [0,1])')
axs[7].set_xlabel("Value")
axs[7].set_ylabel("Frequency")

# Family Anti-Vax: Categorical distribution (0 or 1)
bins_cat = [-0.5, 0.5, 1.5]
axs[8].hist(family_anti_vax, bins=bins_cat, edgecolor='black', color='lightgrey')
axs[8].set_title('Family Anti-Vax (p(0)=0.8, p(1)=0.2)')
axs[8].set_xlabel("Value")
axs[8].set_xticks([0, 1])

plt.tight_layout()
plt.suptitle("Belief System Parameter Distributions", fontsize=20, y=1.02)
plt.subplots_adjust(top=0.92)

# Save the chart to an image file for your presentation
plt.savefig("beliefs_presentation_chart.png", dpi=300, bbox_inches="tight")
plt.show()
