import csv
import matplotlib.pyplot as plt

def plot_domain_shift_impact(csv_file):
    domain_shifts = []
    cumulative_rewards = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            domain_shifts.append(float(row['domain_shift']))
            cumulative_rewards.append(float(row['cumulative_reward']))

    # Now we'll plot the domain shift impact on cumulative reward
    plt.figure(figsize=(10, 5))
    plt.scatter(domain_shifts, cumulative_rewards, alpha=0.6)
    plt.title('Domain Shift Impact on Cumulative Reward')
    plt.xlabel('Domain Shift (Difference in Pole Length)')
    plt.ylabel('Cumulative Reward')
    plt.grid(True)
    plt.show()

# Call the function with the path to CSV file
plot_domain_shift_impact('random_change_training_data_with_predictor.csv')
   