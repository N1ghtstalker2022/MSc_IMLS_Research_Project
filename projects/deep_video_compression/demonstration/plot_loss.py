import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # For a nicer color palette


def plot_loss(file1, file2, label, save_as):
    # Read the first loss data
    df1 = pd.read_csv(file1)
    # Read the second loss data
    df2 = pd.read_csv(file2)

    # Concatenate and sort by 'Step'
    df_combined = pd.concat([df1, df2]).reset_index(drop=True)
    df_combined = df_combined.sort_values('Step')

    # Extract sorted x and y values for plotting
    x_combined = df_combined['Step']
    y_combined = df_combined['loss']

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Use Seaborn's color palette
    sns.set_palette("husl")

    plt.plot(x_combined, y_combined, label=label, linestyle='-', linewidth=2, marker='o', markersize=5)

    # Add labels, title, and legend with custom font sizes
    plt.xlabel("Steps", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title(f"{label} Over Time", fontsize=16)
    plt.legend(fontsize=12)

    # Set x and y axis limits
    plt.xlim(min(x_combined), max(x_combined))

    # Style the ticks
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Add grid
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')

    # Save the plot to a file
    plt.savefig(save_as)

    plt.show()


# Plot and save validation loss chart
plot_loss("validation_loss1.csv", "validation_loss2.csv", "Validation Loss", "validation-loss.png")

# Plot and save training loss chart
plot_loss("train_loss1.csv", "train_loss2.csv", "Training Loss", "training-loss.png")
