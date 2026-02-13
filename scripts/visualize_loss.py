import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('log', help='log file path.')
    args = parser.parse_args()

    # Load your JSONL file
    data = []
    with open(args.log, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Calculate moving average (e.g., window size = 5)
    df['loss_smooth'] = df['loss'].rolling(window=100, min_periods=1).mean()

    # Plot original and smoothed loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(df['iter'], df['loss'], alpha=0.3, label='Original Loss')
    plt.plot(df['iter'], df['loss_smooth'], label='Smoothed Loss', linewidth=2)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Smoothed Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss.jpg')
