import pandas as pd

# read the file in chunks in ./datasets
x_part1 = pd.read_csv('./datasets/X_part1.csv')
x_part2 = pd.read_csv('./datasets/X_part2.csv')
y_part1 = pd.read_csv('./datasets/Y_part1.csv')
y_part2 = pd.read_csv('./datasets/Y_part2.csv')


x_combined = pd.concat([x_part1, x_part2])
y_combined = pd.concat([y_part1, y_part2])

# Saving the combined files
x_combined.to_csv('./datasets/X_combined.csv', index=False)
y_combined.to_csv('./datasets/Y_combined.csv', index=False)
