import pandas as pd

# Function to process chunks
def process_and_save_chunks(file_path, chunksize=10000):
    # Create an empty DataFrame to store the processed data
    processed_data_part1 = pd.DataFrame()
    processed_data_part2 = pd.DataFrame()

    # Read the file in chunks
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        # Process the chunk (split into two parts in this example)
        part1, part2 = chunk.iloc[:len(chunk)//2, :], chunk.iloc[len(chunk)//2:, :]

        # Concatenate the processed parts to the respective DataFrames
        processed_data_part1 = pd.concat([processed_data_part1, part1])
        processed_data_part2 = pd.concat([processed_data_part2, part2])

    # Saving the processed parts as new CSV files
    processed_data_part1.to_csv(file_path.replace('.csv', '_part1.csv'), index=False)
    processed_data_part2.to_csv(file_path.replace('.csv', '_part2.csv'), index=False)

# Process 'X.csv' and 'Y.csv'
process_and_save_chunks('./datasets/X.csv')
process_and_save_chunks('./datasets/Y.csv')

# Reading and concatenating the processed parts
x_part1 = pd.read_csv('./datasets/X_part1.csv')
x_part2 = pd.read_csv('./datasets/X_part2.csv')
y_part1 = pd.read_csv('./datasets/Y_part1.csv')
y_part2 = pd.read_csv('./datasets/Y_part2.csv')

# x_combined = pd.concat([x_part1, x_part2])
# y_combined = pd.concat([y_part1, y_part2])

# # Saving the combined files
# x_combined.to_csv('./datasets/X_combined.csv', index=False)
# y_combined.to_csv('./datasets/Y_combined.csv', index=False)
