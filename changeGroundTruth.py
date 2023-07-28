import csv

def process_csv_file(input_file, labels_dict):
    # Create a list to store the modified rows
    modified_rows = []
    
    # Open the CSV file and read its content
    with open(input_file, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        fieldnames = csv_reader.fieldnames
        
        for row in csv_reader:
            label = row['Label']
            if label in labels_dict:
                valence, arousal = labels_dict[label]
                row['Valence'] = str(valence)
                row['Arousal'] = str(arousal)
            #else:
            #  row['Valence'] = '0'
            #  row['Arousal'] = '0'
                
            # Append the modified row to the list
            modified_rows.append(row)
    
    # Write the modified rows back to a new CSV file
    output_file = input_file.replace('.csv', '_modified.csv')
    with open(output_file, 'w', newline='') as csv_output_file:
        writer = csv.DictWriter(csv_output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(modified_rows)

    print(f"Processed data saved to: {output_file}")

# Replace 'input.csv' with the path to your input .csv file
input_file = 'input\oldPrepFile.csv'

# Replace this dictionary with your LabelsDictionary
labels_dict = {
    'vid10': (0.99, 0.1),
    'vid13': (0.99, 0.1),
    'vid138': (0.1, 0.1),
    'vid18': (0.99, 0.1),
    'vid19': (0.1, 0.1),
    'vid20': (0.1, 0.1),
    'vid23': (0.1, 0.1),
    'vid30': (0.1, 0.99),
    'vid31': (0.1, 0.99),
    'vid34': (0.1, 0.99),
    'vid36': (0.1, 0.99),
    'vid4': (0.99, 0.99),
    'vid5': (0.99, 0.99),
    'vid58': (0.99, 0.1),
    'vid80': (0.99, 0.99),
    'vid9': (0.99, 0.99),
    # Add more key-value pairs as needed
}

process_csv_file(input_file, labels_dict)
