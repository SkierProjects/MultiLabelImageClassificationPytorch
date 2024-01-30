import csv

def analyze_csv(csv_file_path):
    # Initialize dictionaries to store annotation counts and file counts
    annotation_counts = {}
    file_counts = {'with_annotations': 0, 'without_annotations': 0}

    with open(csv_file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Iterate through each row in the CSV
        for row in reader:
            file_name = row['filepath']
            
            # Count files without any annotations
            if all(value == '0' for key, value in row.items() if key != 'filepath'):
                file_counts['without_annotations'] += 1
            else:
                file_counts['with_annotations'] += 1
            
            # Count the usage of each annotation
            for annotation_name, annotation_value in row.items():
                if annotation_name != 'filepath':
                    annotation_counts[annotation_name] = annotation_counts.get(annotation_name, 0) + int(annotation_value)

    return annotation_counts, file_counts

# Replace 'your_file.csv' with the actual path to your CSV file
csv_file_path = 'Dataset/datasetV2.csv'
annotation_counts, file_counts = analyze_csv(csv_file_path)

# Print results
print("Annotation Counts:")
for annotation, count in annotation_counts.items():
    print(f"{annotation}: {count}")

print("\nFile Counts:")
print(f"Files with annotations: {file_counts['with_annotations']}")
print(f"Files without annotations: {file_counts['without_annotations']}")
