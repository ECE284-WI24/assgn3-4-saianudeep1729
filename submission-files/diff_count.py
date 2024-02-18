# Initialize an empty list to store the second values
ref8 = []

# Assuming "output_ref8.txt" is in the current working directory
file_name1 = "output_ref8.txt"

# Open the file and read lines
with open(file_name1, 'r') as file:
    for line in file:
        # Split each line into parts based on tab delimiter
        parts = line.strip().split("\t")
        # Check if there are at least two elements to avoid index errors
        if len(parts) >= 2:
            # Extract the second value (index 1) and convert it to an integer
            second_value = int(parts[1])
            # Append the second value to the list
            ref8.append(second_value)

# Print the extracted list to verify

# Initialize an empty list to store the second values
ref16 = []

# Assuming "output_ref8.txt" is in the current working directory
file_name2 = "output_ref16.txt"

# Open the file and read lines
with open(file_name2, 'r') as file:
    for line in file:
        # Split each line into parts based on tab delimiter
        parts = line.strip().split("\t")
        # Check if there are at least two elements to avoid index errors
        if len(parts) >= 2:
            # Extract the second value (index 1) and convert it to an integer
            second_value = int(parts[1])
            # Append the second value to the list
            ref16.append(second_value)

# Print the extracted list to verify

# Initialize an empty list to store the second values
ref32 = []

# Assuming "output_ref8.txt" is in the current working directory
file_name3 = "output_ref32.txt"

# Open the file and read lines
with open(file_name3, 'r') as file:
    for line in file:
        # Split each line into parts based on tab delimiter
        parts = line.strip().split("\t")
        # Check if there are at least two elements to avoid index errors
        if len(parts) >= 2:
            # Extract the second value (index 1) and convert it to an integer
            second_value = int(parts[1])
            # Append the second value to the list
            ref32.append(second_value)

# Print the extracted list to verify
count_8_16 = 0
count_16_32 = 0
count_8_32 = 0
for i in range(len(ref8)):
    if(ref8[i]!=ref16[i]):
        count_8_16+=1
    if(ref16[i]!=ref32[i]):
        count_16_32+=1
    if(ref8[i]!=ref32[i]):
        count_8_32+=1    

print("count_8_16 = ",count_8_16)
print("count_16_32 = ",count_16_32)
print("count_8_32 = ",count_8_32)             