import os

def read_file(filepath):
  """
  Reads the contents of a file and returns them as a string.

  Args:
      filepath: The path to the file to read.

  Returns:
      The contents of the file as a string, or None if there's an error.
  """

  try:
    with open(filepath, 'r') as file:
      # Read the entire file
      contents = file.read()
      return contents
  except FileNotFoundError:
    print(f"Error: File not found: {filepath}")
  except PermissionError:
    print(f"Error: Insufficient permissions to read file: {filepath}")
  except Exception as e:
    print(f"Error reading file: {filepath} - {e}")
  return None

# Example usage
filepath = "/bbb//Downloadsbbb/Debian_update.sh"  # Replace with the actual path
contents = read_file(filepath)

if contents:
  print("File contents:")
  print(contents)
else:
  print("Failed to read file.")
