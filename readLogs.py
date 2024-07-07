import re
from pprint import pprint

def parse_log_line(line):
    """
    Parse a single line of log data and return a dictionary with the parsed components.
    """
    regex = r'(\d{2}:\d{2}:\d{2}\.\d{6}) IP ([\w\.\-]+)\.(\d+) > ([\w\.\-]+)\.(\w+): (.*)'
    match = re.match(regex, line)
    if match:
        return {
            'timestamp': match.group(1),
            'source_ip': match.group(2),
            'source_port': match.group(3),
            'destination_ip': match.group(4),
            'destination_port': match.group(5),
            'details': match.group(6),
        }
    else:
        return None

def read_and_parse_log(file_path):
    """
    Read the log file and parse each line into a structured format.
    """
    parsed_data = []
    with open(file_path, 'r') as file:
        for line in file:
            parsed_line = parse_log_line(line.strip())
            if parsed_line:
                parsed_data.append(parsed_line)
    return parsed_data

def main():
    log_file_path = '/tcpdum_7072024.txt'  # Replace with the path to your log file
    parsed_data = read_and_parse_log(log_file_path)
    pprint(parsed_data)

if __name__ == "__main__":
    main()
