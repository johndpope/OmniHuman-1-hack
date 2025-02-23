# detect_duplicates_to_file.py
import os
import tempfile
from collections import defaultdict
from logger import logger

def extract_functions_to_file(directory='.', output_file=None):
    if output_file is None:
        output_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt').name
    
    logger.debug(f"Dumping functions to {output_file}")
    with open(output_file, 'w') as f:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    logger.debug(f"Processing {file_path}")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as source:
                            lines = source.readlines()
                            for i, line in enumerate(lines, 1):
                                stripped = line.strip()
                                if stripped.startswith('def '):
                                    parts = stripped.split('(', 1)
                                    if len(parts) > 1:
                                        func_name = parts[0].replace('def ', '').strip()
                                        f.write(f"{file_path}:{i}:{func_name}\n")
                    except Exception as e:
                        logger.error(f"Error reading {file_path}: {str(e)}")
    
    with open(output_file, 'r') as f:
        content = f.read().strip()
        logger.debug(f"Temp file contents:\n{content}")
    
    return output_file

def find_duplicates(temp_file):
    logger.debug("Checking for duplicates...")
    # Group by function name
    func_locations = defaultdict(list)
    with open(temp_file, 'r') as f:
        for line in f:
            if line.strip():
                path, line_num, func_name = line.strip().split(':', 2)
                func_locations[func_name].append(f"{path}:{line_num}")
    
    duplicates_found = False
    for func_name, locations in func_locations.items():
        if len(locations) > 1:
            duplicates_found = True
            logger.error(f"\033[91mDuplicate function '{func_name}' found at:\033[0m")
            for loc in locations:
                logger.error(f"  {loc}")
    
    if not duplicates_found:
        logger.error("\033[92mNo duplicate functions found.\033[0m")
    return duplicates_found

def main():
    temp_file = extract_functions_to_file()
    logger.debug(f"Temp file created: {temp_file}")
    try:
        find_duplicates(temp_file)
    finally:
        logger.debug(f"Cleaning up {temp_file}")
        # Keep file for now
        logger.debug(f"Temp file retained at {temp_file} for inspection")
        # os.remove(temp_file)

if __name__ == "__main__":
    main()