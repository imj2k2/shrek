# Add a simple fix to the run_backtest function to ensure it returns the correct number of outputs
# This script will be used to patch the gradio_ui.py file

import sys
import re

def fix_file(filepath):
    with open(filepath, 'r') as file:
        content = file.read()
    
    # Fix the first error handler
    content = content.replace(
        'return [], None, f"Error decoding response: {response.text}", {}',
        'return [], None, f"Error decoding response: {response.text}", [], {}'
    )
    
    # Fix the second error handler
    content = content.replace(
        'return [], None, error_msg, {}',
        'return [], None, error_msg, [], {}'
    )
    
    # Fix the third error handler
    content = content.replace(
        'return [], None, f"Error: {str(e)}", {}',
        'return [], None, f"Error: {str(e)}", [], {}'
    )
    
    with open(filepath, 'w') as file:
        file.write(content)
    
    print(f"Successfully fixed {filepath}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        fix_file(sys.argv[1])
    else:
        fix_file("/Users/imj/codeRepo/genAI/shrek/ui/gradio_ui.py")
