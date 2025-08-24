"""
PyTorch Tutorial Runner
Run this to execute all tutorial parts
"""

import subprocess
import sys
import os

def run_tutorial_part(filename, description):
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"File: {filename}")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, filename], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            print(result.stdout)
            if result.stderr:
                print("Warnings:", result.stderr)
        else:
            print(f"Error running {filename}:")
            print(result.stderr)
            
    except Exception as e:
        print(f"Failed to run {filename}: {e}")

def main():
    tutorials = [
        ("01_tensor_basics.py", "Part 1: Tensor Basics and Operations"),
        ("02_autograd_gradients.py", "Part 2: Autograd and Gradients"),
        ("03_neural_networks.py", "Part 3: Neural Networks"),
        ("04_interview_problems.py", "Part 4: Common Interview Problems"),
        ("05_advanced_topics.py", "Part 5: Advanced Topics and Best Practices")
    ]
    
    print("PyTorch Tutorial Series")
    print("Choose an option:")
    print("0. Run all tutorials")
    for i, (filename, description) in enumerate(tutorials, 1):
        print(f"{i}. {description}")
    
    try:
        choice = int(input("\nEnter your choice (0-5): "))
        
        if choice == 0:
            for filename, description in tutorials:
                run_tutorial_part(filename, description)
        elif 1 <= choice <= len(tutorials):
            filename, description = tutorials[choice - 1]
            run_tutorial_part(filename, description)
        else:
            print("Invalid choice!")
            
    except ValueError:
        print("Please enter a valid number!")
    except KeyboardInterrupt:
        print("\nTutorial interrupted by user.")

if __name__ == "__main__":
    main()