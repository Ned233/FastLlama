import subprocess
import sys
import time

def main():
    print("\nGenerating distillation data...")
    print("-"*80)
    result = subprocess.run([sys.executable, "generate_distill_data.py"] + sys.argv[1:])
    if result.returncode != 0:
        print("distillation data failed!")
        sys.exit(1)
    time.sleep(2)
    
    print("\nLayer-wise training with early stopping...")
    print("-"*80)
    result = subprocess.run([sys.executable, "train_fftchain.py"] + sys.argv[1:])
    if result.returncode != 0:
        print("failed!")
        sys.exit(1)
    time.sleep(2)
    
    print("\nValidating trained model...")
    print("-"*80)
    result = subprocess.run([sys.executable, "validate.py"] + sys.argv[1:])
    if result.returncode != 0:
        print("failed!")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("completed successfully!")
    print("="*80)

if __name__ == '__main__':
    main()