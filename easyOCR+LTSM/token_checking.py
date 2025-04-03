import pickle
import os

file_path = "models/tokenizer.pkl"

# Check if file exists
if not os.path.exists(file_path):
    print("❌ tokenizer.pkl does not exist!")
else:
    print(f"✅ tokenizer.pkl found! Size: {os.path.getsize(file_path)} bytes")

    # Try loading the tokenizer
    try:
        with open(file_path, "rb") as handle:
            tokenizer = pickle.load(handle)
        print("✅ tokenizer.pkl loaded successfully!")
    except EOFError:
        print("❌ ERROR: tokenizer.pkl is empty or corrupted!")
    except pickle.UnpicklingError:
        print("❌ ERROR: tokenizer.pkl is not a valid pickle file!")
