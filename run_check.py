import traceback
from data_processor import process_and_report
import json

if __name__ == '__main__':
    path = r"C:\Users\USER\Documents\HR - HR.csv"
    try:
        report = process_and_report(path)
        print(json.dumps(report, indent=2))
    except Exception:
        traceback.print_exc()
