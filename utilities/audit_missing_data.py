import csv
import os
import sys

# Configuration
# This assumes the script is run from inside the 'utilities' folder
PLAYER_FILE_PATH = os.path.join('..', 'atp_players.csv')
OUTPUT_FILE = 'missing_bio_report.csv'

def audit_players():
    """
    Scans the master ATP player file and identifies players with 
    missing critical biographical information (Birthdate or Hand).
    Generates a CSV report of these players.
    """
    
    # Verify the player file exists
    if not os.path.exists(PLAYER_FILE_PATH):
        print(f"Error: Could not find player file at {PLAYER_FILE_PATH}")
        print("Make sure you are running this script from the 'utilities' directory.")
        return

    print(f"Scanning {PLAYER_FILE_PATH} for missing data...")
    
    missing_count = 0
    
    try:
        with open(PLAYER_FILE_PATH, 'r', encoding='utf-8') as f_in, \
             open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f_out:
            
            # The atp_players.csv file does not have a header, so we read directly
            reader = csv.reader(f_in)
            writer = csv.writer(f_out)
            
            # Write a header for our report
            writer.writerow(['Player ID', 'First Name', 'Last Name', 'Missing Fields'])
            
            for row in reader:
                # Ensure row has enough columns to avoid index errors
                if len(row) < 5:
                    continue
                    
                # ATP Player File Columns:
                # 0: ID, 1: First, 2: Last, 3: Hand, 4: DOB, 5: Country
                p_id, first, last, hand, dob = row[0], row[1], row[2], row[3], row[4]
                
                missing_fields = []
                
                # Check for missing Hand (often 'U' for unknown or empty)
                if not hand or hand == 'U':
                    missing_fields.append('Hand')
                
                # Check for missing DOB (often '00000000' or empty)
                if not dob or dob == '00000000':
                    missing_fields.append('Birthdate')
                
                if missing_fields:
                    writer.writerow([p_id, first, last, ", ".join(missing_fields)])
                    missing_count += 1

        print(f"Audit complete.")
        print(f"Found {missing_count} players with missing biographical data.")
        print(f"Report saved to: {os.path.abspath(OUTPUT_FILE)}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    audit_players()
