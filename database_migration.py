"""
Database Migration Script - Add equivalent_grip column
Run this on your AWS server to update the database schema
"""

import sqlite3
import os

DB_PATH = '/home/ubuntu/grip-monitor/grip-monitor/grip_strength_production.db'

def migrate_database():
    print("=" * 60)
    print("DATABASE MIGRATION - Adding equivalent_grip column")
    print("=" * 60)
    
    # Check if database exists
    if not os.path.exists(DB_PATH):
        print(f"❌ ERROR: Database not found at {DB_PATH}")
        return False
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if column already exists
        cursor.execute("PRAGMA table_info(grip_data)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'equivalent_grip' in columns:
            print("✓ Column 'equivalent_grip' already exists")
            print("  No migration needed")
        else:
            print("Adding 'equivalent_grip' column to grip_data table...")
            
            # Add new column
            cursor.execute("""
                ALTER TABLE grip_data 
                ADD COLUMN equivalent_grip REAL
            """)
            
            # Update existing rows with calculated values
            # equivalent_grip = total_grip * 1.28
            cursor.execute("""
                UPDATE grip_data 
                SET equivalent_grip = total_grip * 1.28
                WHERE equivalent_grip IS NULL
            """)
            
            conn.commit()
            print("✓ Column added successfully")
            
            # Verify
            cursor.execute("SELECT COUNT(*) FROM grip_data WHERE equivalent_grip IS NOT NULL")
            count = cursor.fetchone()[0]
            print(f"✓ Updated {count} existing records")
        
        # Show table schema
        print("\n" + "=" * 60)
        print("CURRENT TABLE SCHEMA:")
        print("=" * 60)
        cursor.execute("PRAGMA table_info(grip_data)")
        for col in cursor.fetchall():
            print(f"  {col[1]:20s} {col[2]:10s} {'NOT NULL' if col[3] else ''}")
        
        conn.close()
        print("\n✓ Migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = migrate_database()
    exit(0 if success else 1)
