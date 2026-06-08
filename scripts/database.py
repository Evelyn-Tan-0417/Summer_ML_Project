"""
Smart Food Waste Estimator - SQLite Relational Database Module
==============================================================
This module handles all database operations: logging meal records,
calculating long-term statistics, and extracting trends for the recommender.
"""

import os
import sqlite3
from datetime import datetime

class MealDatabase:
    def __init__(self, db_path="results/food_waste.db"):
        """Initialize the SQLite database and ensure tables exist."""
        self.db_path = db_path
        # Ensure parent directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        self._init_db()

    def _get_connection(self):
        """Returns a connection to the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Returns rows as dictionary-like objects
        return conn

    def _init_db(self):
        """Creates the necessary tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS meal_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    food_category TEXT NOT NULL,
                    before_photo_path TEXT,
                    after_photo_path TEXT,
                    leftover_percentage REAL NOT NULL,
                    wasted_weight_grams REAL NOT NULL,
                    calories_wasted REAL NOT NULL,
                    protein_wasted_g REAL NOT NULL,
                    carbs_wasted_g REAL NOT NULL,
                    fat_wasted_g REAL NOT NULL,
                    co2_wasted_kg REAL NOT NULL,
                    hunger_before INTEGER,        -- Survey: 1-10
                    taste_enjoyment INTEGER,      -- Survey: 1-10
                    reason_leftover TEXT          -- Survey: ex. too full, too much carbs, etc.
                )
            """)
            conn.commit()

    def log_meal(self, food_category, before_photo, after_photo, leftover_pct, 
                 wasted_weight, calories, protein, carbs, fat, co2, 
                 hunger_before=None, taste_enjoyment=None, reason_leftover=None):
        """Logs a completed meal waste entry into the database."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO meal_logs (
                    timestamp, food_category, before_photo_path, after_photo_path,
                    leftover_percentage, wasted_weight_grams, calories_wasted,
                    protein_wasted_g, carbs_wasted_g, fat_wasted_g, co2_wasted_kg,
                    hunger_before, taste_enjoyment, reason_leftover
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, food_category.lower(), before_photo, after_photo,
                leftover_pct, wasted_weight, calories,
                protein, carbs, fat, co2,
                hunger_before, taste_enjoyment, reason_leftover
            ))
            conn.commit()
            return cursor.lastrowid

    def get_all_logs(self, limit=100):
        """Fetches all logged entries, ordered by newest first."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM meal_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_summary_totals(self):
        """Calculates historical summary aggregates across all entries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_meals,
                    SUM(wasted_weight_grams) as total_weight_wasted,
                    SUM(calories_wasted) as total_calories_wasted,
                    SUM(co2_wasted_kg) as total_co2_wasted,
                    AVG(leftover_percentage) as avg_leftover_percentage
                FROM meal_logs
            """)
            row = cursor.fetchone()
            return dict(row) if row and row["total_meals"] > 0 else {
                "total_meals": 0,
                "total_weight_wasted": 0.0,
                "total_calories_wasted": 0.0,
                "total_co2_wasted": 0.0,
                "avg_leftover_percentage": 0.0
            }

    def get_category_stats(self):
        """Returns aggregated stats grouped by food category."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    food_category,
                    COUNT(*) as count,
                    AVG(leftover_percentage) as avg_leftover,
                    SUM(wasted_weight_grams) as total_wasted_weight,
                    SUM(calories_wasted) as total_calories_wasted,
                    SUM(co2_wasted_kg) as total_co2_wasted
                FROM meal_logs
                GROUP BY food_category
                ORDER BY total_wasted_weight DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_reason_distribution(self):
        """Returns counts of reasons why food was left behind."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT reason_leftover, COUNT(*) as count
                FROM meal_logs
                WHERE reason_leftover IS NOT NULL AND reason_leftover != ''
                GROUP BY reason_leftover
                ORDER BY count DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_category_trend(self, category, limit=5):
        """Returns the last N entries for a specific food category."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT leftover_percentage, hunger_before, taste_enjoyment
                FROM meal_logs
                WHERE food_category = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (category.lower(), limit))
            return [dict(row) for row in cursor.fetchall()]

if __name__ == "__main__":
    # Test database creation and logging
    db = MealDatabase("results/test_food_waste.db")
    print("[Db Test] Database initialized.")
    
    # Insert a dummy record
    dummy_id = db.log_meal(
        food_category="rice",
        before_photo="images/steak.jfif",
        after_photo="results/steak_simulated_after.jfif",
        leftover_pct=33.3,
        wasted_weight=59.9,
        calories=77.8,
        protein=1.6,
        carbs=16.8,
        fat=0.2,
        co2=0.09,
        hunger_before=6,
        taste_enjoyment=8,
        reason_leftover="too full"
    )
    print(f"[Db Test] Logged dummy meal. Row ID: {dummy_id}")
    
    # Retrieve stats
    totals = db.get_summary_totals()
    print(f"[Db Test] Total Meals Logged: {totals['total_meals']}")
    print(f"[Db Test] Average Leftovers : {totals['avg_leftover_percentage']:.1f}%")
    
    # Clean up test database file
    if os.path.exists("results/test_food_waste.db"):
        os.remove("results/test_food_waste.db")
        print("[Db Test] Test database cleaned up.")
