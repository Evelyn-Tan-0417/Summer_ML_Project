"""
Callbacks and event handling logic for the food waste estimator.
Contains the main analyze_waste pipeline.
"""

import os
import sys
import cv2

# Ensure scripts directory is in python search path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent_dir, "scripts"))

from waste_estimator import FoodWasteEstimator, create_simulated_after_image
from database import MealDatabase
from recommender import WasteRecommender

def analyze_waste(before_img, after_img, hunger_before, fullness_after, taste_enjoyment, reason_leftover):
    """
    Main callback that processes before and after images, logs data to database,
    and returns visualization overlays and detailed HTML reports.
    """
    if before_img is None:
        return None, None, "<div style='color:#e53e3e; font-weight:bold; padding:20px; background:#fff5f5; border-radius:10px; border:1px solid #fed7d7;'>⚠️ Error: Please upload a Before-eating photo!</div>"
        
    before_path = before_img
    
    # Handle missing after image by simulating one
    if not after_img:
        results_dir = os.path.join(parent_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        after_path = os.path.join(results_dir, "simulated_after_meal.jpg")
        create_simulated_after_image(before_path, after_path)
    else:
        after_path = after_img

    try:
        # Load and run the pipeline
        estimator = FoodWasteEstimator()
        summary, before_drawn, after_drawn = estimator.estimate_waste(before_path, after_path)
        
        # Convert annotated BGR images to RGB for Gradio
        before_rgb = cv2.cvtColor(before_drawn, cv2.COLOR_BGR2RGB)
        after_rgb = cv2.cvtColor(after_drawn, cv2.COLOR_BGR2RGB)
        
        totals = summary["totals"]
        details = summary["details"]
        
        # Initialize Database and Recommender
        db = MealDatabase()
        recommender = WasteRecommender()
        
        # Clean survey inputs
        reason_clean = None if reason_leftover == "None" else reason_leftover
        
        # Log each detected food item in the SQLite DB
        for name, data in details.items():
            db.log_meal(
                food_category=name,
                before_photo=before_path,
                after_photo=after_path,
                leftover_pct=data["leftover_percentage"],
                wasted_weight=data["wasted_weight_grams"],
                calories=data["calories_wasted"],
                protein=data["protein_wasted_g"],
                carbs=data["carbs_wasted_g"],
                fat=data["fat_wasted_g"],
                co2=data["co2_wasted_kg"],
                hunger_before=int(hunger_before),
                fullness_after=int(fullness_after),
                taste_enjoyment=int(taste_enjoyment),
                reason_leftover=reason_clean
            )
        
        # Calculate overall leftover percentage based on weight
        total_baseline = sum([info["baseline_weight_g"] for info in details.values()])
        total_wasted = totals["wasted_weight_grams"]
        overall_pct = (total_wasted / total_baseline * 100.0) if total_baseline > 0 else 0.0
        
        # 1. Dashboard KPI Cards (Overall Leftover, Calories Wasted, Carbon footprint)
        kpi_html = f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px;">
            <div style="background: linear-gradient(135deg, #2b6cb0, #3182ce); color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 14px rgba(49, 130, 206, 0.15); text-align: center; border: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.85; font-weight: 700;">Overall Leftover</div>
                <div style="font-size: 36px; font-weight: 800; margin-top: 6px;">{overall_pct:.1f}%</div>
                <div style="font-size: 12px; margin-top: 4px; opacity: 0.9;">Wasted {total_wasted:.1f}g of {total_baseline:.1f}g</div>
            </div>
            <div style="background: linear-gradient(135deg, #c53030, #e53e3e); color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 14px rgba(229, 62, 62, 0.15); text-align: center; border: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.85; font-weight: 700;">Calories Wasted</div>
                <div style="font-size: 36px; font-weight: 800; margin-top: 6px;">{totals['calories_wasted']:.1f} kcal</div>
                <div style="font-size: 12px; margin-top: 4px; opacity: 0.9;">Total energy lost</div>
            </div>
            <div style="background: linear-gradient(135deg, #22543d, #2f855a); color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 14px rgba(47, 133, 90, 0.15); text-align: center; border: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.85; font-weight: 700;">Carbon Impact</div>
                <div style="font-size: 36px; font-weight: 800; margin-top: 6px;">{totals['co2_wasted_kg']:.3f} kg</div>
                <div style="font-size: 12px; margin-top: 4px; opacity: 0.9;">CO2e greenhouse impact</div>
            </div>
        </div>
        """
        
        # 2. Serving Sizing Recommendations Section
        recs_html = ""
        for name in details.keys():
            rec_text = recommender.get_portion_recommendation(db, name)
            recs_html += f"""
            <div style="margin-bottom: 12px; font-size: 13px; line-height: 1.5; color: #2d3748; padding-bottom: 8px; border-bottom: 1px solid #edf2f7;">
                <strong style="text-transform: uppercase; color: #2b6cb0;">• {name}:</strong> {rec_text}
            </div>
            """
        
        recommendations_section = f"""
        <div style="background: #ebf8ff; border-left: 4px solid #3182ce; padding: 22px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.01); border-top: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0;">
            <h4 style="margin-top: 0; color: #2b6cb0; font-size: 16px; font-weight: 800; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">💡 AI Serving Size Recommendations for Next Time</h4>
            {recs_html}
        </div>
        """
        
        # 3. Leftover Storage & Repurposing Strategies Section
        strats_html = ""
        for name, data in details.items():
            if data["leftover_percentage"] >= 5.0:
                strat = recommender.get_leftover_strategy(name, reason_clean)
                strats_html += f"""
                <div style="margin-bottom: 14px; font-size: 13px; line-height: 1.5; color: #2d3748; padding-bottom: 10px; border-bottom: 1px dashed #e2e8f0;">
                    <div style="font-weight: 800; text-transform: uppercase; color: #276749; margin-bottom: 4px;">• {name}</div>
                    <div style="margin-bottom: 2px;"><strong>Storage:</strong> {strat['storage_tip']}</div>
                    <div style="margin-bottom: 2px;"><strong>Recipe idea:</strong> {strat['repurposing_recipe']}</div>
                    <div style="color: #4a5568; font-style: italic; margin-top: 4px;">💡 Context Tip: {strat['contextual_tip']}</div>
                </div>
                """
        
        if strats_html:
            strategies_section = f"""
            <div style="background: #f0fff4; border-left: 4px solid #38a169; padding: 22px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.01); border-top: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0;">
                <h4 style="margin-top: 0; color: #276749; font-size: 16px; font-weight: 800; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">👨‍🍳 Storage & Leftover Repurposing Strategies</h4>
                {strats_html}
            </div>
            """
        else:
            strategies_section = ""
 
        # 4. Macronutrients Breakdown progress bars
        prot = totals["protein_wasted_g"]
        carb = totals["carbs_wasted_g"]
        fat = totals["fat_wasted_g"]
        sum_macros = prot + carb + fat
        
        p_pct = (prot / sum_macros * 100) if sum_macros > 0 else 0
        c_pct = (carb / sum_macros * 100) if sum_macros > 0 else 0
        f_pct = (fat / sum_macros * 100) if sum_macros > 0 else 0
        
        macro_html = f"""
        <div style="background: #ffffff; padding: 22px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.01);">
            <h4 style="margin-top: 0; color: #2d3748; font-size: 16px; font-weight: 800; margin-bottom: 18px; border-bottom: 1px solid #edf2f7; padding-bottom: 10px; display: flex; align-items: center; gap: 8px;">📊 Macronutrients Lost Breakdown</h4>
            <div style="margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; font-size: 13px; color: #4a5568; margin-bottom: 6px;">
                    <span style="font-weight: 700; color: #3182ce;">💪 Protein Wasted</span>
                    <span style="font-weight: 800; color: #2d3748;">{prot:.1f} g</span>
                </div>
                <div style="background: #edf2f7; height: 10px; border-radius: 5px; overflow: hidden;">
                    <div style="background: #3182ce; width: {p_pct}%; height: 100%; border-radius: 5px;"></div>
                </div>
            </div>
            <div style="margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; font-size: 13px; color: #4a5568; margin-bottom: 6px;">
                    <span style="font-weight: 700; color: #dd6b20;">🍞 Carbohydrates Wasted</span>
                    <span style="font-weight: 800; color: #2d3748;">{carb:.1f} g</span>
                </div>
                <div style="background: #edf2f7; height: 10px; border-radius: 5px; overflow: hidden;">
                    <div style="background: #dd6b20; width: {c_pct}%; height: 100%; border-radius: 5px;"></div>
                </div>
            </div>
            <div>
                <div style="display: flex; justify-content: space-between; font-size: 13px; color: #4a5568; margin-bottom: 6px;">
                    <span style="font-weight: 700; color: #e53e3e;">🥑 Fats Wasted</span>
                    <span style="font-weight: 800; color: #2d3748;">{fat:.1f} g</span>
                </div>
                <div style="background: #edf2f7; height: 10px; border-radius: 5px; overflow: hidden;">
                    <div style="background: #e53e3e; width: {f_pct}%; height: 100%; border-radius: 5px;"></div>
                </div>
            </div>
        </div>
        """
        
        # 5. Detailed Breakdown Table
        rows_html = ""
        for name, data in details.items():
            leftover = data["leftover_percentage"]
            
            if leftover >= 50.0:
                bg = "#fed7d7"
                txt = "#9b2c2c"
            elif leftover >= 10.0:
                bg = "#feebc8"
                txt = "#9c4221"
            else:
                bg = "#c6f6d5"
                txt = "#22543d"
                
            rows_html += f"""
            <tr style="border-bottom: 1px solid #edf2f7;">
                <td style="padding: 12px 8px; font-weight: 700; text-transform: capitalize; color: #2d3748;">{name}</td>
                <td style="padding: 12px 8px;"><span style="background: {bg}; color: {txt}; padding: 3px 12px; border-radius: 20px; font-size: 11px; font-weight: 800; border: 1px solid rgba(0,0,0,0.02); display: inline-block;">{leftover:.1f}%</span></td>
                <td style="padding: 12px 8px; font-weight: 700; color: #4a5568;">{data['wasted_weight_grams']:.1f}g <span style="font-size: 11px; color: #a0aec0; font-weight: normal;">(of {data['baseline_weight_g']:.0f}g)</span></td>
                <td style="padding: 12px 8px; font-weight: 700; color: #e53e3e;">{data['calories_wasted']:.1f} kcal</td>
                <td style="padding: 12px 8px; font-weight: 700; color: #2f855a;">{data['co2_wasted_kg']:.3f} kg</td>
            </tr>
            """
            
        table_html = f"""
        <div style="overflow-x: auto; background: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; padding: 22px; box-shadow: 0 4px 6px rgba(0,0,0,0.01);">
            <h4 style="margin-top: 0; color: #2d3748; font-size: 16px; font-weight: 800; margin-bottom: 18px; border-bottom: 1px solid #edf2f7; padding-bottom: 10px; display: flex; align-items: center; gap: 8px;">🍽️ Specific Ingredient Breakdown</h4>
            <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; color: #4a5568;">
                <thead>
                    <tr style="border-bottom: 2px solid #edf2f7; color: #718096; font-weight: 700; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px;">
                        <th style="padding: 10px 8px;">Ingredient</th>
                        <th style="padding: 10px 8px;">Leftover %</th>
                        <th style="padding: 10px 8px;">Est. Grams Lost</th>
                        <th style="padding: 10px 8px;">Calories Wasted</th>
                        <th style="padding: 10px 8px;">Carbon Impact</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """
        
        dashboard_content = kpi_html + recommendations_section + strategies_section + macro_html + table_html
        return before_rgb, after_rgb, dashboard_content
        
    except Exception as e:
        return None, None, f"<div style='color:#e53e3e; font-weight:bold; padding:20px; background:#fff5f5; border-radius:10px; border:1px solid #fed7d7;'>⚠️ Pipeline error: {str(e)}</div>"
