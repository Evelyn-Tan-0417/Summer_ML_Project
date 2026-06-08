"""
Smart Food Waste Estimator - Recommender Engine
================================================
This module generates customized portion reduction recommendations based on
historical meal logs and provides storage or recipe repurposing strategies
based on the food category and reason for waste.
"""

# Dict of category preservation & leftover repurposing recipes
LEFTOVER_STRATEGIES = {
    "rice": {
        "storage": "Refrigerate in an airtight container within 1 hour of cooking to prevent bacterial growth. Safe to eat for up to 4 days.",
        "recipe": "Transform into Egg Fried Rice: Sauté garlic, onions, and scrambled eggs, then toss in your cold leftover rice with a splash of soy sauce."
    },
    "noodles": {
        "storage": "Store separately from broth in the fridge. Safe for 3-4 days.",
        "recipe": "Noodle Stir-fry: Toss cold noodles in a hot skillet with some sesame oil, vegetables, and a spoonful of oyster sauce for 2 minutes."
    },
    "pasta": {
        "storage": "Mix with a splash of olive oil to keep from sticking, store in a sealed container for up to 5 days.",
        "recipe": "Frittata/Pasta Bake: Mix leftovers with whisked egg, parmesan, and herbs, then bake in a skillet until set and golden brown."
    },
    "beef": {
        "storage": "Wrap tightly in foil or store in an airtight container. Refrigerate up to 4 days or freeze for up to 3 months.",
        "recipe": "Leftover Beef Tacos/Fajitas: Shred or slice the beef thin, reheat in a pan with cumin, chili powder, and sliced onions, and serve in tortillas."
    },
    "pork": {
        "storage": "Refrigerate for up to 4 days. Shred or chop before storing to make reheating quicker.",
        "recipe": "Quick Pulled Pork Sliders: Reheat shredded pork in a small pot with a splash of BBQ sauce and serve on toasted burger buns."
    },
    "chicken": {
        "storage": "Shred or cut chicken off the bone before refrigerating (keeps fresh up to 4 days, freezes up to 4 months).",
        "recipe": "Creamy Chicken Salad or Quesadillas: Toss shredded chicken with mayonnaise and celery for sandwiches, or melt in tortillas with cheese."
    },
    "fish": {
        "storage": "Consume within 1-2 days. Wrap tightly as seafood odors spread easily.",
        "recipe": "Seafood Fishcakes: Flake the fish, mix with mashed potatoes, an egg, and breadcrumbs, shape into patties and pan-fry."
    },
    "vegetables": {
        "storage": "Refrigerate for up to 3-5 days. Best stored separately from sauces.",
        "recipe": "Rich Vegetable Soup: Simmer remaining veggies in vegetable broth with canned tomatoes, herbs, and beans."
    },
    "bread": {
        "storage": "Store at room temp in a paper bag (refrigerating makes it go stale faster). Freeze sliced bread for up to 6 months.",
        "recipe": "Homemade Garlic Croutons: Dice stale bread, toss with olive oil, garlic powder, and salt, then bake at 180°C for 10-15 minutes."
    },
    "default": {
        "storage": "Store in an airtight container in the refrigerator for up to 3 days, or freeze for up to 3 months.",
        "recipe": "Quick Sauté: Reheat in a pan with a splash of oil, garlic, and fresh herbs to revive flavor and texture."
    }
}

class WasteRecommender:
    def __init__(self):
        pass

    def get_portion_recommendation(self, db, category):
        """
        Analyzes the last 5 entries for a specific food category and generates
        a portion adjustment advice.
        """
        logs = db.get_category_trend(category, limit=5)
        
        if not logs:
            return "This is your first time logging this food category. Eat mindfully and see how much you consume."
            
        leftovers = [log["leftover_percentage"] for log in logs]
        avg_leftover = sum(leftovers) / len(leftovers)
        
        # Check if they are consistently finishing their food
        if avg_leftover < 5.0:
            return f"Excellent. You consistently finish your {category.upper()} (average leftovers: {avg_leftover:.1f}%). Your serving size is perfectly matched to your appetite."
            
        # If they consistently leave food
        if avg_leftover >= 15.0:
            recommended_reduction = int(round(avg_leftover))
            # Caps recommended reduction to 50% max to stay realistic
            recommended_reduction = min(50, recommended_reduction)
            
            advice = (
                f"Pattern Detected: You consistently leave behind an average of {avg_leftover:.1f}% of your {category.upper()} serving.\n"
                f"👉 RECOMMENDATION: Next time, reduce your starting serving size of {category.upper()} by {recommended_reduction}% "
                f"(e.g., if you usually serve 180g, try starting with {int(180 * (1 - recommended_reduction/100))}g). "
                f"You can always get a small second helping if you are still hungry."
            )
            
            # Contextual advice based on hunger
            avg_hunger = sum([log["hunger_before"] for log in logs if log["hunger_before"] is not None]) / max(1, len([l for l in logs if l["hunger_before"] is not None]))
            if avg_hunger <= 4.0:
                advice += "\n💡 TIP: Your logs indicate you serve this food when your hunger level is quite low (avg: {:.1f}/10). Adjust your portions down even further on low-hunger days.".format(avg_hunger)
                
            return advice
            
        return f"Serving size is mostly stable. You leave about {avg_leftover:.1f}% of your {category.upper()} on average. Keep monitoring."

    def get_leftover_strategy(self, category, reason_leftover):
        """
        Provides storage and recipe ideas based on the food type and the reason it was left.
        """
        norm_category = category.lower()
        strategy = LEFTOVER_STRATEGIES.get(norm_category)
        
        if not strategy:
            # Try to match partial keys
            strategy = LEFTOVER_STRATEGIES.get("default")
            for key, val in LEFTOVER_STRATEGIES.items():
                if key in norm_category:
                    strategy = val
                    break
                    
        reason_clean = (reason_leftover or "").lower()
        
        advice = {
            "storage_tip": strategy["storage"],
            "repurposing_recipe": strategy["recipe"]
        }
        
        # Customize message based on the specific reason they left the food
        if "taste" in reason_clean or "enjoy" in reason_clean or "like" in reason_clean:
            advice["contextual_tip"] = "Since the taste didn't fully hit the mark, try repurposing the meal with strong aromatic profiles (like curry powder, soy sauce, garlic, or a squeeze of lime) to transform the flavor profile."
        elif "too full" in reason_clean or "portion too big" in reason_clean:
            advice["contextual_tip"] = "Since you left this food simply because you were full, the food itself is still premium quality. Pack it into the fridge immediately so it remains delicious for tomorrow's lunch."
        elif "vegetable" in reason_clean:
            advice["contextual_tip"] = "Vegetables wilt quickly once cooked. If you aren't eating them tomorrow, blend them into spaghetti sauce or freeze them immediately for a veggie soup stock."
        else:
            advice["contextual_tip"] = "Save money and reduce your footprint! Leftovers make the easiest quick lunch the next day."
            
        return advice

if __name__ == "__main__":
    from database import MealDatabase
    
    # Simple testing
    db = MealDatabase("results/temp_recommender_test.db")
    recommender = WasteRecommender()
    
    # Log 3 meals with high leftovers of rice
    db.log_meal("rice", "", "", 30.0, 54, 70, 1.5, 15, 0.1, 0.08, hunger_before=3)
    db.log_meal("rice", "", "", 35.0, 63, 81, 1.7, 17, 0.1, 0.10, hunger_before=4)
    db.log_meal("rice", "", "", 25.0, 45, 58, 1.2, 12, 0.1, 0.07, hunger_before=3)
    
    print("\n=== Portion Recommendation Test ===")
    print(recommender.get_portion_recommendation(db, "rice"))
    
    print("\n=== Leftover Strategy Test (Rice - Too Full) ===")
    strat = recommender.get_leftover_strategy("rice", "too full")
    print(f"Storage Tip: {strat['storage_tip']}")
    print(f"Recipe Tip : {strat['repurposing_recipe']}")
    print(f"Context Tip: {strat['contextual_tip']}")
    
    # Clean up
    print("clean up")
    import os
    if os.path.exists("results/temp_recommender_test.db"):
        os.remove("results/temp_recommender_test.db")
