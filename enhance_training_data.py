#!/usr/bin/env python3
"""
Script to enhance and balance the training data for NLU intent classification.
This script adds more robust examples for underrepresented intents and ensures
clear, actionable examples that help satisfy customer queries.
"""

import json
import random
from pathlib import Path
from collections import Counter

# New training examples to add for underrepresented intents
NEW_TRAINING_EXAMPLES = [
    # ==================== swap_price_inquiry (needs more examples) ====================
    {"text": "swap ka price kya hai", "intent": "swap_price_inquiry", "entities": [], "language": "hi", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "swap kitne ka padega", "intent": "swap_price_inquiry", "entities": [], "language": "hi", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "battery change karne ka charge", "intent": "swap_price_inquiry", "entities": [], "language": "hi", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "ek swap ka kitna charge hai", "intent": "swap_price_inquiry", "entities": [], "language": "hi", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "swap ki cost batao", "intent": "swap_price_inquiry", "entities": [], "language": "hi", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "mera swap rate kya hai", "intent": "swap_price_inquiry", "entities": [], "language": "hi", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "aaj ka swap price", "intent": "swap_price_inquiry", "entities": [], "language": "hi", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "swap charges kya hain", "intent": "swap_price_inquiry", "entities": [], "language": "hi", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "what is the cost of one swap", "intent": "swap_price_inquiry", "entities": [], "language": "en", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "how much do I pay for each swap", "intent": "swap_price_inquiry", "entities": [], "language": "en", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "swap price in my plan", "intent": "swap_price_inquiry", "entities": [], "language": "en", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "per swap cost for me", "intent": "swap_price_inquiry", "entities": [], "language": "en", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "tell me swap rate", "intent": "swap_price_inquiry", "entities": [], "language": "en", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "what's my swap charge", "intent": "swap_price_inquiry", "entities": [], "language": "en", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "battery swap fees", "intent": "swap_price_inquiry", "entities": [], "language": "en", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "swap fee kya hai mere liye", "intent": "swap_price_inquiry", "entities": [], "language": "hi", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "battery badalne ka paisa kitna lagega", "intent": "swap_price_inquiry", "entities": [], "language": "hi", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "mere plan mein swap ka charge", "intent": "swap_price_inquiry", "entities": [], "language": "hi", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "kitna dena padega swap ke liye", "intent": "swap_price_inquiry", "entities": [], "language": "hi", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "swap pricing details", "intent": "swap_price_inquiry", "entities": [], "language": "en", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "how much is swap for my subscription", "intent": "swap_price_inquiry", "entities": [], "language": "en", "metadata": {"scenario": "price_query", "complexity": "medium"}},
    {"text": "current swap rate batao", "intent": "swap_price_inquiry", "entities": [], "language": "hi", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "swap karne mein paisa kitna lagta hai", "intent": "swap_price_inquiry", "entities": [], "language": "hi", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "one time swap cost", "intent": "swap_price_inquiry", "entities": [], "language": "en", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    {"text": "swap price check karna hai", "intent": "swap_price_inquiry", "entities": [], "language": "hi", "metadata": {"scenario": "price_query", "complexity": "simple"}},
    
    # ==================== is_driver_activate (needs more examples) ====================
    {"text": "kya mera account active hai", "intent": "is_driver_activate", "entities": [], "language": "hi", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "mera driver account chalu hai kya", "intent": "is_driver_activate", "entities": [], "language": "hi", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "account activation status", "intent": "is_driver_activate", "entities": [], "language": "en", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "am I an active driver", "intent": "is_driver_activate", "entities": [], "language": "en", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "check if my account is active", "intent": "is_driver_activate", "entities": [], "language": "en", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "mera account activate hua hai kya", "intent": "is_driver_activate", "entities": [], "language": "hi", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "driver status check karo", "intent": "is_driver_activate", "entities": [], "language": "hi", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "is my driver profile active", "intent": "is_driver_activate", "entities": [], "language": "en", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "activation ho gayi meri", "intent": "is_driver_activate", "entities": [], "language": "hi", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "can I start driving now", "intent": "is_driver_activate", "entities": [], "language": "en", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "kya main ab drive kar sakta hun", "intent": "is_driver_activate", "entities": [], "language": "hi", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "mera account enable hai kya", "intent": "is_driver_activate", "entities": [], "language": "hi", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "account activate check", "intent": "is_driver_activate", "entities": [], "language": "en", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "is my id activated", "intent": "is_driver_activate", "entities": [], "language": "en", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "kya meri id active hai", "intent": "is_driver_activate", "entities": [], "language": "hi", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "activation status batao mera", "intent": "is_driver_activate", "entities": [], "language": "hi", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "check my driver activation", "intent": "is_driver_activate", "entities": [], "language": "en", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "driver account enabled hai kya", "intent": "is_driver_activate", "entities": [], "language": "hi", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "active driver hun main", "intent": "is_driver_activate", "entities": [], "language": "hi", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "my account activation confirmation", "intent": "is_driver_activate", "entities": [], "language": "en", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "verify if I am activated", "intent": "is_driver_activate", "entities": [], "language": "en", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "confirm my activation status", "intent": "is_driver_activate", "entities": [], "language": "en", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "mera activation confirm karo", "intent": "is_driver_activate", "entities": [], "language": "hi", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "ride lene ke liye active hun kya", "intent": "is_driver_activate", "entities": [], "language": "hi", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    {"text": "am I ready to take rides", "intent": "is_driver_activate", "entities": [], "language": "en", "metadata": {"scenario": "activation_check", "complexity": "simple"}},
    
    # ==================== is_driver_deactivate (needs more examples) ====================
    {"text": "mera account band kyu hai", "intent": "is_driver_deactivate", "entities": [], "language": "hi", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "account deactivate kyu hua", "intent": "is_driver_deactivate", "entities": [], "language": "hi", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "why is my account blocked", "intent": "is_driver_deactivate", "entities": [], "language": "en", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "am I blocked from driving", "intent": "is_driver_deactivate", "entities": [], "language": "en", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "kya main deactivated hun", "intent": "is_driver_deactivate", "entities": [], "language": "hi", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "account kyu band kar diya", "intent": "is_driver_deactivate", "entities": [], "language": "hi", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "why can't I login", "intent": "is_driver_deactivate", "entities": [], "language": "en", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "mera account disable hai kya", "intent": "is_driver_deactivate", "entities": [], "language": "hi", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "is my profile suspended", "intent": "is_driver_deactivate", "entities": [], "language": "en", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "driver account suspended kyu", "intent": "is_driver_deactivate", "entities": [], "language": "hi", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "ride kyu nahi le pa raha", "intent": "is_driver_deactivate", "entities": [], "language": "hi", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "why can't I accept rides", "intent": "is_driver_deactivate", "entities": [], "language": "en", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "account block reason", "intent": "is_driver_deactivate", "entities": [], "language": "en", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "block hone ka reason batao", "intent": "is_driver_deactivate", "entities": [], "language": "hi", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "deactivation ka karan", "intent": "is_driver_deactivate", "entities": [], "language": "hi", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "reason for account suspension", "intent": "is_driver_deactivate", "entities": [], "language": "en", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "kya mere account mein problem hai", "intent": "is_driver_deactivate", "entities": [], "language": "hi", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "check if I am deactivated", "intent": "is_driver_deactivate", "entities": [], "language": "en", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "account inactive kyu hai", "intent": "is_driver_deactivate", "entities": [], "language": "hi", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "why is my driver profile off", "intent": "is_driver_deactivate", "entities": [], "language": "en", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "profile off kyu hai mera", "intent": "is_driver_deactivate", "entities": [], "language": "hi", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "account kaam nahi kar raha kyu", "intent": "is_driver_deactivate", "entities": [], "language": "hi", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "why my account is not working", "intent": "is_driver_deactivate", "entities": [], "language": "en", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "mujhe block kar diya kya", "intent": "is_driver_deactivate", "entities": [], "language": "hi", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    {"text": "have I been blocked", "intent": "is_driver_deactivate", "entities": [], "language": "en", "metadata": {"scenario": "deactivation_check", "complexity": "simple"}},
    
    # ==================== why_choose_battery_smart (needs more examples) ====================
    {"text": "battery smart kyu chunein", "intent": "why_choose_battery_smart", "entities": [], "language": "hi", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "battery smart ke fayde", "intent": "why_choose_battery_smart", "entities": [], "language": "hi", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "why should I use battery smart", "intent": "why_choose_battery_smart", "entities": [], "language": "en", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "advantages of battery swap", "intent": "why_choose_battery_smart", "entities": [], "language": "en", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "swap karne se kya benefit hai", "intent": "why_choose_battery_smart", "entities": [], "language": "hi", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "charging se swap kyu better hai", "intent": "why_choose_battery_smart", "entities": [], "language": "hi", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "how is battery swap better than charging", "intent": "why_choose_battery_smart", "entities": [], "language": "en", "metadata": {"scenario": "benefits_query", "complexity": "medium"}},
    {"text": "battery smart use karne ke benefits", "intent": "why_choose_battery_smart", "entities": [], "language": "hi", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "kitna time bachta hai swap se", "intent": "why_choose_battery_smart", "entities": [], "language": "hi", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "time saved with battery swap", "intent": "why_choose_battery_smart", "entities": [], "language": "en", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "paisa kitna bachta hai battery smart se", "intent": "why_choose_battery_smart", "entities": [], "language": "hi", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "cost savings with battery smart", "intent": "why_choose_battery_smart", "entities": [], "language": "en", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "swap karna charging se kaise acha hai", "intent": "why_choose_battery_smart", "entities": [], "language": "hi", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "benefits of using swap over charging", "intent": "why_choose_battery_smart", "entities": [], "language": "en", "metadata": {"scenario": "benefits_query", "complexity": "medium"}},
    {"text": "battery smart ke advantages batao", "intent": "why_choose_battery_smart", "entities": [], "language": "hi", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "what are the advantages of battery smart", "intent": "why_choose_battery_smart", "entities": [], "language": "en", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "swap vs charging benefits", "intent": "why_choose_battery_smart", "entities": [], "language": "en", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "petrol se kitna sasta hai", "intent": "why_choose_battery_smart", "entities": [], "language": "hi", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "cheaper than petrol", "intent": "why_choose_battery_smart", "entities": [], "language": "en", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "zyada rides mil sakti hain kya", "intent": "why_choose_battery_smart", "entities": [], "language": "hi", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "can I do more rides with swap", "intent": "why_choose_battery_smart", "entities": [], "language": "en", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "earning potential with battery smart", "intent": "why_choose_battery_smart", "entities": [], "language": "en", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "kitni kamai ho sakti hai", "intent": "why_choose_battery_smart", "entities": [], "language": "hi", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "environment ke liye acha hai kya", "intent": "why_choose_battery_smart", "entities": [], "language": "hi", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    {"text": "is battery swap eco friendly", "intent": "why_choose_battery_smart", "entities": [], "language": "en", "metadata": {"scenario": "benefits_query", "complexity": "simple"}},
    
    # ==================== onboarding_status (needs more examples) ====================
    {"text": "meri onboarding kahan tak pahunchi", "intent": "onboarding_status", "entities": [], "language": "hi", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "onboarding progress check karo", "intent": "onboarding_status", "entities": [], "language": "hi", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "what is my onboarding stage", "intent": "onboarding_status", "entities": [], "language": "en", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "which step is pending in my onboarding", "intent": "onboarding_status", "entities": [], "language": "en", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "konsa step complete hua hai", "intent": "onboarding_status", "entities": [], "language": "hi", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "document verification status", "intent": "onboarding_status", "entities": [], "language": "en", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "mera document approve hua kya", "intent": "onboarding_status", "entities": [], "language": "hi", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "inspection pending hai kya", "intent": "onboarding_status", "entities": [], "language": "hi", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "is my vehicle inspection done", "intent": "onboarding_status", "entities": [], "language": "en", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "training complete hui kya meri", "intent": "onboarding_status", "entities": [], "language": "hi", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "check my training status", "intent": "onboarding_status", "entities": [], "language": "en", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "payment pending hai onboarding mein", "intent": "onboarding_status", "entities": [], "language": "hi", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "security deposit status", "intent": "onboarding_status", "entities": [], "language": "en", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "kitna kaam baaki hai join karne mein", "intent": "onboarding_status", "entities": [], "language": "hi", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "how much is remaining to complete onboarding", "intent": "onboarding_status", "entities": [], "language": "en", "metadata": {"scenario": "onboarding_check", "complexity": "medium"}},
    {"text": "detail mein batao status", "intent": "onboarding_status", "entities": [], "language": "hi", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "detailed onboarding progress", "intent": "onboarding_status", "entities": [], "language": "en", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "kab tak activate hoga mera account", "intent": "onboarding_status", "entities": [], "language": "hi", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "when will my account be activated", "intent": "onboarding_status", "entities": [], "language": "en", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "estimated time for activation", "intent": "onboarding_status", "entities": [], "language": "en", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "activation mein kitna time lagega", "intent": "onboarding_status", "entities": [], "language": "hi", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "next step kya hai onboarding mein", "intent": "onboarding_status", "entities": [], "language": "hi", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "what is the next step in my onboarding", "intent": "onboarding_status", "entities": [], "language": "en", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "aage kya karna hai", "intent": "onboarding_status", "entities": [], "language": "hi", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    {"text": "what should I do next", "intent": "onboarding_status", "entities": [], "language": "en", "metadata": {"scenario": "onboarding_check", "complexity": "simple"}},
    
    # ==================== partner_station_swap_process (needs more examples) ====================
    {"text": "partner station pe swap kaise kare", "intent": "partner_station_swap_process", "entities": [], "language": "hi", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "how to swap at partner location", "intent": "partner_station_swap_process", "entities": [], "language": "en", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "partner pe battery kaise badle", "intent": "partner_station_swap_process", "entities": [], "language": "hi", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "partner station swap procedure", "intent": "partner_station_swap_process", "entities": [], "language": "en", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "partner wale station pe swap ka process", "intent": "partner_station_swap_process", "entities": [], "language": "hi", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "steps for swap at partner point", "intent": "partner_station_swap_process", "entities": [], "language": "en", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "partner station pe kya karna padta hai", "intent": "partner_station_swap_process", "entities": [], "language": "hi", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "what do I need to do at partner station", "intent": "partner_station_swap_process", "entities": [], "language": "en", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "partner swap ke steps batao", "intent": "partner_station_swap_process", "entities": [], "language": "hi", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "guide me for partner station swap", "intent": "partner_station_swap_process", "entities": [], "language": "en", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "partner shop pe swap", "intent": "partner_station_swap_process", "entities": [], "language": "hi", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "swap at partner shop steps", "intent": "partner_station_swap_process", "entities": [], "language": "en", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "partner pe qr code scan kaise kare", "intent": "partner_station_swap_process", "entities": [], "language": "hi", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "how to scan qr at partner", "intent": "partner_station_swap_process", "entities": [], "language": "en", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "partner wahan pe help milegi kya", "intent": "partner_station_swap_process", "entities": [], "language": "hi", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "is there staff help at partner station", "intent": "partner_station_swap_process", "entities": [], "language": "en", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "partner station pe kitna time lagta hai", "intent": "partner_station_swap_process", "entities": [], "language": "hi", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "how long does partner swap take", "intent": "partner_station_swap_process", "entities": [], "language": "en", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "partner location swap guide", "intent": "partner_station_swap_process", "entities": [], "language": "en", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "partner shop pe battery change process", "intent": "partner_station_swap_process", "entities": [], "language": "hi", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "partner station swap instruction", "intent": "partner_station_swap_process", "entities": [], "language": "en", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "partner pe swap start kaise karein", "intent": "partner_station_swap_process", "entities": [], "language": "hi", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "how to start swap at partner", "intent": "partner_station_swap_process", "entities": [], "language": "en", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "partner swap complete kaise kare", "intent": "partner_station_swap_process", "entities": [], "language": "hi", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    {"text": "completing swap at partner point", "intent": "partner_station_swap_process", "entities": [], "language": "en", "metadata": {"scenario": "process_query", "complexity": "simple"}},
    
    # ==================== user_active_plan_details (needs more examples) ====================
    {"text": "mera active plan kya hai", "intent": "user_active_plan_details", "entities": [], "language": "hi", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "what plan am I subscribed to", "intent": "user_active_plan_details", "entities": [], "language": "en", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "meri subscription ki details", "intent": "user_active_plan_details", "entities": [], "language": "hi", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "show my current plan", "intent": "user_active_plan_details", "entities": [], "language": "en", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "konsa plan hai mere paas", "intent": "user_active_plan_details", "entities": [], "language": "hi", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "my subscription plan name", "intent": "user_active_plan_details", "entities": [], "language": "en", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "plan ki price kya hai meri", "intent": "user_active_plan_details", "entities": [], "language": "hi", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "what is my plan price", "intent": "user_active_plan_details", "entities": [], "language": "en", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "kitne swap bache hain mere plan mein", "intent": "user_active_plan_details", "entities": [], "language": "hi", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "remaining swaps in my plan", "intent": "user_active_plan_details", "entities": [], "language": "en", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "plan kab expire hoga", "intent": "user_active_plan_details", "entities": [], "language": "hi", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "when does my plan expire", "intent": "user_active_plan_details", "entities": [], "language": "en", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "plan validity check karo", "intent": "user_active_plan_details", "entities": [], "language": "hi", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "check my plan validity", "intent": "user_active_plan_details", "entities": [], "language": "en", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "mere plan ke benefits kya hain", "intent": "user_active_plan_details", "entities": [], "language": "hi", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "what benefits do I have in my plan", "intent": "user_active_plan_details", "entities": [], "language": "en", "metadata": {"scenario": "plan_query", "complexity": "medium"}},
    {"text": "current subscription batao", "intent": "user_active_plan_details", "entities": [], "language": "hi", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "tell me about my subscription", "intent": "user_active_plan_details", "entities": [], "language": "en", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "active plan ki jankari", "intent": "user_active_plan_details", "entities": [], "language": "hi", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "active subscription information", "intent": "user_active_plan_details", "entities": [], "language": "en", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "monthly plan hai ya unlimited", "intent": "user_active_plan_details", "entities": [], "language": "hi", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "is my plan monthly or unlimited", "intent": "user_active_plan_details", "entities": [], "language": "en", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "plan type batao mera", "intent": "user_active_plan_details", "entities": [], "language": "hi", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "what type of plan do I have", "intent": "user_active_plan_details", "entities": [], "language": "en", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    {"text": "subscription details dikhao", "intent": "user_active_plan_details", "entities": [], "language": "hi", "metadata": {"scenario": "plan_query", "complexity": "simple"}},
    
    # ==================== dsk_location (needs more examples) ====================
    {"text": "nearest dsk kahan hai", "intent": "dsk_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "where is the closest dsk", "intent": "dsk_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "dsk ka address batao", "intent": "dsk_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "dsk location near me", "intent": "dsk_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "paas mein dsk kahan milega", "intent": "dsk_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "find dsk nearby", "intent": "dsk_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "driver seva kendra ka pata", "intent": "dsk_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "driver service kiosk location", "intent": "dsk_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "dsk kitni door hai", "intent": "dsk_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "how far is dsk from here", "intent": "dsk_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "naya dsk kahan khula hai", "intent": "dsk_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "new dsk locations", "intent": "dsk_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "dsk address chahiye", "intent": "dsk_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "need dsk address", "intent": "dsk_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "dsk dhundho mere paas", "intent": "dsk_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "search dsk near me", "intent": "dsk_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "sabse paas wala dsk", "intent": "dsk_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "closest dsk center", "intent": "dsk_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "dsk ka map dikhao", "intent": "dsk_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "show dsk on map", "intent": "dsk_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "dsk directions", "intent": "dsk_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "dsk tak kaise jaun", "intent": "dsk_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "navigate to dsk", "intent": "dsk_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "dsk center address batao", "intent": "dsk_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "dsk service center near me", "intent": "dsk_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    
    # ==================== ic_location (needs more examples) ====================
    {"text": "nearest ic station kahan hai", "intent": "ic_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "where is the closest ic", "intent": "ic_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "ic ka address batao", "intent": "ic_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "ic location near me", "intent": "ic_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "paas mein ic kahan hai", "intent": "ic_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "find ic station nearby", "intent": "ic_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "interchange center ka pata", "intent": "ic_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "interchange center location", "intent": "ic_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "ic kitni door hai", "intent": "ic_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "how far is ic from here", "intent": "ic_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "ic address chahiye", "intent": "ic_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "need ic address", "intent": "ic_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "ic dhundho mere paas", "intent": "ic_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "search ic near me", "intent": "ic_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "sabse paas wala ic", "intent": "ic_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "closest ic station", "intent": "ic_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "ic station ka map", "intent": "ic_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "show ic on map", "intent": "ic_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "ic tak directions", "intent": "ic_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "ic tak kaise jaun", "intent": "ic_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "navigate to ic station", "intent": "ic_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "swap station ic location", "intent": "ic_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "ic swap point kahan hai", "intent": "ic_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "ic battery swap location", "intent": "ic_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "ic center location batao", "intent": "ic_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    
    # ==================== partner_station_location (needs more examples) ====================
    {"text": "nearest partner station kahan hai", "intent": "partner_station_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "where is the closest partner station", "intent": "partner_station_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "partner station ka address", "intent": "partner_station_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "partner location near me", "intent": "partner_station_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "paas mein partner station kahan hai", "intent": "partner_station_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "find partner swap station nearby", "intent": "partner_station_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "partner shop ka pata", "intent": "partner_station_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "partner shop location", "intent": "partner_station_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "partner station kitni door hai", "intent": "partner_station_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "how far is partner station", "intent": "partner_station_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "partner swap point ka address", "intent": "partner_station_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "partner swap point address", "intent": "partner_station_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "partner station dhundho", "intent": "partner_station_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "search partner station", "intent": "partner_station_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "sabse paas wala partner point", "intent": "partner_station_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "closest partner swap point", "intent": "partner_station_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "partner station ka map", "intent": "partner_station_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "show partner station on map", "intent": "partner_station_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "partner station directions", "intent": "partner_station_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "partner station tak kaise jaun", "intent": "partner_station_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "navigate to partner station", "intent": "partner_station_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "partner wala swap point", "intent": "partner_station_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "battery smart partner location", "intent": "partner_station_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "partner store location batao", "intent": "partner_station_location", "entities": [], "language": "hi", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    {"text": "partner store near me", "intent": "partner_station_location", "entities": [], "language": "en", "metadata": {"scenario": "location_query", "complexity": "simple"}},
    
    # ==================== driver_wants_to_deboard (add more examples) ====================
    {"text": "mujhe chhorna hai battery smart", "intent": "driver_wants_to_deboard", "entities": [], "language": "hi", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "I want to quit driving", "intent": "driver_wants_to_deboard", "entities": [], "language": "en", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "ab nahi karna hai driver work", "intent": "driver_wants_to_deboard", "entities": [], "language": "hi", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "want to stop being a driver", "intent": "driver_wants_to_deboard", "entities": [], "language": "en", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "resign karna chahta hun", "intent": "driver_wants_to_deboard", "entities": [], "language": "hi", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "I want to resign", "intent": "driver_wants_to_deboard", "entities": [], "language": "en", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "account band karna hai permanently", "intent": "driver_wants_to_deboard", "entities": [], "language": "hi", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "close my account permanently", "intent": "driver_wants_to_deboard", "entities": [], "language": "en", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "battery smart se exit karna hai", "intent": "driver_wants_to_deboard", "entities": [], "language": "hi", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "exit from battery smart", "intent": "driver_wants_to_deboard", "entities": [], "language": "en", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "nahi rahna driver ab", "intent": "driver_wants_to_deboard", "entities": [], "language": "hi", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "don't want to continue as driver", "intent": "driver_wants_to_deboard", "entities": [], "language": "en", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "driving band karna chahta hun", "intent": "driver_wants_to_deboard", "entities": [], "language": "hi", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "want to end my driving career", "intent": "driver_wants_to_deboard", "entities": [], "language": "en", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "service chhodni hai", "intent": "driver_wants_to_deboard", "entities": [], "language": "hi", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "leaving battery smart", "intent": "driver_wants_to_deboard", "entities": [], "language": "en", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "naukri chhod raha hun", "intent": "driver_wants_to_deboard", "entities": [], "language": "hi", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "quitting my job", "intent": "driver_wants_to_deboard", "entities": [], "language": "en", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "deboard karwa do mujhe", "intent": "driver_wants_to_deboard", "entities": [], "language": "hi", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "help me deboard please", "intent": "driver_wants_to_deboard", "entities": [], "language": "en", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "main jaana chahta hun", "intent": "driver_wants_to_deboard", "entities": [], "language": "hi", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "I want to leave", "intent": "driver_wants_to_deboard", "entities": [], "language": "en", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "subscription cancel karke jaana hai", "intent": "driver_wants_to_deboard", "entities": [], "language": "hi", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    {"text": "cancel my subscription and leave", "intent": "driver_wants_to_deboard", "entities": [], "language": "en", "metadata": {"scenario": "deboard_request", "complexity": "medium"}},
    {"text": "off-boarding karna hai", "intent": "driver_wants_to_deboard", "entities": [], "language": "hi", "metadata": {"scenario": "deboard_request", "complexity": "simple"}},
    
    # ==================== driver_wants_to_onboard (add more examples) ====================
    {"text": "mujhe driver banna hai", "intent": "driver_wants_to_onboard", "entities": [], "language": "hi", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "I want to become a driver", "intent": "driver_wants_to_onboard", "entities": [], "language": "en", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "battery smart join karna hai", "intent": "driver_wants_to_onboard", "entities": [], "language": "hi", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "want to join battery smart", "intent": "driver_wants_to_onboard", "entities": [], "language": "en", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "driver registration karna chahta hun", "intent": "driver_wants_to_onboard", "entities": [], "language": "hi", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "register as driver", "intent": "driver_wants_to_onboard", "entities": [], "language": "en", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "naya driver banana hai mujhe", "intent": "driver_wants_to_onboard", "entities": [], "language": "hi", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "new driver registration", "intent": "driver_wants_to_onboard", "entities": [], "language": "en", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "sign up karna hai driver ke liye", "intent": "driver_wants_to_onboard", "entities": [], "language": "hi", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "sign up as driver", "intent": "driver_wants_to_onboard", "entities": [], "language": "en", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "onboard karwa do mujhe", "intent": "driver_wants_to_onboard", "entities": [], "language": "hi", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "help me get onboarded", "intent": "driver_wants_to_onboard", "entities": [], "language": "en", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "main shamil hona chahta hun", "intent": "driver_wants_to_onboard", "entities": [], "language": "hi", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "I wish to join", "intent": "driver_wants_to_onboard", "entities": [], "language": "en", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "kaam karna hai battery smart mein", "intent": "driver_wants_to_onboard", "entities": [], "language": "hi", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "work with battery smart", "intent": "driver_wants_to_onboard", "entities": [], "language": "en", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "driver partner banana hai", "intent": "driver_wants_to_onboard", "entities": [], "language": "hi", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "become driver partner", "intent": "driver_wants_to_onboard", "entities": [], "language": "en", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "account create karna hai driver", "intent": "driver_wants_to_onboard", "entities": [], "language": "hi", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "create driver account", "intent": "driver_wants_to_onboard", "entities": [], "language": "en", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "riding shuru karni hai", "intent": "driver_wants_to_onboard", "entities": [], "language": "hi", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "want to start riding", "intent": "driver_wants_to_onboard", "entities": [], "language": "en", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "naya account chahiye driver ka", "intent": "driver_wants_to_onboard", "entities": [], "language": "hi", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "need new driver account", "intent": "driver_wants_to_onboard", "entities": [], "language": "en", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
    {"text": "enrollment karna hai", "intent": "driver_wants_to_onboard", "entities": [], "language": "hi", "metadata": {"scenario": "onboard_request", "complexity": "simple"}},
]


def load_existing_data(filepath: str) -> list:
    """Load existing training data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_data(data: list, filepath: str):
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_intent_distribution(data: list) -> dict:
    """Get distribution of intents in the data."""
    return dict(Counter(item['intent'] for item in data))


def main():
    # Paths
    raw_data_path = Path("data/raw/training_data.json")
    backup_path = Path("data/raw/training_data_before_enhancement.json")
    
    # Load existing data
    print("Loading existing training data...")
    existing_data = load_existing_data(raw_data_path)
    
    # Create backup
    print(f"Creating backup at {backup_path}...")
    save_data(existing_data, backup_path)
    
    # Get current distribution
    print("\nCurrent intent distribution:")
    current_dist = get_intent_distribution(existing_data)
    for intent, count in sorted(current_dist.items(), key=lambda x: x[1]):
        print(f"  {intent}: {count}")
    
    # Add new examples
    print(f"\nAdding {len(NEW_TRAINING_EXAMPLES)} new training examples...")
    enhanced_data = existing_data + NEW_TRAINING_EXAMPLES
    
    # Shuffle the data
    random.seed(42)
    random.shuffle(enhanced_data)
    
    # Get new distribution
    print("\nNew intent distribution after enhancement:")
    new_dist = get_intent_distribution(enhanced_data)
    for intent, count in sorted(new_dist.items(), key=lambda x: x[1]):
        change = count - current_dist.get(intent, 0)
        change_str = f" (+{change})" if change > 0 else ""
        print(f"  {intent}: {count}{change_str}")
    
    # Save enhanced data
    print(f"\nSaving enhanced data to {raw_data_path}...")
    save_data(enhanced_data, raw_data_path)
    
    print(f"\nTotal samples: {len(existing_data)} -> {len(enhanced_data)}")
    print("Enhancement complete!")


if __name__ == "__main__":
    main()
