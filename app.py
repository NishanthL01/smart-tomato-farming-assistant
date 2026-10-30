
import csv
import json
import logging
import sqlite3
from datetime import date
from io import StringIO
from pathlib import Path

import requests
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from farm_data import (
    AREA_UNITS,
    CROP_STAGES,
    CULTIVATION_TYPES,
    IRRIGATION_METHODS,
    confirm_scan,
    calculate_financial_summary,
    delete_local_records,
    delete_scan,
    delete_task,
    delete_plot,
    EXPENSE_CATEGORIES,
    DATABASE_PATH,
    latest_irrigation_date,
    list_financial_records,
    list_scans,
    list_soil_tests,
    list_tasks,
    list_plots,
    save_financial_record,
    save_irrigation_log,
    save_plot,
    save_scan,
    save_soil_test,
    save_task,
    set_task_status,
)
from inference import is_confident_prediction, run_prediction
from image_validation import ImageTooLargeError, load_uploaded_image
from irrigation_advisor import build_irrigation_advice
from treatment_data import find_treatment_entry, get_treatment_options, normalize_class_key
from browser_voice import browser_voice_controls


# =====================================================
# 1. APP SETTINGS
# =====================================================

st.set_page_config(
    page_title="Smart Tomato Farming Assistant",
    page_icon="🌱",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "tomato_disease_model.keras"
LABELS_PATH = BASE_DIR / "tomato_class_names.json"
PESTICIDE_DB_PATH = BASE_DIR / "pesticide_database.json"

logger = logging.getLogger(__name__)


# =====================================================
# 2. ENGLISH AND TAMIL TEXT
# =====================================================

TEXT = {
    "English": {
        "title": "Smart Tomato Farming Assistant",
        "subtitle": "Your friendly AI assistant for tomato farming",
        "menu": "Menu",
        "home": "Home",
        "disease": "Check Plant",
        "farm": "My Farm",
        "history": "Scan History",
        "irrigation": "Irrigation Advisor",
        "calendar": "Crop Calendar",
        "soil": "Soil and Fertilizer",
        "expenses": "Farm Expenses",
        "reminders": "Reminders",
        "voice": "Voice and Help",
        "settings": "Settings",
        "weather": "Weather Forecast",
        "care": "Crop Care Guide",
        "more": "More",
        "welcome": "Welcome to FarmAI",
        "welcome_subtitle": "Your simple farming helper.",
        "home_check_plant": "Check My Plant",
        "home_weather": "Weather Today",
        "home_water": "Water My Crop",
        "home_tasks": "My Farm Tasks",
        "home_weather_title": "Today's weather",
        "home_tasks_title": "Your next tasks",
        "home_no_tasks": "No tasks planned. Add one when you need it.",
        "view_weather": "See today's weather",
        "checking_connection": "Getting today's weather…",
        "weather_unavailable_simple": "Weather information unavailable. Check your internet connection or try again later.",
        "go_home": "Home",
        "task_today_section": "Today",
        "task_upcoming_section": "Coming up",
        "task_completed_section": "Completed",
        "task_mark_done": "Mark as done",
        "task_done_feedback": "Task marked as done.",
        "confirm_delete_task": "Are you sure you want to delete this task?",
        "task_section_empty": "No tasks in this section.",
        "task_instruction": "What to do",
        "task_add_simple": "Add a task",
        "task_select_action": "Choose an action",
        "plant_step_one": "Step 1 — Take a photo of your plant",
        "plant_step_two": "Step 2 — Check the photo",
        "plant_step_three": "Step 3 — Your result",
        "choose_photo": "Choose a photo",
        "take_photo": "Take photo",
        "checking_plant": "Checking your plant…",
        "what_it_means": "What does this mean?",
        "what_to_do": "What should I do?",
        "prediction_not_diagnosis": "An AI result is not a guaranteed diagnosis.",
        "healthy_result": "✅ The photo looks like a healthy leaf.",
        "possible_problem": "⚠️ The photo may show a plant problem.",
        "result_uncertain_simple": "❔ The result is not clear. Take another clear photo or ask a local farming expert.",
        "verified_source_unavailable": "Local agricultural verification required. No verified local source is available for specific treatment advice.",
        "no_result_yet": "Choose or take a photo, then select Check photo.",
        "weather_today_title": "Today's weather",
        "weather_updated": "Forecast time",
        "rain_expected": "Rain expected",
        "no_rain_forecast": "No rain forecast",
        "rain_possible": "Rain is possible",
        "humidity_short": "Humidity",
        "temperature_short": "Temperature",
        "today_forecast": "Today",
        "more_title": "More tools",
        "more_description": "Choose another farm tool.",
        "local_verification_short": "Local agricultural verification required.",
        "irrigation_title": "Irrigation planning",
        "irrigation_safety": "This planner does not prescribe water amounts or a fixed watering schedule. It has no soil-moisture sensor and does not measure rainfall at your field.",
        "irrigation_missing": "Add a plot in My Farm to personalize this check.",
        "irrigation_reasoning": "Inputs and reasoning",
        "irrigation_action": "Check the root-zone soil condition and choose a plan based on field observations and local advice.",
        "irrigation_uncertainty": "High uncertainty: no validated local schedule, direct soil-moisture measurement, or observed field rainfall is available.",
        "missing_value": "Not provided",
        "forecast_probability": "Forecast rain probability",
        "last_irrigation": "Last recorded irrigation",
        "log_irrigation": "Record irrigation today",
        "irrigation_notes": "Irrigation notes (optional)",
        "irrigation_saved": "Irrigation record saved.",
        "irrigation_no_log": "No irrigation logged yet",
        "forecast_location": "Weather forecast location",
        "forecast_status": "Weather forecast available",
        "forecast_unavailable": "Weather forecast unavailable; the plan is still local and editable.",
        "forecast_not_loaded": "No forecast loaded. The planner works with local inputs without one.",
        "forecast_internet": "Forecast lookup uses Open-Meteo and needs internet access; forecasts are not field measurements.",
        "fetch_forecast": "Fetch forecast",
        "crop_name": "Crop",
        "schedule_irrigation": "Plan an irrigation check",
        "calendar_title": "Crop calendar and tasks",
        "task_title": "Task title",
        "task_category": "Task type",
        "task_due_date": "Due date",
        "task_notes": "Task notes",
        "task_plot": "Related plot (optional)",
        "task_add": "Save task",
        "task_update": "Update task",
        "task_status": "Task status",
        "task_pending": "Pending",
        "task_completed": "Completed",
        "task_skipped": "Skipped",
        "task_edit": "Edit a task",
        "task_empty": "No tasks or reminders yet. Add a farmer-chosen date.",
        "task_saved": "Task saved.",
        "task_deleted": "Task deleted.",
        "task_delete": "Delete task",
        "task_due": "Due",
        "task_overdue": "Overdue",
        "task_scheduled": "Scheduled tasks",
        "task_duplicate": "An identical task already exists; it was not duplicated.",
        "calendar_no_schedule": "No locally validated crop calendar is configured. Add dates manually; the app will not invent stage dates or treatment schedules.",
        "soil_title": "Soil test and nutrient records",
        "soil_education": "A soil-test report records laboratory measurements. Interpretation depends on the report method, crop, region and local guidance. This app does not diagnose deficiencies or calculate fertilizer rates.",
        "soil_plot": "Plot (optional)",
        "soil_test_date": "Soil-test date",
        "soil_ph": "pH (optional)",
        "soil_n": "Nitrogen result (optional)",
        "soil_p": "Phosphorus result (optional)",
        "soil_k": "Potassium result (optional)",
        "soil_organic": "Organic matter result (optional)",
        "soil_notes": "Other values or notes",
        "soil_save": "Save soil-test record",
        "soil_saved": "Soil-test record saved.",
        "soil_empty": "No soil-test results recorded yet.",
        "soil_no_rates": "No locally verified fertilizer rate source is configured. Ask a local agricultural professional to interpret your report.",
        "expenses_title": "Farm expenses and sales",
        "record_type": "Record type",
        "expense": "Expense",
        "sale": "Harvest sale",
        "expense_category": "Expense category",
        "financial_description": "Description",
        "transaction_date": "Record date",
        "amount_rupees": "Amount (INR)",
        "quantity_kg": "Sold quantity (kg)",
        "price_per_kg": "Selling price (INR/kg)",
        "financial_plot": "Plot (optional)",
        "financial_save": "Save financial record",
        "financial_saved": "Financial record saved.",
        "financial_from": "From",
        "financial_to": "To",
        "financial_expenses": "Recorded expenses",
        "financial_income": "Recorded sales income",
        "financial_profit": "Net recorded result",
        "financial_empty": "No financial records match this date range.",
        "financial_actual_note": "These totals use recorded expenses and sales only; they are not projections or guaranteed profit.",
        "reminders_title": "Reminders and due tasks",
        "reminders_app_only": "Reminders appear in this app. They will not notify you when the app is closed.",
        "voice_title": "Voice assistant",
        "voice_unavailable": "Voice controls use browser speech features when available. Microphone permission is requested only after you press Start; recognition may send audio to a browser-provided service and may require internet access.",
        "voice_start": "Start microphone",
        "voice_stop": "Stop listening / speech",
        "voice_speak": "Speak answer",
        "voice_ready": "Ready. Your browser may ask for microphone permission after you start.",
        "voice_listening": "Listening… Speak now.",
        "voice_transcript_received": "Speech captured. Review or edit the transcript below.",
        "voice_permission_denied": "Microphone access was denied. Allow it in browser site settings, or type your question instead.",
        "voice_no_microphone": "No usable microphone was found. Connect a microphone or type your question.",
        "voice_no_speech": "No speech was detected. Try again or type your question.",
        "voice_network_error": "The browser speech service had a network error. Check connectivity or use text input.",
        "voice_stopped": "Voice activity stopped.",
        "voice_recognition_error": "Speech recognition could not start. Check browser permissions and try text input.",
        "voice_recognition_unsupported": "Speech recognition is unavailable in this browser or connection. Use the text input below. Microphone recognition may require a supported browser, HTTPS/localhost, and an internet service.",
        "voice_synthesis_unsupported": "Speech output is unavailable in this browser. The written answer is still available.",
        "voice_speaking": "Speaking answer.",
        "voice_speech_complete": "Speech finished.",
        "voice_synthesis_error": "Speech output failed. Read the written answer instead.",
        "voice_no_answer": "Ask a question to create an answer first.",
        "voice_privacy": "Voice recognition uses your browser's speech service; audio/transcripts may be processed by that provider. Text fallback works without microphone access. Speech output depends on installed browser/device voices.",
        "voice_input": "Ask about your saved farm records or general crop-care guidance",
        "voice_send": "Get answer",
        "voice_language": "Voice language",
        "voice_answer": "Answer",
        "voice_no_records": "I can only answer from the app's saved plot data and the crop-care guidance shown here. No matching saved information was found.",
        "voice_general": "General guidance: inspect the crop regularly, check soil conditions before irrigation, and consult a qualified local agricultural professional for crop-specific decisions.",
        "settings_title": "Settings, privacy and data",
        "local_only": "Storage: local SQLite only. Cloud synchronization is not configured.",
        "sync_unavailable": "No backend is configured, so records are not queued or synchronized between devices.",
        "backup_path": "Database file",
        "delete_all_data": "Delete all local farm records",
        "delete_all_confirm": "I understand this permanently deletes plots, scans, tasks, soil tests and financial records.",
        "delete_all_button": "Permanently delete local records",
        "delete_all_done": "Local farm records deleted.",
        "source_unverified": "No locally reviewed agricultural reference sources are configured. Specific pesticide, fertilizer and calendar prescriptions remain unavailable.",
        "local_error": "Local data could not be read or saved. Check storage availability and permissions; technical details were written to the local log.",
        "farm_title": "My Farm and Plots",
        "no_plots": "No plots saved yet. Add a plot to start keeping farm records.",
        "select_plot": "Select a plot to edit",
        "add_plot": "Add plot",
        "save_plot": "Save plot",
        "update_plot": "Update plot",
        "delete_plot": "Delete this plot",
        "confirm_delete": "I confirm that I want to delete this plot.",
        "farm_name": "Farm name",
        "plot_name": "Plot name",
        "village": "Village",
        "district": "District",
        "state": "State",
        "variety": "Tomato variety",
        "planting_date": "Planting or transplanting date",
        "area": "Plot area",
        "area_unit": "Area unit",
        "soil_type": "Soil type (optional)",
        "cultivation_type": "Cultivation",
        "irrigation_method": "Irrigation method",
        "crop_stage": "Current crop stage",
        "notes": "Notes (optional)",
        "stage_note": "Crop stage is manually selected. A date-based estimate needs a locally validated crop calendar.",
        "plot_saved": "Plot saved.",
        "plot_deleted": "Plot deleted.",
        "export_plots": "Download plot data as CSV",
        "plot_area_error": "Enter an area greater than zero.",
        "plot_input_error": "Enter farm and plot names and check that the plot details are valid.",
        "scan_plot": "Associate this scan with a plot (optional)",
        "unassigned_plot": "No plot selected",
        "scan_save_error": "Prediction is shown, but the scan could not be saved to history.",
        "history_title": "Scan and Farm History",
        "history_empty": "No scans have been saved yet.",
        "history_plot": "Farm / plot",
        "history_prediction": "AI prediction",
        "history_score": "Model score",
        "history_confirmation": "Farmer-confirmed diagnosis",
        "history_unconfirmed": "Unconfirmed",
        "history_select": "Select a scan",
        "history_no_confirmation": "Not confirmed",
        "history_save_confirmation": "Save farmer confirmation",
        "history_confirmation_saved": "Farmer confirmation saved.",
        "history_confirmation_note": "This is a farmer-entered confirmation, not an expert assessment. The AI prediction remains unchanged.",
        "history_export": "Download scan history as CSV",
        "history_delete": "Delete this scan",
        "history_confirm_delete": "I confirm that I want to delete this scan.",
        "language": "Language / மொழி",
        "upload": "Upload a tomato leaf photo",
        "camera": "Or take a photo using your camera",
        "upload_help": "Take a clear photo of one leaf in good natural light.",
        "analyze": "Check photo",
        "prediction": "AI Prediction",
        "confidence": "Model confidence",
        "top3": "Other possible predictions",
        "uncertain": "The result is uncertain. Take a clearer photo or ask an agricultural expert before treatment.",
        "symptoms": "Possible symptoms",
        "prevention": "Prevention and next steps",
        "treatment": "Treatment and Prevention",
        "confirm": "AI predictions are preliminary. Confirm the disease before applying treatment.",
        "healthy": "No disease-specific pesticide is indicated.",
        "virus": "No pesticide cures a virus inside the plant. Focus on prevention, sanitation and suitable management of insect vectors.",
        "no_data": "No treatment information is available for this disease yet.",
        "candidate": "These are treatment candidates only, not recommendations. Confirm the diagnosis and check that the current product label permits use on tomato.",
        "source_note": "This dataset has no source URLs or review dates. Verify all advice with a qualified local agricultural professional.",
        "english_detail_note": "The detailed guidance below is available in English only.",
        "natural_options": "Prevention and non-chemical guidance",
        "chemical_options": "Chemical treatment candidates",
        "active": "Active ingredient",
        "formulation": "Formulation",
        "target": "Target",
        "when": "When to consider treatment",
        "dose": "Dose",
        "interval": "Spray interval",
        "harvest": "Harvest waiting period",
        "not_verified": "Not verified",
        "weather_location": "Select your farm location",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "rain": "Rain probability",
        "wind": "Wind speed",
        "weather_error": "Weather is temporarily unavailable. Check your internet connection and try again.",
        "current_weather": "Current weather",
        "forecast": "Seven-day forecast",
        "max_temp": "Maximum",
        "min_temp": "Minimum",
        "weather_tip": "Check local conditions before irrigation or spraying. Rain and wind can affect treatment timing.",
        "weather_note": "Weather is forecast data, not a direct measurement from your farm.",
        "care_title": "Simple Tomato Crop Care",
        "care_items": [
            "Inspect leaves and stems regularly for new symptoms.",
            "Avoid unnecessary overhead watering when possible.",
            "Keep tools clean and remove diseased plant material appropriately.",
            "Check soil moisture before deciding to irrigate.",
            "Use fertilizer based on soil testing and suitable local advice.",
            "Follow pesticide labels, safety precautions and harvest restrictions."
        ],
        "bad_image": "Could not read that image. Please upload a JPG or PNG photo.",
        "large_image": "The image is too large. Use an image under 10 MB and 25 megapixels.",
        "prediction_error": "The image could not be analyzed. Please try again or check the model files.",
        "missing_model": "Model file not found. Check your project folder.",
        "missing_labels": "Class names file not found. Check tomato_class_names.json.",
        "footer": "AI results are preliminary. Confirm disease and treatment with a qualified agricultural professional.",
        "recheck": "Please analyze the photo again after changing the language.",
    },
    "தமிழ்": {
        "title": "ஸ்மார்ட் தக்காளி விவசாய உதவியாளர்",
        "subtitle": "தக்காளி விவசாயத்திற்கு உங்கள் AI நண்பர்",
        "menu": "பட்டியல்",
        "home": "முகப்பு",
        "disease": "செடியைப் பாருங்கள்",
        "farm": "என் பண்ணை",
        "history": "பரிசோதனை வரலாறு",
        "irrigation": "நீர்ப்பாசன ஆலோசனை",
        "calendar": "பயிர் காலண்டர்",
        "soil": "மண் மற்றும் உரம்",
        "expenses": "பண்ணை செலவுகள்",
        "reminders": "நினைவூட்டல்கள்",
        "voice": "குரல் மற்றும் உதவி",
        "settings": "அமைப்புகள்",
        "weather": "வானிலை முன்னறிவிப்பு",
        "care": "பயிர் பராமரிப்பு",
        "more": "மேலும்",
        "welcome": "FarmAI-க்கு வரவேற்கிறோம்",
        "welcome_subtitle": "உங்கள் எளிய விவசாய உதவியாளர்.",
        "home_check_plant": "என் செடியைப் பாருங்கள்",
        "home_weather": "இன்றைய வானிலை",
        "home_water": "பயிருக்கு நீர்",
        "home_tasks": "என் பண்ணைப் பணிகள்",
        "home_weather_title": "இன்றைய வானிலை",
        "home_tasks_title": "அடுத்த பணிகள்",
        "home_no_tasks": "பணிகள் எதுவும் இல்லை. தேவைப்பட்டால் ஒன்றைச் சேர்க்கவும்.",
        "view_weather": "இன்றைய வானிலையைப் பார்க்கவும்",
        "checking_connection": "இன்றைய வானிலை பெறப்படுகிறது…",
        "weather_unavailable_simple": "வானிலைத் தகவல் இல்லை. இணைய இணைப்பைச் சரிபார்க்கவும் அல்லது பிறகு முயற்சிக்கவும்.",
        "go_home": "முகப்பு",
        "task_today_section": "இன்று",
        "task_upcoming_section": "அடுத்து",
        "task_completed_section": "முடிந்தவை",
        "task_mark_done": "முடிந்தது",
        "task_done_feedback": "பணி முடிந்ததாகக் குறிக்கப்பட்டது.",
        "confirm_delete_task": "இந்தப் பணியை நீக்க வேண்டுமா?",
        "task_section_empty": "இந்தப் பகுதியில் பணிகள் இல்லை.",
        "task_instruction": "செய்ய வேண்டியது",
        "task_add_simple": "பணியைச் சேர்க்கவும்",
        "task_select_action": "ஒரு செயலைத் தேர்ந்தெடுக்கவும்",
        "plant_step_one": "படி 1 — செடியைப் படம் எடுக்கவும்",
        "plant_step_two": "படி 2 — படத்தைப் பாருங்கள்",
        "plant_step_three": "படி 3 — முடிவைப் பாருங்கள்",
        "choose_photo": "படத்தைத் தேர்வு செய்க",
        "take_photo": "படம் எடுக்கவும்",
        "checking_plant": "செடியைப் பரிசோதிக்கிறது…",
        "what_it_means": "இதன் பொருள் என்ன?",
        "what_to_do": "நான் என்ன செய்ய வேண்டும்?",
        "prediction_not_diagnosis": "செயற்கை நுண்ணறிவு முடிவு உறுதியான நோயறிதல் அல்ல.",
        "healthy_result": "✅ படம் ஆரோக்கியமான இலை போலத் தெரிகிறது.",
        "possible_problem": "⚠️ படத்தில் செடிப் பிரச்சினை இருக்கக்கூடும்.",
        "result_uncertain_simple": "❔ முடிவு தெளிவாக இல்லை. தெளிவான படம் எடுக்கவும் அல்லது உள்ளூர் வேளாண் நிபுணரிடம் கேட்கவும்.",
        "verified_source_unavailable": "உள்ளூர் வேளாண் சரிபார்ப்பு தேவை. குறிப்பிட்ட சிகிச்சைக்கு சரிபார்க்கப்பட்ட உள்ளூர் ஆதாரம் இல்லை.",
        "no_result_yet": "படத்தைத் தேர்ந்தெடுத்து அல்லது எடுத்து, படத்தைப் பாருங்கள் என்பதைத் தேர்ந்தெடுக்கவும்.",
        "weather_today_title": "இன்றைய வானிலை",
        "weather_updated": "முன்னறிவிப்பு நேரம்",
        "rain_expected": "மழை எதிர்பார்க்கப்படுகிறது",
        "no_rain_forecast": "மழை முன்னறிவிப்பு இல்லை",
        "rain_possible": "மழை பெய்ய வாய்ப்பு உள்ளது",
        "humidity_short": "ஈரப்பதம்",
        "temperature_short": "வெப்பநிலை",
        "today_forecast": "இன்று",
        "more_title": "மேலும் கருவிகள்",
        "more_description": "மற்றொரு பண்ணை வசதியைத் தேர்ந்தெடுக்கவும்.",
        "local_verification_short": "உள்ளூர் வேளாண் சரிபார்ப்பு தேவை.",
        "irrigation_title": "நீர்ப்பாசனத் திட்டம்",
        "irrigation_safety": "இந்தத் திட்டம் நீர் அளவையோ நிரந்தர நீர்ப்பாசன அட்டவணையையோ பரிந்துரைக்காது. மண் ஈரப்பத உணரியோ உங்கள் வயல் மழை அளவீடோ இதில் இல்லை.",
        "irrigation_missing": "இந்த ஆலோசனையைப் பயன்படுத்த என் பண்ணையில் ஒரு நிலத்தைச் சேர்க்கவும்.",
        "irrigation_reasoning": "பயன்படுத்திய விவரங்களும் காரணங்களும்",
        "irrigation_action": "வேர் பகுதியில் மண் நிலையைச் சரிபார்த்து, வயல் பார்வை மற்றும் உள்ளூர் ஆலோசனை அடிப்படையில் திட்டத்தைத் தேர்ந்தெடுக்கவும்.",
        "irrigation_uncertainty": "நிச்சயமின்மை அதிகம்: உள்ளூரில் சரிபார்க்கப்பட்ட அட்டவணை, நேரடி மண் ஈரப்பத அளவீடு அல்லது வயல் மழை அளவீடு இல்லை.",
        "missing_value": "வழங்கப்படவில்லை",
        "forecast_probability": "முன்னறிவிப்பு மழை வாய்ப்பு",
        "last_irrigation": "கடைசியாகப் பதிவு செய்த நீர்ப்பாசனம்",
        "log_irrigation": "இன்றைய நீர்ப்பாசனத்தைப் பதிவு செய்க",
        "irrigation_notes": "நீர்ப்பாசனக் குறிப்பு (விருப்பம்)",
        "irrigation_saved": "நீர்ப்பாசனப் பதிவு சேமிக்கப்பட்டது.",
        "irrigation_no_log": "நீர்ப்பாசனம் இன்னும் பதிவு செய்யப்படவில்லை",
        "forecast_location": "வானிலை முன்னறிவிப்பு இடம்",
        "forecast_status": "வானிலை முன்னறிவிப்பு கிடைக்கிறது",
        "forecast_unavailable": "வானிலை முன்னறிவிப்பு இல்லை; உள்ளூர் திட்டத்தைத் திருத்திச் சேமிக்கலாம்.",
        "forecast_not_loaded": "முன்னறிவிப்பு பெறப்படவில்லை. உள்ளூர் விவரங்களைக் கொண்டு திட்டத்தைப் பயன்படுத்தலாம்.",
        "forecast_internet": "முன்னறிவிப்பிற்கு இணைய இணைப்புடன் Open-Meteo சேவை தேவை; இது வயல் அளவீடு அல்ல.",
        "fetch_forecast": "முன்னறிவிப்பைப் பெறுக",
        "crop_name": "பயிர்",
        "schedule_irrigation": "நீர்ப்பாசனச் சரிபார்ப்பைத் திட்டமிடுக",
        "calendar_title": "பயிர் காலண்டர் மற்றும் பணிகள்",
        "task_title": "பணியின் பெயர்",
        "task_category": "பணி வகை",
        "task_due_date": "செய்ய வேண்டிய தேதி",
        "task_notes": "பணி குறிப்பு",
        "task_plot": "தொடர்புடைய நிலம் (விருப்பம்)",
        "task_add": "பணியைச் சேமிக்கவும்",
        "task_update": "பணியைப் புதுப்பிக்கவும்",
        "task_status": "பணி நிலை",
        "task_pending": "நிலுவை",
        "task_completed": "முடிந்தது",
        "task_skipped": "தவிர்க்கப்பட்டது",
        "task_edit": "திருத்த வேண்டிய பணி",
        "task_empty": "பணிகளோ நினைவூட்டல்களோ இல்லை. விவசாயி தேர்ந்தெடுக்கும் தேதியைச் சேர்க்கவும்.",
        "task_saved": "பணி சேமிக்கப்பட்டது.",
        "task_deleted": "பணி நீக்கப்பட்டது.",
        "task_delete": "பணியை நீக்குக",
        "task_due": "செய்ய வேண்டியவை",
        "task_overdue": "காலம் கடந்தவை",
        "task_scheduled": "திட்டமிட்ட பணிகள்",
        "task_duplicate": "இதுபோன்ற பணி ஏற்கனவே உள்ளது; மீண்டும் சேர்க்கப்படவில்லை.",
        "calendar_no_schedule": "உள்ளூரில் சரிபார்க்கப்பட்ட பயிர் காலண்டர் அமைக்கப்படவில்லை. தேதிகளை நீங்களே சேர்க்கவும்; பயிர் நிலை அல்லது மருந்து அட்டவணையை ஆப் உருவாக்காது.",
        "soil_title": "மண் பரிசோதனை மற்றும் ஊட்டச்சத்து பதிவுகள்",
        "soil_education": "மண் அறிக்கை ஆய்வக அளவீடுகளைப் பதிவு செய்கிறது. விளக்கம் அறிக்கை முறை, பயிர், பகுதி மற்றும் உள்ளூர் ஆலோசனையைப் பொறுத்தது. இந்த ஆப் குறைபாட்டைக் கண்டறியவோ உர அளவைக் கணக்கிடவோாது.",
        "soil_plot": "நிலம் (விருப்பம்)",
        "soil_test_date": "மண் பரிசோதனை தேதி",
        "soil_ph": "pH (விருப்பம்)",
        "soil_n": "நைட்ரஜன் முடிவு (விருப்பம்)",
        "soil_p": "பாஸ்பரஸ் முடிவு (விருப்பம்)",
        "soil_k": "பொட்டாசியம் முடிவு (விருப்பம்)",
        "soil_organic": "கரிமப் பொருள் முடிவு (விருப்பம்)",
        "soil_notes": "மற்ற மதிப்புகள் அல்லது குறிப்புகள்",
        "soil_save": "மண் பரிசோதனைப் பதிவைச் சேமிக்கவும்",
        "soil_saved": "மண் பரிசோதனைப் பதிவு சேமிக்கப்பட்டது.",
        "soil_empty": "மண் பரிசோதனை முடிவுகள் இன்னும் பதிவு செய்யப்படவில்லை.",
        "soil_no_rates": "உள்ளூரில் சரிபார்க்கப்பட்ட உர அளவு ஆதாரம் அமைக்கப்படவில்லை. உங்கள் அறிக்கையை உள்ளூர் வேளாண் நிபுணரிடம் விளக்கிக் கொள்ளவும்.",
        "expenses_title": "பண்ணை செலவுகள் மற்றும் விற்பனை",
        "record_type": "பதிவு வகை",
        "expense": "செலவு",
        "sale": "அறுவடை விற்பனை",
        "expense_category": "செலவு வகை",
        "financial_description": "விவரம்",
        "transaction_date": "பதிவு தேதி",
        "amount_rupees": "தொகை (ரூபாய்)",
        "quantity_kg": "விற்ற அளவு (கிலோ)",
        "price_per_kg": "விற்பனை விலை (ரூபாய்/கிலோ)",
        "financial_plot": "நிலம் (விருப்பம்)",
        "financial_save": "நிதிப் பதிவைச் சேமிக்கவும்",
        "financial_saved": "நிதிப் பதிவு சேமிக்கப்பட்டது.",
        "financial_from": "தொடக்கம்",
        "financial_to": "முடிவு",
        "financial_expenses": "பதிவு செய்யப்பட்ட செலவுகள்",
        "financial_income": "பதிவு செய்யப்பட்ட விற்பனை வருமானம்",
        "financial_profit": "பதிவு செய்யப்பட்ட நிகரத் தொகை",
        "financial_empty": "இந்த தேதி வரம்பில் நிதிப் பதிவுகள் இல்லை.",
        "financial_actual_note": "இந்தத் தொகைகள் பதிவு செய்யப்பட்ட செலவு மற்றும் விற்பனை மட்டுமே; இவை கணிப்போ உறுதியான லாபமோ அல்ல.",
        "reminders_title": "நினைவூட்டல்கள் மற்றும் செய்ய வேண்டிய பணிகள்",
        "reminders_app_only": "நினைவூட்டல்கள் இந்தச் செயலியில் மட்டுமே தெரியும். செயலி மூடப்பட்டிருக்கும்போது அறிவிப்பு வராது.",
        "voice_title": "குரல் உதவியாளர்",
        "voice_unavailable": "கிடைக்கும் போது உலாவி குரல் வசதிகளைப் பயன்படுத்தும். தொடங்கு என்பதை அழுத்திய பின்னரே மைக்ரோஃபோன் அனுமதி கேட்கப்படும்; குரல் பதிவு உலாவி பேச்சு சேவைக்கு அனுப்பப்படலாம், இணையம் தேவைப்படலாம்.",
        "voice_start": "மைக்ரோஃபோனைத் தொடங்கு",
        "voice_stop": "கேட்பதை / பேச்சை நிறுத்து",
        "voice_speak": "பதிலைப் பேசுக",
        "voice_ready": "தயார். தொடங்கிய பின்னர் உலாவி மைக்ரோஃபோன் அனுமதி கேட்கலாம்.",
        "voice_listening": "கேட்கிறது… இப்போது பேசுங்கள்.",
        "voice_transcript_received": "குரல் பதிவு கிடைத்தது. கீழே உள்ள உரையைப் பார்த்து திருத்தவும்.",
        "voice_permission_denied": "மைக்ரோஃபோன் அனுமதி மறுக்கப்பட்டது. உலாவி தள அமைப்புகளில் அனுமதிக்கவும் அல்லது கேள்வியைத் தட்டச்சு செய்யவும்.",
        "voice_no_microphone": "பயன்படுத்தக்கூடிய மைக்ரோஃபோன் இல்லை. ஒன்றை இணைக்கவும் அல்லது கேள்வியைத் தட்டச்சு செய்யவும்.",
        "voice_no_speech": "குரல் கண்டறியப்படவில்லை. மீண்டும் முயற்சிக்கவும் அல்லது தட்டச்சு செய்யவும்.",
        "voice_network_error": "உலாவி பேச்சு சேவையில் இணையப் பிழை. இணைப்பைச் சரிபார்க்கவும் அல்லது உரையைப் பயன்படுத்தவும்.",
        "voice_stopped": "குரல் செயல்பாடு நிறுத்தப்பட்டது.",
        "voice_recognition_error": "குரல் அடையாளத்தைத் தொடங்க முடியவில்லை. உலாவி அனுமதியைச் சரிபார்த்து உரையைப் பயன்படுத்தவும்.",
        "voice_recognition_unsupported": "இந்த உலாவி அல்லது இணைப்பில் குரல் அடையாளம் கிடைக்கவில்லை. கீழே உரையைப் பயன்படுத்தவும். ஆதரவு உலாவி, HTTPS/localhost மற்றும் இணையச் சேவை தேவைப்படலாம்.",
        "voice_synthesis_unsupported": "இந்த உலாவியில் குரல் வெளியீடு கிடைக்கவில்லை. எழுத்துப் பதிலைப் படிக்கலாம்.",
        "voice_speaking": "பதில் பேசப்படுகிறது.",
        "voice_speech_complete": "பேச்சு முடிந்தது.",
        "voice_synthesis_error": "குரல் வெளியீடு தோல்வியடைந்தது. எழுத்துப் பதிலைப் படிக்கவும்.",
        "voice_no_answer": "முதலில் கேள்வி கேட்டு பதிலை உருவாக்கவும்.",
        "voice_privacy": "குரல் அடையாளம் உலாவியின் பேச்சு சேவையைப் பயன்படுத்துகிறது; ஒலி/உரை அந்த வழங்குநரால் செயலாக்கப்படலாம். மைக்ரோஃபோன் இல்லாமலும் உரை உள்ளீடு கிடைக்கும். பேச்சு வெளியீடு சாதனத்தில் உள்ள உலாவி குரல்களைப் பொறுத்தது.",
        "voice_input": "சேமித்த பண்ணை விவரம் அல்லது பொதுப் பயிர் பராமரிப்பு பற்றி கேளுங்கள்",
        "voice_send": "பதில் பெறுக",
        "voice_language": "குரல் மொழி",
        "voice_answer": "பதில்",
        "voice_no_records": "செயலியில் சேமித்த நில விவரங்களையும் இங்கு காட்டப்படும் பயிர் பராமரிப்பு வழிகாட்டுதலையும் மட்டுமே பயன்படுத்த முடியும். பொருந்தும் பதிவு இல்லை.",
        "voice_general": "பொது வழிகாட்டுதல்: பயிரைத் தொடர்ந்து பாருங்கள், நீர்ப்பாசனத்திற்கு முன் மண் நிலையைச் சரிபாருங்கள், பயிர் சார்ந்த முடிவுகளுக்கு உள்ளூர் வேளாண் நிபுணரை அணுகுங்கள்.",
        "settings_title": "அமைப்புகள், தனியுரிமை மற்றும் தரவு",
        "local_only": "சேமிப்பு: உள்ளூர் SQLite மட்டும். மேக ஒத்திசைவு அமைக்கப்படவில்லை.",
        "sync_unavailable": "சேவையகம் இல்லாததால் பதிவுகள் ஒத்திசைவுக்காகக் காத்திருக்கவோ சாதனங்களுக்கு இடையே பகிரப்படவோ இல்லை.",
        "backup_path": "தரவுத்தளக் கோப்பு",
        "delete_all_data": "உள்ளூர் பண்ணைப் பதிவுகளை நீக்குக",
        "delete_all_confirm": "நிலங்கள், பரிசோதனைகள், பணிகள், மண் மற்றும் நிதிப் பதிவுகள் நிரந்தரமாக நீக்கப்படும் என்பதைப் புரிந்துகொண்டேன்.",
        "delete_all_button": "உள்ளூர் பதிவுகளை நிரந்தரமாக நீக்குக",
        "delete_all_done": "உள்ளூர் பண்ணைப் பதிவுகள் நீக்கப்பட்டன.",
        "source_unverified": "உள்ளூரில் மதிப்பாய்வு செய்த வேளாண் ஆதாரங்கள் எதுவும் அமைக்கப்படவில்லை. பூச்சிக்கொல்லி, உரம் மற்றும் பயிர் காலண்டர் குறிப்பிட்ட பரிந்துரைகள் வழங்கப்படாது.",
        "local_error": "உள்ளூர் தரவைப் படிக்கவோ சேமிக்கவோ முடியவில்லை. சேமிப்பிடம் மற்றும் அனுமதிகளைச் சரிபார்க்கவும்; தொழில்நுட்ப விவரங்கள் உள்ளூர் பதிவில் எழுதப்பட்டன.",
        "farm_title": "என் பண்ணை மற்றும் நிலங்கள்",
        "no_plots": "இன்னும் நில விவரங்கள் சேமிக்கப்படவில்லை. பதிவுகளைத் தொடங்க ஒரு நிலத்தைச் சேர்க்கவும்.",
        "select_plot": "திருத்த வேண்டிய நிலத்தைத் தேர்ந்தெடுக்கவும்",
        "add_plot": "நிலத்தைச் சேர்க்கவும்",
        "save_plot": "நிலத்தைச் சேமிக்கவும்",
        "update_plot": "நில விவரத்தைப் புதுப்பிக்கவும்",
        "delete_plot": "இந்த நிலத்தை நீக்கவும்",
        "confirm_delete": "இந்த நிலத்தை நீக்க விரும்புகிறேன் என்பதை உறுதிப்படுத்துகிறேன்.",
        "farm_name": "பண்ணையின் பெயர்",
        "plot_name": "நிலத்தின் பெயர்",
        "village": "கிராமம்",
        "district": "மாவட்டம்",
        "state": "மாநிலம்",
        "variety": "தக்காளி ரகம்",
        "planting_date": "நடவு அல்லது மாற்று நடவு தேதி",
        "area": "நிலத்தின் பரப்பளவு",
        "area_unit": "பரப்பளவு அலகு",
        "soil_type": "மண் வகை (விருப்பம்)",
        "cultivation_type": "சாகுபடி முறை",
        "irrigation_method": "நீர்ப்பாசன முறை",
        "crop_stage": "தற்போதைய பயிர் நிலை",
        "notes": "குறிப்புகள் (விருப்பம்)",
        "stage_note": "பயிர் நிலையை விவசாயி தேர்ந்தெடுக்கிறார். தேதிப்படி கணிக்க உள்ளூரில் சரிபார்க்கப்பட்ட பயிர் காலண்டர் தேவை.",
        "plot_saved": "நில விவரம் சேமிக்கப்பட்டது.",
        "plot_deleted": "நில விவரம் நீக்கப்பட்டது.",
        "export_plots": "நில விவரங்களை CSV ஆகப் பதிவிறக்கவும்",
        "plot_area_error": "பூஜ்ஜியத்தை விட அதிகமான பரப்பளவை உள்ளிடவும்.",
        "plot_input_error": "பண்ணை மற்றும் நிலத்தின் பெயர்களை உள்ளிட்டு, விவரங்கள் சரியாக உள்ளனவா எனச் சரிபார்க்கவும்.",
        "scan_plot": "இந்தப் பரிசோதனையை ஒரு நிலத்துடன் இணைக்கவும் (விருப்பம்)",
        "unassigned_plot": "நிலம் தேர்ந்தெடுக்கப்படவில்லை",
        "scan_save_error": "கணிப்பு காட்டப்படுகிறது; ஆனால் வரலாற்றில் சேமிக்க முடியவில்லை.",
        "history_title": "பரிசோதனை மற்றும் பண்ணை வரலாறு",
        "history_empty": "இதுவரை எந்தப் பரிசோதனையும் சேமிக்கப்படவில்லை.",
        "history_plot": "பண்ணை / நிலம்",
        "history_prediction": "AI கணிப்பு",
        "history_score": "மாதிரி மதிப்பெண்",
        "history_confirmation": "விவசாயி உறுதிப்படுத்திய நோய்",
        "history_unconfirmed": "உறுதிப்படுத்தப்படவில்லை",
        "history_select": "பரிசோதனையைத் தேர்ந்தெடுக்கவும்",
        "history_no_confirmation": "உறுதிப்படுத்தப்படவில்லை",
        "history_save_confirmation": "விவசாயி உறுதிப்படுத்தலைச் சேமிக்கவும்",
        "history_confirmation_saved": "விவசாயி உறுதிப்படுத்தல் சேமிக்கப்பட்டது.",
        "history_confirmation_note": "இது விவசாயி உள்ளிட்ட உறுதிப்படுத்தல்; நிபுணர் மதிப்பீடு அல்ல. AI கணிப்பு மாற்றப்படாது.",
        "history_export": "பரிசோதனை வரலாற்றை CSV ஆகப் பதிவிறக்கவும்",
        "history_delete": "இந்தப் பரிசோதனையை நீக்கவும்",
        "history_confirm_delete": "இந்தப் பரிசோதனையை நீக்க விரும்புகிறேன் என்பதை உறுதிப்படுத்துகிறேன்.",
        "language": "மொழி / Language",
        "upload": "தக்காளி இலைப் படத்தைப் பதிவேற்றவும்",
        "camera": "அல்லது கேமராவில் படம் எடுக்கவும்",
        "upload_help": "நல்ல இயற்கை வெளிச்சத்தில் ஒரு இலையின் தெளிவான படத்தை எடுக்கவும்.",
        "analyze": "படத்தைப் பாருங்கள்",
        "prediction": "AI கணிப்பு",
        "confidence": "மாதிரியின் நம்பிக்கை அளவு",
        "top3": "மற்ற சாத்தியமான கணிப்புகள்",
        "uncertain": "கணிப்பு உறுதியானதாக இல்லை. தெளிவான படம் எடுக்கவும் அல்லது சிகிச்சைக்கு முன் வேளாண் நிபுணரை அணுகவும்.",
        "symptoms": "காணப்படக்கூடிய அறிகுறிகள்",
        "prevention": "தடுப்பு மற்றும் அடுத்த நடவடிக்கைகள்",
        "treatment": "சிகிச்சை மற்றும் தடுப்பு",
        "confirm": "AI கணிப்பு ஆரம்ப மதிப்பீடு மட்டுமே. சிகிச்சைக்கு முன் நோயை உறுதிப்படுத்தவும்.",
        "healthy": "குறிப்பிட்ட நோய்க்கான பூச்சிக்கொல்லி தேவையில்லை.",
        "virus": "செடிக்குள் உள்ள வைரஸை எந்தப் பூச்சிக்கொல்லியும் குணப்படுத்தாது. தடுப்பு, சுத்தம் மற்றும் நோயைப் பரப்பும் பூச்சிகளை நிர்வகிப்பதில் கவனம் செலுத்தவும்.",
        "no_data": "இந்த நோய்க்கான சிகிச்சைத் தகவல் இன்னும் சேர்க்கப்படவில்லை.",
        "candidate": "இவை சிகிச்சைக்கான சாத்தியமான விருப்பங்கள் மட்டுமே; பரிந்துரைகள் அல்ல. நோயை உறுதிப்படுத்தி, தற்போதைய தயாரிப்பு லேபிளில் தக்காளிக்கான அனுமதி உள்ளதா எனச் சரிபார்க்கவும்.",
        "source_note": "இந்தத் தரவுத்தளத்தில் ஆதார இணைப்புகளும் மதிப்பாய்வு தேதிகளும் இல்லை. உள்ளூர் வேளாண் நிபுணரிடம் ஆலோசனை பெறவும்.",
        "english_detail_note": "கீழே உள்ள விரிவான வழிகாட்டுதல் ஆங்கிலத்தில் மட்டுமே உள்ளது.",
        "natural_options": "தடுப்பு மற்றும் ரசாயனமற்ற வழிகாட்டுதல்",
        "chemical_options": "ரசாயன சிகிச்சைக்கான சாத்தியமான விருப்பங்கள்",
        "active": "செயல்படும் மூலப்பொருள்",
        "formulation": "தயாரிப்பு வகை",
        "target": "கட்டுப்படுத்த வேண்டிய நோய் அல்லது பூச்சி",
        "when": "எப்போது சிகிச்சையைப் பரிசீலிக்கலாம்",
        "dose": "அளவு",
        "interval": "தெளிப்பு இடைவெளி",
        "harvest": "அறுவடைக்கு முன் காத்திருக்கும் காலம்",
        "not_verified": "சரிபார்க்கப்படவில்லை",
        "weather_location": "உங்கள் பண்ணை அமைவிடத்தைத் தேர்ந்தெடுக்கவும்",
        "temperature": "வெப்பநிலை",
        "humidity": "காற்றின் ஈரப்பதம்",
        "rain": "மழை வாய்ப்பு",
        "wind": "காற்றின் வேகம்",
        "weather_error": "வானிலைத் தகவல் இப்போது கிடைக்கவில்லை. இணைய இணைப்பைச் சரிபார்த்து மீண்டும் முயற்சிக்கவும்.",
        "current_weather": "தற்போதைய வானிலை",
        "forecast": "ஏழு நாள் முன்னறிவிப்பு",
        "max_temp": "அதிகபட்சம்",
        "min_temp": "குறைந்தபட்சம்",
        "weather_tip": "நீர் பாய்ச்சுவதற்கு அல்லது மருந்து தெளிப்பதற்கு முன் உள்ளூர் வானிலையைச் சரிபார்க்கவும். மழையும் காற்றும் சிகிச்சை நேரத்தைப் பாதிக்கலாம்.",
        "weather_note": "இது வானிலை முன்னறிவிப்பு; உங்கள் பண்ணையில் நேரடியாக அளந்த தகவல் அல்ல.",
        "care_title": "எளிய தக்காளிப் பயிர் பராமரிப்பு",
        "care_items": [
            "புதிய அறிகுறிகள் உள்ளனவா என இலைகளையும் தண்டுகளையும் தொடர்ந்து பாருங்கள்.",
            "முடிந்தவரை தேவையற்ற மேலிருந்து நீர் பாய்ச்சுவதைத் தவிர்க்கவும்.",
            "கருவிகளைச் சுத்தமாக வைத்துக் கொண்டு பாதிக்கப்பட்ட தாவரப் பகுதிகளை முறையாக அகற்றவும்.",
            "நீர் பாய்ச்சுவதற்கு முன் மண்ணின் ஈரப்பதத்தைச் சரிபார்க்கவும்.",
            "மண் பரிசோதனை மற்றும் உள்ளூர் வேளாண் ஆலோசனைப்படி உரமிடவும்.",
            "பூச்சிக்கொல்லி லேபிள், பாதுகாப்பு விதிகள் மற்றும் அறுவடை இடைவெளியைப் பின்பற்றவும்."
        ],
        "bad_image": "படத்தைத் திறக்க முடியவில்லை. JPG அல்லது PNG படத்தைப் பதிவேற்றவும்.",
        "large_image": "படம் மிகவும் பெரியது. 10 MB மற்றும் 25 மெகாபிக்சலுக்கு குறைவான படத்தைப் பயன்படுத்தவும்.",
        "prediction_error": "படத்தைப் பரிசோதிக்க முடியவில்லை. மீண்டும் முயற்சிக்கவும் அல்லது மாதிரிக் கோப்புகளைச் சரிபார்க்கவும்.",
        "missing_model": "AI மாதிரிக் கோப்பு இல்லை. திட்டக் கோப்புறையைச் சரிபார்க்கவும்.",
        "missing_labels": "நோய்களின் பெயர் கோப்பு இல்லை. tomato_class_names.json கோப்பைச் சரிபார்க்கவும்.",
        "footer": "AI கணிப்பு ஆரம்ப மதிப்பீடு மட்டுமே. நோய் மற்றும் சிகிச்சையை வேளாண் நிபுணரிடம் உறுதிப்படுத்தவும்.",
        "recheck": "மொழியை மாற்றியுள்ளீர்கள். படத்தை மீண்டும் பரிசோதிக்கவும்.",
    },
}


# =====================================================
# 3. DISEASE INFORMATION
# Keys are normalized to match the model's class names.
# =====================================================

DISEASES = {
    "Bacterial_spot": {
        "en": "Bacterial Spot",
        "ta": "பாக்டீரியா இலைப்புள்ளி நோய்",
        "description_en": "A bacterial disease that can cause dark spots on leaves and fruit.",
        "description_ta": "இலைகளிலும் பழங்களிலும் கருமையான புள்ளிகளை ஏற்படுத்தக்கூடிய பாக்டீரியா நோய்.",
        "prevention_en": "Clean tools, avoid handling wet plants unnecessarily, and confirm the disease before treatment.",
        "prevention_ta": "கருவிகளைச் சுத்தமாக வைத்துக் கொள்ளவும். ஈரமான செடிகளைத் தேவையின்றிக் கையாளாதீர்கள். சிகிச்சைக்கு முன் நோயை உறுதிப்படுத்தவும்.",
    },
    "Early_blight": {
        "en": "Early Blight",
        "ta": "ஆரம்ப இலைக்கருகல் நோய்",
        "description_en": "Brown leaf spots may develop, sometimes with ring-like patterns and yellowing around them.",
        "description_ta": "இலைகளில் பழுப்பு நிறப் புள்ளிகள், வளைய வடிவ அடையாளங்கள் மற்றும் மஞ்சளாதல் ஏற்படலாம்.",
        "prevention_en": "Remove badly affected leaves appropriately, avoid prolonged leaf wetness and confirm the disease.",
        "prevention_ta": "கடுமையாகப் பாதிக்கப்பட்ட இலைகளை முறையாக அகற்றவும். இலைகள் நீண்ட நேரம் ஈரமாக இருப்பதைத் தவிர்த்து நோயை உறுதிப்படுத்தவும்.",
    },
    "Late_blight": {
        "en": "Late Blight",
        "ta": "தாமத இலைக்கருகல் நோய்",
        "description_en": "May cause water-soaked dark lesions and can spread rapidly in suitable conditions.",
        "description_ta": "நீர் நனைந்தது போன்ற கருமையான புள்ளிகள் தோன்றலாம். ஏற்ற சூழலில் வேகமாகப் பரவக்கூடும்.",
        "prevention_en": "Seek prompt expert confirmation and avoid moving potentially infected plant material.",
        "prevention_ta": "உடனடியாக வேளாண் நிபுணரிடம் உறுதிப்படுத்தவும். பாதிக்கப்பட்டிருக்கக்கூடிய தாவரப் பகுதிகளைப் பரப்பாதீர்கள்.",
    },
    "Leaf_Mold": {
        "en": "Leaf Mold",
        "ta": "இலைப் பூஞ்சை நோய்",
        "description_en": "Pale or yellow areas may appear on upper leaf surfaces, with mold underneath.",
        "description_ta": "இலையின் மேற்புறத்தில் வெளிர் அல்லது மஞ்சள் பகுதிகளும் கீழ்ப்புறத்தில் பூஞ்சை வளர்ச்சியும் காணப்படலாம்.",
        "prevention_en": "Improve airflow where possible and avoid excessive leaf wetness.",
        "prevention_ta": "முடிந்தவரை காற்றோட்டத்தை மேம்படுத்தி இலைகள் அதிக நேரம் ஈரமாக இருப்பதைத் தவிர்க்கவும்.",
    },
    "Septoria_leaf_spot": {
        "en": "Septoria Leaf Spot",
        "ta": "செப்டோரியா இலைப்புள்ளி நோய்",
        "description_en": "Many small spots with pale centers and darker edges may appear.",
        "description_ta": "வெளிர் நடுப்பகுதியும் கருமையான விளிம்புகளும் கொண்ட பல சிறிய புள்ளிகள் தோன்றலாம்.",
        "prevention_en": "Remove affected plant debris and avoid splashing soil onto leaves.",
        "prevention_ta": "பாதிக்கப்பட்ட தாவரக் கழிவுகளை அகற்றி மண் இலைகளில் தெறிப்பதைத் தவிர்க்கவும்.",
    },
    "Spider_mites Two-spotted_spider_mite": {
        "en": "Two-Spotted Spider Mites",
        "ta": "இரு புள்ளி சிலந்திப் பூச்சிகள்",
        "description_en": "Mites may cause tiny pale speckles, leaf bronzing or fine webbing.",
        "description_ta": "சிறிய வெளிர் புள்ளிகள், இலை வெண்கல நிறமாக மாறுதல் அல்லது நுண்ணிய வலை போன்ற அமைப்பு காணப்படலாம்.",
        "prevention_en": "Inspect the undersides of leaves and confirm mites before selecting a control method.",
        "prevention_ta": "இலைகளின் அடிப்புறத்தை நன்றாகப் பார்த்து பூச்சிகள் இருப்பதை உறுதிப்படுத்திய பிறகே கட்டுப்பாட்டு முறையைத் தேர்ந்தெடுக்கவும்.",
    },
    "Target_Spot": {
        "en": "Target Spot",
        "ta": "இலக்கு வடிவ இலைப்புள்ளி நோய்",
        "description_en": "Circular brown lesions with concentric rings may appear.",
        "description_ta": "வளையங்களுடன் கூடிய வட்டமான பழுப்பு நிறப் புள்ளிகள் தோன்றலாம்.",
        "prevention_en": "Confirm the disease because several tomato problems can look similar.",
        "prevention_ta": "பல தக்காளி நோய்கள் ஒரே மாதிரி தோன்றலாம். எனவே நோயை உறுதிப்படுத்தவும்.",
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "en": "Tomato Yellow Leaf Curl Virus",
        "ta": "தக்காளி மஞ்சள் இலை சுருள் வைரஸ் நோய்",
        "description_en": "May cause yellowing, upward leaf curling and stunted growth.",
        "description_ta": "இலை மஞ்சளாதல், மேல்நோக்கிச் சுருளுதல் மற்றும் வளர்ச்சி குறைதல் ஏற்படலாம்.",
        "prevention_en": "Seek confirmation and follow local agricultural advice for whitefly and infected plant management.",
        "prevention_ta": "நோயை உறுதிப்படுத்தி வெள்ளை ஈக்கள் மற்றும் பாதிக்கப்பட்ட செடிகளை நிர்வகிக்க உள்ளூர் வேளாண் ஆலோசனையைப் பின்பற்றவும்.",
    },
    "Tomato_mosaic_virus": {
        "en": "Tomato Mosaic Virus",
        "ta": "தக்காளி மொசைக் வைரஸ் நோய்",
        "description_en": "May cause mottled leaf coloration, leaf distortion and reduced growth.",
        "description_ta": "இலைகளில் கலந்த நிற மாற்றம், வடிவச் சிதைவு மற்றும் வளர்ச்சி குறைதல் ஏற்படலாம்.",
        "prevention_en": "Clean tools and hands, avoid transferring plant sap between plants, and seek confirmation.",
        "prevention_ta": "கருவிகளையும் கைகளையும் சுத்தமாக வைத்துக் கொள்ளவும். செடிகளுக்கிடையே சாறு பரவுவதைத் தவிர்த்து நோயை உறுதிப்படுத்தவும்.",
    },
    "healthy": {
        "en": "Healthy Leaf",
        "ta": "ஆரோக்கியமான இலை",
        "description_en": "The model did not identify one of its known disease classes.",
        "description_ta": "மாதிரி அறிந்துள்ள நோய் வகைகளில் எதையும் கண்டறியவில்லை.",
        "prevention_en": "Continue regular inspection and good crop care.",
        "prevention_ta": "தொடர்ந்து பயிரைப் பரிசோதித்து நல்ல பராமரிப்பை மேற்கொள்ளவும்.",
    },
}


def readable_class_name(class_key, language):
    key = normalize_class_key(class_key)
    info = DISEASES.get(key)

    if info:
        return info["ta"] if language == "தமிழ்" else info["en"]

    # Friendly fallback for any class not translated yet.
    return key.replace("_", " ").replace("  ", " ")


# =====================================================
# 5. LOAD MODEL AND JSON FILES
# =====================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_labels():
    if not LABELS_PATH.exists():
        return None

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        labels = json.load(file)

    if isinstance(labels, dict):
        try:
            labels = [
                labels[str(i)] if str(i) in labels else labels[i]
                for i in range(len(labels))
            ]
        except (KeyError, TypeError):
            labels = list(labels.values())

    if not isinstance(labels, list):
        raise ValueError("Class names must be a JSON list or indexed dictionary.")

    return [str(label) for label in labels]


@st.cache_data
def load_treatment_database():
    if not PESTICIDE_DB_PATH.exists():
        return {}

    with open(PESTICIDE_DB_PATH, "r", encoding="utf-8") as file:
        database = json.load(file)

    return database if isinstance(database, dict) else {}


# =====================================================
# 6. IMAGE PREDICTION
# =====================================================

# =====================================================
# 7. DISEASE INFORMATION
# =====================================================

def display_disease_information(class_key, language, t):
    key = normalize_class_key(class_key)
    info = DISEASES.get(key)

    if not info:
        st.info(t["no_data"])
        return

    description_key = "description_ta" if language == "தமிழ்" else "description_en"
    prevention_key = "prevention_ta" if language == "தமிழ்" else "prevention_en"

    st.markdown("**" + t["symptoms"] + "**")
    st.write(info[description_key])

    st.markdown("**" + t["prevention"] + "**")
    st.write(info[prevention_key])


# =====================================================
# 8. TREATMENT DATABASE DISPLAY
# =====================================================

def display_treatment(class_key, database, language, t):
    key = normalize_class_key(class_key)

    st.subheader("🧪 " + t["treatment"])

    if key == "healthy":
        st.success(t["healthy"])
        return

    is_virus = key in {
        "Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato_mosaic_virus",
    }
    if is_virus:
        st.info(t["virus"])

    entry = find_treatment_entry(database, key)
    if not entry:
        st.warning(t["no_data"])
        return

    natural_options = get_treatment_options(entry, "natural_options")
    chemical_options = [] if is_virus else get_treatment_options(entry, "chemical_options")
    if not natural_options and not chemical_options:
        st.info(t["no_data"])
        return

    st.warning(t["confirm"])

    if natural_options:
        st.markdown("**" + t["natural_options"] + "**")
    for option in natural_options:
        option_name = option.get("ta_name" if language == "தமிழ்" else "name")
        if not option_name:
            option_name = option.get("name", "Option")
        with st.container(border=True):
            st.write(f"**{option_name}**")
            st.caption(str(option.get("type", "Preventive")))
            if language == "தமிழ்":
                st.caption(t["english_detail_note"])
            st.write(str(option.get("use", "")))

    database_warning = entry.get("warning")
    if database_warning:
        if language == "தமிழ்":
            st.caption(t["english_detail_note"])
        st.warning(str(database_warning))

    if chemical_options:
        st.markdown("**" + t["chemical_options"] + "**")
        st.warning(t["candidate"])
        st.caption(t["source_note"])
        for product in chemical_options:
            with st.container(border=True):
                st.write(f"**{t['active']}:** {product.get('name', 'Not specified')}")
                st.write(f"**{t['target']}:** {product.get('target', t['not_verified'])}")
                st.write(f"**{t['when']}:** {product.get('use', t['not_verified'])}")
                st.write(f"**{t['dose']}:** {product.get('dose') or t['not_verified']}")
                st.write(f"**{t['interval']}:** {product.get('interval') or t['not_verified']}")
                st.write(f"**{t['harvest']}:** {product.get('pre_harvest_interval') or t['not_verified']}")


# =====================================================
# 9. WEATHER
# =====================================================

LOCATIONS = {
    "Coimbatore": (11.0168, 76.9558),
    "Pollachi": (10.6582, 77.0089),
    "Chennai": (13.0827, 80.2707),
    "Madurai": (9.9252, 78.1198),
    "Erode": (11.3410, 77.7172),
    "Tiruppur": (11.1085, 77.3411),
    "Salem": (11.6643, 78.1460),
    "Thanjavur": (10.7870, 79.1378),
    "Tirunelveli": (8.7139, 77.7567),
}


@st.cache_data(ttl=1800)
def get_weather(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m"
        ),
        "daily": (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_probability_max"
        ),
        "forecast_days": 7,
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()

    data = response.json()

    if "current" not in data or "daily" not in data:
        raise ValueError("Weather data is incomplete.")

    return data


def display_weather(t):
    st.subheader("🌦️ " + t["weather_today_title"])

    place = st.selectbox(
        t["weather_location"],
        list(LOCATIONS.keys()),
        key="weather_page_location",
    )

    latitude, longitude = LOCATIONS[place]

    try:
        weather = get_weather(latitude, longitude)
        current = weather["current"]
        daily = weather["daily"]

        st.caption(f"{t['current_weather']} — {place}")
        st.caption(f"{t['weather_updated']}: {current.get('time', t['missing_value'])}")

        rain_values = daily.get("precipitation_probability_max", [])
        rain_today = rain_values[0] if rain_values else None

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "🌡️ " + t["temperature_short"],
            f"{current.get('temperature_2m', t['missing_value'])} °C"
            if current.get("temperature_2m") is not None else t["missing_value"],
        )
        col2.metric(
            "💧 " + t["humidity_short"],
            f"{current.get('relative_humidity_2m', t['missing_value'])}%"
            if current.get("relative_humidity_2m") is not None else t["missing_value"],
        )
        col3.metric(
            "🌧️ " + t["rain"],
            f"{rain_today}%" if rain_today is not None else t["missing_value"],
        )
        if rain_today is not None:
            rain_message = (
                t["no_rain_forecast"] if rain_today == 0
                else t["rain_expected"] if rain_today >= 50
                else t["rain_possible"]
            )
            st.info(rain_message)

        st.info(t["weather_tip"])
        st.caption(t["weather_note"])

        st.markdown("### 📅 " + t["forecast"])

        dates = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])
        rains = daily.get("precipitation_probability_max", [])

        for row_start in range(0, len(dates), 4):
            row_dates = dates[row_start:row_start + 4]
            forecast_columns = st.columns(len(row_dates))
            for column, index in zip(forecast_columns, range(row_start, row_start + len(row_dates))):
                day = dates[index]
                high = highs[index] if index < len(highs) and highs[index] is not None else t["missing_value"]
                low = lows[index] if index < len(lows) and lows[index] is not None else t["missing_value"]
                rain = rains[index] if index < len(rains) and rains[index] is not None else t["missing_value"]
                with column:
                    with st.container(border=True):
                        st.markdown(f"**{day}**")
                        st.caption(f"{t['min_temp']}: {low} °C")
                        st.caption(f"{t['max_temp']}: {high} °C")
                        st.caption(f"{t['rain']}: {rain}{'%' if rain != t['missing_value'] else ''}")
        if not dates:
            st.info(t["weather_unavailable_simple"])

    except (requests.RequestException, ValueError, KeyError, TypeError):
        logger.exception("Weather request or response failed.")
        st.warning(t["weather_error"])


def select_plot_for_scan(t, key):
    plots = list_plots()
    options = [None] + [plot["id"] for plot in plots]
    return st.selectbox(
        t["scan_plot"],
        options=options,
        key=key,
        format_func=lambda plot_id: (
            t["unassigned_plot"]
            if plot_id is None
            else next(
                f"{plot['farm_name']} / {plot['plot_name']}"
                for plot in plots
                if plot["id"] == plot_id
            )
        ),
    )


def store_scan_history(plot_id, results, t):
    try:
        save_scan(plot_id, results)
    except (ValueError, OSError, sqlite3.Error):
        logger.exception("Scan prediction could not be saved to history.")
        st.warning(t["scan_save_error"])


# =====================================================
# 10. MAIN INTERFACE
# =====================================================

st.html("""
<style>
html, body, [class*="stApp"] {
  font-family: "Noto Sans Tamil", Latha, "Nirmala UI", sans-serif;
  line-height: 1.55;
}
h1, h2, h3 { line-height: 1.35; }
div[data-testid="stButton"] > button,
div[data-testid="stFormSubmitButton"] > button {
  min-height: 3rem;
  border-radius: 0.8rem;
  font-weight: 650;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: #dce7dc;
  border-radius: 1rem;
}
div.st-key-mobile_bottom_nav { display: none; }
@media (max-width: 640px) {
  .main .block-container {
    padding: 1rem 0.8rem 6.5rem;
  }
  div.st-key-mobile_bottom_nav {
    display: block;
    position: fixed;
    bottom: 0.35rem;
    left: 0.35rem;
    right: 0.35rem;
    width: auto;
    max-width: calc(100vw - 0.7rem);
    box-sizing: border-box;
    z-index: 999;
    border: 1px solid #d6e3d7;
    border-radius: 1rem;
    background: #fff;
    box-shadow: 0 2px 14px rgba(31, 58, 39, 0.16);
    padding: 0.2rem 0.4rem;
  }
  div.st-key-mobile_bottom_nav [role="radiogroup"] {
    display: flex;
    justify-content: space-around;
    gap: 0.1rem;
  }
  div.st-key-mobile_bottom_nav [role="radio"] {
    padding: 0.35rem 0.15rem;
    min-height: 2.8rem;
    font-size: 0.72rem;
  }
  div[data-testid="stSidebar"] div[data-testid="stRadio"] {
    display: none;
  }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; }
}
</style>
""")


def _select_primary_route(route_key):
    if route_key in {"home", "disease", "weather", "reminders", "more"}:
        st.session_state["primary_route"] = route_key
        st.session_state["mobile_route"] = route_key
        st.session_state.pop("more_page", None)
    else:
        st.session_state["primary_route"] = "more"
        st.session_state["mobile_route"] = "more"
        st.session_state["more_page"] = route_key


def _select_mobile_route():
    route = st.session_state["mobile_route"]
    st.session_state["primary_route"] = route
    st.session_state.pop("more_page", None)


def _sync_mobile_route():
    st.session_state["mobile_route"] = st.session_state["primary_route"]
    st.session_state.pop("more_page", None)


def _select_more_page(route_key):
    st.session_state["more_page"] = route_key


st.sidebar.title("🌱 FarmAI")
header_columns = st.columns([3, 1])
with header_columns[1]:
    language = st.selectbox(
        "தமிழ் | English",
        ["English", "தமிழ்"],
        key="app_language",
    )

t = TEXT[language]

st.sidebar.markdown("---")
primary_routes = ["home", "disease", "weather", "reminders", "more"]
route_titles = {
    "home": t["home"],
    "disease": t["disease"],
    "weather": t["weather"],
    "reminders": t["home_tasks"],
    "more": t["more"],
}
if st.session_state.get("primary_route") not in primary_routes:
    st.session_state["primary_route"] = "home"
if st.session_state.get("mobile_route") not in primary_routes:
    st.session_state["mobile_route"] = st.session_state["primary_route"]
primary_route = st.sidebar.radio(
    t["menu"],
    primary_routes,
    format_func=lambda route: route_titles[route],
    key="primary_route",
    on_change=_sync_mobile_route,
)
st.sidebar.button(
    "🏠 " + t["go_home"],
    key="sidebar_home_button",
    disabled=primary_route == "home",
    on_click=_select_primary_route,
    args=("home",),
    width="stretch",
)
more_routes = [
    "farm", "irrigation", "calendar", "soil", "expenses",
    "history", "care", "voice", "settings",
]
more_titles = {route: t[route] for route in more_routes}
if primary_route == "more":
    route_key = st.session_state.get("more_page") or "more"
else:
    route_key = primary_route
    st.session_state.pop("more_page", None)

page = t[route_key] if route_key in t else t["home"]

if route_key == "disease":
    try:
        model = load_model()
        labels = load_labels()
        database = load_treatment_database()
    except Exception as error:
        logger.exception("The crop-check model or reference data could not be loaded.")
        st.error(t["missing_model"])
        model = None
        labels = None
        database = {}
else:
    model = None
    labels = None
    database = {}

if route_key not in ("home", "more"):
    if route_key in more_routes:
        st.button(
            "← " + t["more"],
            key="return_to_more",
            on_click=lambda: st.session_state.pop("more_page", None),
        )
    st.title("🌱 " + t[route_key])

# =====================================================
# 11. HOME PAGE
# =====================================================

if page == t["home"]:
    st.title(t["welcome"])
    st.write(t["welcome_subtitle"])
    st.markdown(f"### {t['task_select_action']}")
    action_cards = [
        ("disease", "🌿", t["home_check_plant"], t["upload_help"]),
        ("weather", "🌦️", t["home_weather"], t["weather_today_title"]),
        ("irrigation", "💧", t["home_water"], t["irrigation_action"]),
        ("reminders", "🗓️", t["home_tasks"], t["reminders_app_only"]),
    ]
    for card_row in (action_cards[:2], action_cards[2:]):
        columns = st.columns(2)
        for column, (destination, icon, title, description) in zip(columns, card_row):
            with column:
                with st.container(border=True):
                    st.markdown(f"### {icon} {title}")
                    st.write(description)
                    st.button(
                        title,
                        key=f"home_action_{destination}",
                        type="primary",
                        width="stretch",
                        on_click=_select_primary_route,
                        args=(destination,),
                    )

    weather_col, tasks_col = st.columns([1, 1])
    with weather_col:
        st.subheader("🌤️ " + t["home_weather_title"])
        home_place = st.selectbox(
            t["weather_location"],
            list(LOCATIONS),
            key="home_weather_location",
            label_visibility="collapsed",
        )
        try:
            with st.spinner(t["checking_connection"]):
                home_weather = get_weather(*LOCATIONS[home_place])
            current = home_weather["current"]
            daily = home_weather["daily"]
            rain_today = (
                daily.get("precipitation_probability_max", [None])[0]
                if daily.get("precipitation_probability_max")
                else None
            )
            first, second = st.columns(2)
            first.metric(t["temperature_short"], f"{current['temperature_2m']} °C")
            second.metric(
                t["rain"],
                f"{rain_today}%" if rain_today is not None else t["missing_value"],
            )
            st.caption(f"{home_place} · {t['weather_updated']}: {current.get('time', t['missing_value'])}")
            if rain_today is not None:
                rain_message = (
                    t["no_rain_forecast"] if rain_today == 0
                    else t["rain_expected"] if rain_today >= 50
                    else t["rain_possible"]
                )
                st.info(rain_message)
        except (requests.RequestException, ValueError, KeyError, TypeError):
            logger.info("Home weather preview is unavailable.")
            st.info(t["weather_unavailable_simple"])
        if st.button(t["view_weather"], key="home_view_weather", width="stretch",
                     on_click=_select_primary_route, args=("weather",)):
            pass
    with tasks_col:
        st.subheader("🗓️ " + t["home_tasks_title"])
        try:
            upcoming_tasks = list_tasks(include_completed=False)[:3]
            if upcoming_tasks:
                for task in upcoming_tasks:
                    with st.container(border=True):
                        st.markdown(f"**{task['title']}**")
                        task_due_date = date.fromisoformat(task["due_date"])
                        task_due_label = (
                            t["task_overdue"] if task_due_date < date.today()
                            else t["task_due"] if task_due_date == date.today()
                            else t["task_upcoming_section"]
                        )
                        st.caption(
                            f"{task_due_label}: {task['due_date']} · "
                            f"{task['notes'] or task['category']}"
                        )
            else:
                st.info(t["home_no_tasks"])
        except (OSError, sqlite3.Error):
            logger.exception("Home task preview could not be loaded.")
            st.info(t["local_error"])
        st.button(t["home_tasks"], key="home_view_tasks", width="stretch",
                  on_click=_select_primary_route, args=("reminders",))

elif route_key == "more" and st.session_state.get("more_page") not in more_routes:
    st.title(t["more_title"])
    st.write(t["more_description"])
    for row_start in range(0, len(more_routes), 2):
        columns = st.columns(2)
        for column, destination in zip(columns, more_routes[row_start:row_start + 2]):
            with column:
                if st.button(
                    more_titles[destination],
                    key=f"more_open_{destination}",
                    width="stretch",
                    on_click=_select_more_page,
                    args=(destination,),
                ):
                    pass

# =====================================================
# 12. DISEASE DETECTION PAGE
# =====================================================

elif page == t["disease"]:
    st.subheader("🌿 " + t["disease"])
    st.write(t["prediction_not_diagnosis"])
    st.markdown(f"### {t['plant_step_one']}")
    photo_method = st.radio(
        t["task_select_action"],
        ["camera", "upload"],
        format_func=lambda value: t["take_photo"] if value == "camera" else t["choose_photo"],
        horizontal=True,
        key="disease_photo_method",
    )
    try:
        scan_plot_id = select_plot_for_scan(t, "disease_scan_plot")
    except (OSError, sqlite3.Error):
        logger.exception("Farm plots could not be loaded for scan association.")
        st.error(t["local_error"])
        scan_plot_id = None

    image_file = (
        st.camera_input(t["take_photo"], key="disease_camera")
        if photo_method == "camera"
        else st.file_uploader(
            t["choose_photo"],
            type=["jpg", "jpeg", "png"],
            key="disease_upload",
        )
    )

    if image_file is not None:
        try:
            image = load_uploaded_image(image_file)
            st.image(image, width="stretch")
        except ImageTooLargeError:
            st.error(t["large_image"])
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            st.error(t["bad_image"])
        else:
            st.markdown(f"### {t['plant_step_two']}")
            if st.button(t["analyze"], type="primary", key="disease_analyze", width="stretch"):
                if model is None or labels is None:
                    st.error(t["missing_model"] if model is None else t["missing_labels"])
                else:
                    try:
                        with st.spinner(t["checking_plant"]):
                            results = run_prediction(model, labels, image)
                    except Exception:
                        logger.exception("Tomato image prediction failed.")
                        st.error(t["prediction_error"])
                    else:
                        st.session_state["results"] = results
                        st.session_state["results_language"] = language
                        store_scan_history(scan_plot_id, results, t)
            results = st.session_state.get("results")
            if not results:
                st.info(t["no_result_yet"])
            elif st.session_state.get("results_language") != language:
                st.info(t["recheck"])
            else:
                st.markdown(f"### {t['plant_step_three']}")
                st.subheader("🔍 " + t["prediction"])
                best = results[0]
                class_key = best["key"]
                confidence = best["score"]
                st.markdown("### " + readable_class_name(class_key, language))
                st.write(f"**{t['confidence']}: {confidence:.1%}**")
                st.caption(t["prediction_not_diagnosis"])
                if not is_confident_prediction(confidence):
                    st.warning(t["result_uncertain_simple"])
                elif normalize_class_key(class_key) == "healthy":
                    st.success(t["healthy_result"])
                else:
                    st.warning(t["possible_problem"])
                    with st.expander(t["what_it_means"], expanded=True):
                        display_disease_information(class_key, language, t)
                    with st.expander(t["what_to_do"], expanded=True):
                        st.info(t["verified_source_unavailable"])
                st.caption(t["local_verification_short"])
    else:
        st.info(t["no_result_yet"])


# =====================================================
# 13. FARM PAGE
# =====================================================

elif page == t["farm"]:
    st.subheader("🌾 " + t["farm_title"])
    try:
        plots = list_plots()
        selected_plot = None
        if plots:
            selected_plot_id = st.selectbox(
                t["select_plot"],
                options=[plot["id"] for plot in plots],
                format_func=lambda plot_id: next(
                    f"{plot['farm_name']} / {plot['plot_name']}"
                    for plot in plots
                    if plot["id"] == plot_id
                ),
            )
            selected_plot = next(plot for plot in plots if plot["id"] == selected_plot_id)
        else:
            st.info(t["no_plots"])

        if language == "தமிழ்":
            area_labels = {"cents": "சென்ட்", "acres": "ஏக்கர்", "hectares": "ஹெக்டேர்"}
            cultivation_labels = {
                "Open field": "திறந்த வெளி",
                "Protected cultivation": "பாதுகாக்கப்பட்ட சாகுபடி",
            }
            irrigation_labels = {
                "Not specified": "குறிப்பிடப்படவில்லை",
                "Drip": "சொட்டு நீர்ப்பாசனம்",
                "Furrow": "பாத்தி நீர்ப்பாசனம்",
                "Sprinkler": "தெளிப்பு நீர்ப்பாசனம்",
                "Other": "மற்றவை",
            }
            stage_labels = {
                "Not set": "தேர்ந்தெடுக்கப்படவில்லை",
                "Nursery": "நாற்றங்கால்",
                "Vegetative": "வளர்ச்சி",
                "Flowering": "பூக்கும் நிலை",
                "Fruit development": "காய் வளர்ச்சி",
                "Harvest": "அறுவடை",
                "Completed": "முடிந்தது",
            }
        else:
            area_labels = {value: value for value in AREA_UNITS}
            cultivation_labels = {value: value for value in CULTIVATION_TYPES}
            irrigation_labels = {value: value for value in IRRIGATION_METHODS}
            stage_labels = {value: value for value in CROP_STAGES}

        st.caption(t["stage_note"])
        with st.form("plot_profile_form"):
            col1, col2 = st.columns(2)
            with col1:
                farm_name = st.text_input(t["farm_name"], value=(selected_plot or {}).get("farm_name", ""))
                plot_name = st.text_input(t["plot_name"], value=(selected_plot or {}).get("plot_name", ""))
                village = st.text_input(t["village"], value=(selected_plot or {}).get("village", ""))
                district = st.text_input(t["district"], value=(selected_plot or {}).get("district", ""))
                state = st.text_input(t["state"], value=(selected_plot or {}).get("state", ""))
                variety = st.text_input(t["variety"], value=(selected_plot or {}).get("tomato_variety", ""))
                planting_date = st.date_input(
                    t["planting_date"],
                    value=date.fromisoformat(selected_plot["planting_date"]) if selected_plot else date.today(),
                )
            with col2:
                area = st.number_input(
                    t["area"],
                    min_value=0.01,
                    value=float((selected_plot or {}).get("area", 1.0)),
                    step=0.1,
                )
                area_unit = st.selectbox(
                    t["area_unit"],
                    options=sorted(AREA_UNITS),
                    index=sorted(AREA_UNITS).index((selected_plot or {}).get("area_unit", "acres")),
                    format_func=lambda value: area_labels[value],
                )
                soil_type = st.text_input(t["soil_type"], value=(selected_plot or {}).get("soil_type", ""))
                cultivation_type = st.selectbox(
                    t["cultivation_type"],
                    options=sorted(CULTIVATION_TYPES),
                    index=sorted(CULTIVATION_TYPES).index((selected_plot or {}).get("cultivation_type", "Open field")),
                    format_func=lambda value: cultivation_labels[value],
                )
                irrigation_method = st.selectbox(
                    t["irrigation_method"],
                    options=sorted(IRRIGATION_METHODS),
                    index=sorted(IRRIGATION_METHODS).index((selected_plot or {}).get("irrigation_method", "Not specified")),
                    format_func=lambda value: irrigation_labels[value],
                )
                crop_stage = st.selectbox(
                    t["crop_stage"],
                    options=sorted(CROP_STAGES),
                    index=sorted(CROP_STAGES).index((selected_plot or {}).get("crop_stage", "Not set")),
                    format_func=lambda value: stage_labels[value],
                )
                notes = st.text_area(t["notes"], value=(selected_plot or {}).get("notes", ""))

            submitted = st.form_submit_button(
                t["update_plot"] if selected_plot else t["add_plot"],
                type="primary",
            )
        if submitted:
            try:
                save_plot(
                    {
                        "farm_name": farm_name,
                        "plot_name": plot_name,
                        "village": village,
                        "district": district,
                        "state": state,
                        "tomato_variety": variety,
                        "planting_date": planting_date,
                        "area": area,
                        "area_unit": area_unit,
                        "soil_type": soil_type,
                        "cultivation_type": cultivation_type,
                        "irrigation_method": irrigation_method,
                        "crop_stage": crop_stage,
                        "notes": notes,
                    },
                    (selected_plot or {}).get("id"),
                )
                st.success(t["plot_saved"])
                st.rerun()
            except (ValueError, OSError, sqlite3.Error) as error:
                if isinstance(error, ValueError):
                    st.error(t["plot_input_error"])
                else:
                    logger.exception("Farm plot could not be saved.")
                    st.error(f"Farm records could not be saved: {error}")

        if selected_plot:
            confirm_delete = st.checkbox(
                t["confirm_delete"],
                key=f"confirm_delete_{selected_plot['id']}",
            )
            if st.button(t["delete_plot"], disabled=not confirm_delete):
                try:
                    delete_plot(selected_plot["id"])
                    st.success(t["plot_deleted"])
                    st.rerun()
                except (ValueError, OSError, sqlite3.Error) as error:
                    logger.exception("Farm plot could not be deleted.")
                    st.error(f"Farm records could not be deleted: {error}")

        if plots:
            csv_output = StringIO()
            writer = csv.DictWriter(csv_output, fieldnames=plots[0].keys())
            writer.writeheader()
            writer.writerows(plots)
            st.download_button(
                t["export_plots"],
                data=csv_output.getvalue(),
                file_name="tomato_farm_plots.csv",
                mime="text/csv",
            )
    except (OSError, sqlite3.Error) as error:
        logger.exception("Farm plot database operation failed.")
        st.error(f"Farm records could not be loaded or saved: {error}")

# =====================================================
# 14. IRRIGATION ADVISOR
# =====================================================

elif page == t["irrigation"]:
    st.subheader("💧 " + t["irrigation_title"])
    st.info(t["irrigation_safety"])
    try:
        plots = list_plots()
        if not plots:
            st.info(t["irrigation_missing"])
        else:
            plot_id = st.selectbox(
                t["select_plot"],
                [plot["id"] for plot in plots],
                key="irrigation_plot",
                format_func=lambda selected_id: next(
                    f"{plot['farm_name']} / {plot['plot_name']}"
                    for plot in plots
                    if plot["id"] == selected_id
                ),
            )
            selected_plot = next(plot for plot in plots if plot["id"] == plot_id)
            latest_date_text = latest_irrigation_date(plot_id)
            latest_date = date.fromisoformat(latest_date_text) if latest_date_text else None
            st.write(f"**{t['last_irrigation']}:** {latest_date_text or t['irrigation_no_log']}")

            stage_labels = {
                "Not set": "தேர்ந்தெடுக்கப்படவில்லை",
                "Nursery": "நாற்றங்கால்",
                "Vegetative": "வளர்ச்சி",
                "Flowering": "பூக்கும் நிலை",
                "Fruit development": "காய் வளர்ச்சி",
                "Harvest": "அறுவடை",
                "Completed": "முடிந்தது",
            } if language == "தமிழ்" else {value: value for value in CROP_STAGES}
            selected_stage = st.selectbox(
                t["crop_stage"],
                sorted(CROP_STAGES),
                index=sorted(CROP_STAGES).index(selected_plot["crop_stage"]),
                format_func=lambda value: stage_labels[value],
                key="irrigation_stage",
            )
            st.caption(f"{t['soil_type']}: {selected_plot['soil_type'] or t['no_data']}")
            weather_location = st.selectbox(
                t["forecast_location"],
                list(LOCATIONS),
                key="irrigation_weather_location",
            )
            st.caption(t["forecast_internet"])
            if st.button(t["fetch_forecast"], key="irrigation_fetch_forecast"):
                try:
                    forecast = get_weather(*LOCATIONS[weather_location])
                    rain = forecast["daily"].get("precipitation_probability_max", [])
                    rain_value = float(rain[0]) if rain and rain[0] is not None else None
                    st.session_state["irrigation_forecast"] = {
                        "location": weather_location,
                        "rain_probability": rain_value,
                    }
                except (requests.RequestException, ValueError, KeyError, TypeError):
                    logger.warning("Irrigation weather forecast was unavailable.")
                    st.session_state["irrigation_forecast"] = {
                        "location": weather_location,
                        "rain_probability": None,
                        "failed": True,
                    }
            saved_forecast = st.session_state.get("irrigation_forecast", {})
            rain_probability = (
                saved_forecast.get("rain_probability")
                if saved_forecast.get("location") == weather_location
                else None
            )
            if rain_probability is not None:
                st.caption(f"{t['forecast_status']}: {rain_probability:.0f}% rain probability — forecast only.")
            elif saved_forecast.get("location") == weather_location and saved_forecast.get("failed"):
                st.caption(t["forecast_unavailable"])
            else:
                st.caption(t["forecast_not_loaded"])

            advice = build_irrigation_advice(
                "Tomato",
                None if selected_stage == "Not set" else selected_stage,
                selected_plot.get("soil_type"),
                latest_date,
                rain_probability,
            )
            st.markdown(f"**{t['irrigation_reasoning']}**")
            st.write(t["irrigation_action"])
            if language == "தமிழ்":
                reasoning = [
                    "இந்தப் பயிர் மற்றும் பகுதிக்கான உள்ளூரில் சரிபார்க்கப்பட்ட நீர்ப்பாசன அட்டவணை இல்லை.",
                    "முன்னறிவிப்பு உண்மையான மழை அளவீடு அல்ல; இந்தச் செயலியில் மண் ஈரப்பத உணரியும் இல்லை.",
                    "வேர் பகுதியில் மண் ஈரத்தைப் பார்த்து, பயிர் நிலை மற்றும் உள்ளூர் வேளாண் ஆலோசனையை கருத்தில் கொள்ளவும்.",
                ]
                missing_values = [
                    label for field, label in (
                        ("growth_stage", "பயிர் நிலை"),
                        ("soil_type", "மண் வகை"),
                        ("last_irrigation_date", "கடைசி நீர்ப்பாசன தேதி"),
                    )
                    if not advice["inputs"][field]
                ]
                if missing_values:
                    reasoning.append("விடுபட்ட தகவல்: " + ", ".join(missing_values) + ".")
                if rain_probability is not None:
                    reasoning.append(
                        f"மழை வாய்ப்பு: {rain_probability:.0f}% (முன்னறிவிப்பு மட்டும்; வயல் அளவீடு அல்ல)."
                    )
            else:
                reasoning = advice["reasoning"]
            for reason in reasoning:
                st.markdown(f"- {reason}")
            st.warning(t["irrigation_uncertainty"])
            inputs = advice["inputs"]
            st.caption(
                f"{t['crop_name']}: Tomato · {t['crop_stage']}: "
                f"{inputs['growth_stage'] or t['missing_value']} · {t['soil_type']}: "
                f"{inputs['soil_type'] or t['missing_value']} · {t['last_irrigation']}: "
                f"{inputs['last_irrigation_date'] or t['missing_value']} · "
                f"{t['forecast_probability']}: "
                f"{f'{rain_probability:.0f}%' if rain_probability is not None else t['missing_value']}"
            )
            farmer_plan = st.text_area(
                t["schedule_irrigation"],
                value=t["irrigation_action"],
                key="irrigation_farmer_plan",
            )
            reminder_date = st.date_input(
                t["task_due_date"],
                value=date.today(),
                key="irrigation_reminder_date",
            )
            if st.button(t["schedule_irrigation"], key="irrigation_schedule"):
                task_before = list_tasks()
                duplicate = any(
                    task["plot_id"] == plot_id
                    and task["category"] == "irrigation"
                    and task["title"].casefold() == farmer_plan.strip().casefold()
                    and task["due_date"] == reminder_date.isoformat()
                    for task in task_before
                )
                save_task(
                    plot_id,
                    farmer_plan,
                    "irrigation",
                    reminder_date,
                    notes="Farmer-created irrigation check; no water amount prescribed.",
                )
                st.info(t["task_duplicate"] if duplicate else t["task_saved"])
            irrigation_notes = st.text_input(t["irrigation_notes"], key="irrigation_log_notes")
            if st.button(t["log_irrigation"], key="irrigation_log"):
                save_irrigation_log(plot_id, date.today(), irrigation_notes)
                st.success(t["irrigation_saved"])
                st.rerun()
    except (OSError, ValueError, sqlite3.Error) as error:
        if isinstance(error, ValueError):
            st.error(str(error))
        else:
            logger.exception("Irrigation advisor data could not be loaded or saved.")
            st.error(t["local_error"])

# =====================================================
# 15. CALENDAR AND REMINDERS
# =====================================================

elif page in (t["calendar"], t["reminders"]):
    st.subheader("📅 " + (t["reminders_title"] if page == t["reminders"] else t["calendar_title"]))
    st.caption(t["calendar_no_schedule"])
    if page == t["reminders"]:
        st.info(t["reminders_app_only"])
    try:
        plots = list_plots()
        tasks = list_tasks()
        plot_options = [None] + [plot["id"] for plot in plots]
        plot_names = {
            plot["id"]: f"{plot['farm_name']} / {plot['plot_name']}"
            for plot in plots
        }
        category_labels = {
            "irrigation": "நீர்ப்பாசனம்" if language == "தமிழ்" else "Irrigation",
            "crop task": "பயிர்ப் பணி" if language == "தமிழ்" else "Crop task",
            "reminder": "நினைவூட்டல்" if language == "தமிழ்" else "Reminder",
        }
        status_labels = {
            "pending": t["task_pending"],
            "completed": t["task_completed"],
            "skipped": t["task_skipped"],
        }
        with st.form("farm_task_form"):
            task_ids = [task["id"] for task in tasks]
            selected_task_id = st.selectbox(
                t["task_edit"],
                [None] + task_ids,
                format_func=lambda value: (
                    t["task_add"] if value is None else next(
                        f"{task['due_date']} — {task['title']}" for task in tasks if task["id"] == value
                    )
                ),
            )
            selected_task = next(
                (task for task in tasks if task["id"] == selected_task_id), None
            )
            title = st.text_input(t["task_title"], value=(selected_task or {}).get("title", ""))
            category = st.selectbox(
                t["task_category"],
                ["irrigation", "crop task", "reminder"],
                index=["irrigation", "crop task", "reminder"].index(
                    (selected_task or {}).get("category", "reminder")
                ),
                format_func=lambda value: category_labels[value],
            )
            due_date = st.date_input(
                t["task_due_date"],
                value=date.fromisoformat(selected_task["due_date"]) if selected_task else date.today(),
            )
            task_plot_id = st.selectbox(
                t["task_plot"],
                plot_options,
                index=plot_options.index((selected_task or {}).get("plot_id")),
                format_func=lambda value: (
                    t["unassigned_plot"] if value is None else plot_names[value]
                ),
            )
            task_notes = st.text_area(t["task_notes"], value=(selected_task or {}).get("notes", ""))
            task_submitted = st.form_submit_button(
                t["task_update"] if selected_task else t["task_add"],
                type="primary",
            )
        if task_submitted:
            old_task_id = selected_task["id"] if selected_task else None
            duplicate = old_task_id is None and any(
                task["plot_id"] == task_plot_id
                and task["category"] == category
                and task["due_date"] == due_date.isoformat()
                and task["title"].casefold() == title.strip().casefold()
                for task in tasks
            )
            save_task(
                task_plot_id,
                title,
                category,
                due_date,
                task_notes,
                task_id=old_task_id,
            )
            st.info(t["task_duplicate"] if duplicate else t["task_saved"])
            st.rerun()

        today_text = date.today().isoformat()
        task_groups = (
            (
                t["task_today_section"],
                [
                    task for task in tasks
                    if task["status"] == "pending" and task["due_date"] <= today_text
                ],
            ),
            (
                t["task_upcoming_section"],
                [
                    task for task in tasks
                    if task["status"] == "pending" and task["due_date"] > today_text
                ],
            ),
            (
                t["task_completed_section"],
                [task for task in tasks if task["status"] != "pending"],
            ),
        )
        for section_title, section_tasks in task_groups:
            st.markdown(f"### {section_title}")
            if not section_tasks:
                st.caption(t["task_section_empty"])
                continue
            for task in section_tasks:
                due = date.fromisoformat(task["due_date"])
                due_label = (
                    t["task_overdue"] if task["status"] == "pending" and due < date.today()
                    else t["task_due"] if task["status"] == "pending" and due == date.today()
                    else status_labels[task["status"]]
                )
                with st.container(border=True):
                    st.markdown(f"**{task['title']}**")
                    st.caption(f"{due_label}: {task['due_date']}")
                    st.write(task["notes"] or category_labels[task["category"]])
                    if task["status"] == "pending" and st.button(
                        t["task_mark_done"],
                        key=f"mark_task_done_{task['id']}",
                        type="primary",
                        width="stretch",
                    ):
                        set_task_status(task["id"], "completed")
                        st.success(t["task_done_feedback"])
                        st.rerun()

        if tasks:
            task_to_manage = st.selectbox(
                t["task_edit"],
                [task["id"] for task in tasks],
                key="task_manage_id",
                format_func=lambda value: next(
                    f"{task['due_date']} — {task['title']}" for task in tasks if task["id"] == value
                ),
            )
            managed_task = next(task for task in tasks if task["id"] == task_to_manage)
            new_status = st.selectbox(
                t["task_status"],
                ["pending", "completed", "skipped"],
                index=["pending", "completed", "skipped"].index(managed_task["status"]),
                format_func=lambda value: status_labels[value],
                key=f"task_status_{task_to_manage}",
            )
            if st.button(t["task_status"], key="save_task_status"):
                set_task_status(task_to_manage, new_status)
                st.rerun()
            confirm_task_delete = st.checkbox(
                t["confirm_delete_task"],
                key=f"confirm_delete_task_{task_to_manage}",
            )
            if st.button(t["task_delete"], key="delete_task", disabled=not confirm_task_delete):
                delete_task(task_to_manage)
                st.success(t["task_deleted"])
                st.rerun()
        else:
            st.info(t["task_empty"])
    except (OSError, ValueError, sqlite3.Error) as error:
        if isinstance(error, ValueError):
            st.error(str(error))
        else:
            logger.exception("Farm calendar or reminders could not be loaded or saved.")
            st.error(t["local_error"])

# =====================================================
# 16. SOIL TEST RECORDS
# =====================================================

elif page == t["soil"]:
    st.subheader("🧪 " + t["soil_title"])
    st.info(t["soil_education"])
    st.warning(t["soil_no_rates"])
    try:
        plots = list_plots()
        plot_options = [None] + [plot["id"] for plot in plots]
        plot_names = {plot["id"]: f"{plot['farm_name']} / {plot['plot_name']}" for plot in plots}
        with st.form("soil_test_form"):
            soil_plot_id = st.selectbox(
                t["soil_plot"],
                plot_options,
                format_func=lambda value: t["unassigned_plot"] if value is None else plot_names[value],
            )
            crop_name = st.text_input(t["crop_name"], value="Tomato")
            growth_stage = st.text_input(t["crop_stage"])
            soil_type = st.text_input(t["soil_type"])
            test_date = st.date_input(t["soil_test_date"], value=date.today())
            soil_ph = st.text_input(t["soil_ph"], placeholder="e.g. 6.5")
            nitrogen = st.text_input(t["soil_n"])
            phosphorus = st.text_input(t["soil_p"])
            potassium = st.text_input(t["soil_k"])
            organic_matter = st.text_input(t["soil_organic"])
            soil_notes = st.text_area(t["soil_notes"])
            soil_submitted = st.form_submit_button(t["soil_save"], type="primary")
        if soil_submitted:
            save_soil_test(
                soil_plot_id,
                test_date,
                {
                    "ph": soil_ph,
                    "nitrogen": nitrogen,
                    "phosphorus": phosphorus,
                    "potassium": potassium,
                    "organic_matter": organic_matter,
                },
                soil_notes,
                crop=crop_name,
                growth_stage=growth_stage,
                soil_type=soil_type,
            )
            st.success(t["soil_saved"])
            st.rerun()
        soil_records = list_soil_tests()
        if soil_records:
            st.dataframe(soil_records, width="stretch", hide_index=True)
        else:
            st.info(t["soil_empty"])
    except (OSError, ValueError, sqlite3.Error) as error:
        if isinstance(error, ValueError):
            st.error(str(error))
        else:
            logger.exception("Soil-test record could not be loaded or saved.")
            st.error(t["local_error"])

# =====================================================
# 17. FARM FINANCES
# =====================================================

elif page == t["expenses"]:
    st.subheader("💰 " + t["expenses_title"])
    st.caption(t["financial_actual_note"])
    try:
        plots = list_plots()
        plot_options = [None] + [plot["id"] for plot in plots]
        plot_names = {plot["id"]: f"{plot['farm_name']} / {plot['plot_name']}" for plot in plots}
        expense_category_labels = {
            "seeds": "விதைகள்",
            "fertilizer": "உரம்",
            "pesticides": "பூச்சிக்கொல்லிகள்",
            "irrigation": "நீர்ப்பாசனம்",
            "labour": "தொழிலாளர்",
            "transport": "போக்குவரத்து",
            "miscellaneous": "இதர செலவுகள்",
        } if language == "தமிழ்" else {value: value.title() for value in EXPENSE_CATEGORIES}
        entry_type_label = st.selectbox(
            t["record_type"], [t["expense"], t["sale"]], key="financial_type"
        )
        transaction_type = "expense" if entry_type_label == t["expense"] else "sale"
        with st.form("financial_record_form"):
            financial_plot_id = st.selectbox(
                t["financial_plot"],
                plot_options,
                format_func=lambda value: t["unassigned_plot"] if value is None else plot_names[value],
            )
            transaction_day = st.date_input(t["transaction_date"], value=date.today())
            description = st.text_input(t["financial_description"])
            expense_category = st.selectbox(
                t["expense_category"],
                sorted(EXPENSE_CATEGORIES),
                format_func=lambda value: expense_category_labels[value],
            )
            if transaction_type == "expense":
                amount = st.number_input(t["amount_rupees"], min_value=0.0, step=1.0)
                quantity = None
                price = None
            else:
                amount = None
                quantity = st.number_input(t["quantity_kg"], min_value=0.0, step=1.0)
                price = st.number_input(t["price_per_kg"], min_value=0.0, step=1.0)
            financial_submitted = st.form_submit_button(t["financial_save"], type="primary")
        if financial_submitted:
            save_financial_record(
                financial_plot_id,
                transaction_type,
                expense_category if transaction_type == "expense" else "harvest sale",
                description,
                transaction_day,
                amount=amount,
                quantity_kg=quantity,
                selling_price_per_kg=price,
            )
            st.success(t["financial_saved"])
            st.rerun()

        start_day, end_day = st.columns(2)
        with start_day:
            filter_start = st.date_input(t["financial_from"], value=date.today().replace(day=1))
        with end_day:
            filter_end = st.date_input(t["financial_to"], value=date.today())
        if filter_start > filter_end:
            st.error(t["financial_from"] + " <= " + t["financial_to"])
        else:
            records = list_financial_records(filter_start, filter_end)
            summary = calculate_financial_summary(records)
            expenses_rupees = summary["expenses_paise"] / 100
            income_rupees = summary["income_paise"] / 100
            profit_rupees = summary["profit_paise"] / 100
            first, second, third = st.columns(3)
            first.metric(t["financial_expenses"], f"₹{expenses_rupees:,.2f}")
            second.metric(t["financial_income"], f"₹{income_rupees:,.2f}")
            third.metric(t["financial_profit"], f"₹{profit_rupees:,.2f}")
            if records:
                financial_rows = [
                    {
                        t["transaction_date"]: record["transaction_date"],
                        t["record_type"]: t["expense"] if record["transaction_type"] == "expense" else t["sale"],
                        t["expense_category"]: expense_category_labels.get(
                            record["category"], record["category"]
                        ),
                        t["financial_description"]: record["description"],
                        t["amount_rupees"]: f"₹{record['amount_paise'] / 100:,.2f}",
                        t["quantity_kg"]: record["quantity_kg"] or "",
                        t["financial_plot"]: (
                            f"{record['farm_name']} / {record['plot_name']}"
                            if record["farm_name"] else t["unassigned_plot"]
                        ),
                    }
                    for record in records
                ]
                st.dataframe(financial_rows, width="stretch", hide_index=True)
                st.bar_chart({
                    t["financial_expenses"]: [expenses_rupees],
                    t["financial_income"]: [income_rupees],
                    t["financial_profit"]: [profit_rupees],
                })
            else:
                st.info(t["financial_empty"])
    except (OSError, ValueError, sqlite3.Error) as error:
        if isinstance(error, ValueError):
            st.error(str(error))
        else:
            logger.exception("Financial record could not be loaded or saved.")
            st.error(t["local_error"])

# =====================================================
# 18. VOICE/TYPED ASSISTANT AND SETTINGS
# =====================================================

elif page == t["voice"]:
    st.subheader("🗣️ " + t["voice_title"])
    st.info(t["voice_unavailable"])
    assistant_language = st.selectbox(
        t["voice_language"],
        ["English", "தமிழ்"],
        index=0 if language == "English" else 1,
        key="assistant_language",
    )
    assistant_text = TEXT[assistant_language]
    voice_labels = {
        "start": assistant_text["voice_start"],
        "stop": assistant_text["voice_stop"],
        "speak": assistant_text["voice_speak"],
        "ready": assistant_text["voice_ready"],
        "listening": assistant_text["voice_listening"],
        "transcriptReceived": assistant_text["voice_transcript_received"],
        "permissionDenied": assistant_text["voice_permission_denied"],
        "noMicrophone": assistant_text["voice_no_microphone"],
        "noSpeech": assistant_text["voice_no_speech"],
        "networkError": assistant_text["voice_network_error"],
        "stopped": assistant_text["voice_stopped"],
        "recognitionError": assistant_text["voice_recognition_error"],
        "recognitionUnsupported": assistant_text["voice_recognition_unsupported"],
        "synthesisUnsupported": assistant_text["voice_synthesis_unsupported"],
        "speaking": assistant_text["voice_speaking"],
        "speechComplete": assistant_text["voice_speech_complete"],
        "synthesisError": assistant_text["voice_synthesis_error"],
        "noAnswer": assistant_text["voice_no_answer"],
        "privacy": assistant_text["voice_privacy"],
    }
    saved_answer = st.session_state.get("assistant_answer", "")
    voice_result = browser_voice_controls(
        language="en-IN" if assistant_language == "English" else "ta-IN",
        speech_text=(
            saved_answer
            if st.session_state.get("assistant_answer_language") == assistant_language
            else ""
        ),
        labels=voice_labels,
    )
    transcript = getattr(voice_result, "transcript", None)
    if isinstance(transcript, str) and transcript.strip():
        st.session_state["assistant_question"] = transcript.strip()
    question = st.text_area(t["voice_input"], key="assistant_question")

    if st.button(t["voice_send"], key="assistant_send"):
        if not question.strip():
            st.warning(assistant_text["voice_no_answer"])
        else:
            query = question.casefold()
            answer = assistant_text["voice_general"]
            try:
                if any(word in query for word in ("expense", "income", "profit", "செலவு", "வருமானம்", "லாபம்")):
                    records = list_financial_records()
                    summary = calculate_financial_summary(records)
                    if records:
                        answer = (
                            f"{assistant_text['financial_expenses']}: ₹{summary['expenses_paise'] / 100:,.2f}; "
                            f"{assistant_text['financial_income']}: ₹{summary['income_paise'] / 100:,.2f}; "
                            f"{assistant_text['financial_profit']}: ₹{summary['profit_paise'] / 100:,.2f}."
                        )
                    else:
                        answer = assistant_text["voice_no_records"]
                elif any(word in query for word in ("scan", "disease", "நோய்", "பரிசோதனை")):
                    scans = list_scans()
                    answer = (
                        f"{assistant_text['history_title']}: {len(scans)}"
                        if scans
                        else assistant_text["voice_no_records"]
                    )
                elif any(word in query for word in ("task", "reminder", "பணி", "நினைவூட்டல்")):
                    tasks = list_tasks(include_completed=False)
                    answer = (
                        f"{assistant_text['task_scheduled']}: {len(tasks)}"
                        if tasks
                        else assistant_text["voice_no_records"]
                    )
            except (OSError, sqlite3.Error, ValueError) as error:
                if isinstance(error, ValueError):
                    st.error(str(error))
                else:
                    logger.exception("Typed farm assistant could not read local data.")
                    st.error(t["local_error"])
            else:
                st.session_state["assistant_answer"] = answer
                st.session_state["assistant_answer_language"] = assistant_language
                st.rerun()

    if saved_answer:
        if st.session_state.get("assistant_answer_language") != assistant_language:
            st.info(assistant_text["recheck"])
        else:
            st.markdown(f"**{assistant_text['voice_answer']}:** {saved_answer}")

elif page == t["settings"]:
    st.subheader("⚙️ " + t["settings_title"])
    st.info(t["local_only"])
    st.caption(t["sync_unavailable"])
    st.warning(t["source_unverified"])
    st.write(f"**{t['backup_path']}:** `{DATABASE_PATH}`")
    st.caption("Back up the app data directory while the application is stopped. Keep the backup private.")
    try:
        st.write(f"{t['task_scheduled']}: {len(list_tasks())}")
    except (OSError, sqlite3.Error) as error:
        logger.exception("Local storage status could not be read.")
        st.error(f"{t['settings_title']}: {error}")
    confirmed_delete_all = st.checkbox(t["delete_all_confirm"])
    if st.button(t["delete_all_button"], disabled=not confirmed_delete_all):
        try:
            delete_local_records()
            st.success(t["delete_all_done"])
            st.rerun()
        except (OSError, sqlite3.Error, ValueError) as error:
            logger.exception("Local farm records could not be deleted.")
            st.error(t["local_error"])

# =====================================================
# 19. WEATHER PAGE
# =====================================================

elif page == t["weather"]:
    display_weather(t)


# =====================================================
# 20. CROP CARE PAGE
# =====================================================

elif page == t["care"]:
    st.subheader("🌿 " + t["care_title"])

    for item in t["care_items"]:
        st.markdown("- " + item)

    st.info(t["confirm"])

elif page == t["history"]:
    st.subheader("📋 " + t["history_title"])
    try:
        scans = list_scans()
        if not scans:
            st.info(t["history_empty"])
        else:
            history_rows = [
                {
                    "Date/time": scan["created_at"],
                    t["history_plot"]: (
                        f"{scan['farm_name']} / {scan['plot_name']}"
                        if scan["farm_name"] and scan["plot_name"]
                        else t["unassigned_plot"]
                    ),
                    t["history_prediction"]: readable_class_name(scan["predicted_label"], language),
                    t["history_score"]: f"{scan['confidence']:.1%}",
                    t["history_confirmation"]: (
                        readable_class_name(scan["farmer_confirmed_label"], language)
                        if scan["farmer_confirmed_label"]
                        else t["history_unconfirmed"]
                    ),
                }
                for scan in scans
            ]
            st.dataframe(history_rows, width="stretch", hide_index=True)
            history_csv = StringIO()
            history_writer = csv.DictWriter(history_csv, fieldnames=history_rows[0].keys())
            history_writer.writeheader()
            history_writer.writerows(history_rows)
            st.download_button(
                t["history_export"],
                data=history_csv.getvalue(),
                file_name="tomato_scan_history.csv",
                mime="text/csv",
            )
            selected_scan_id = st.selectbox(
                t["history_select"],
                options=[scan["id"] for scan in scans],
                format_func=lambda scan_id: next(
                    f"{scan['created_at']} — {readable_class_name(scan['predicted_label'], language)}"
                    for scan in scans
                    if scan["id"] == scan_id
                ),
            )
            selected_scan = next(scan for scan in scans if scan["id"] == selected_scan_id)
            st.caption(t["history_confirmation_note"])
            try:
                labels = load_labels() or []
                confirmation_options = [None] + [
                    normalize_class_key(label) for label in labels
                ]
                current_confirmation = selected_scan["farmer_confirmed_label"]
                confirmation_index = (
                    confirmation_options.index(current_confirmation)
                    if current_confirmation in confirmation_options
                    else 0
                )
                confirmed_label = st.selectbox(
                    t["history_confirmation"],
                    options=confirmation_options,
                    index=confirmation_index,
                    format_func=lambda label: (
                        t["history_no_confirmation"]
                        if label is None
                        else readable_class_name(label, language)
                    ),
                )
                if st.button(t["history_save_confirmation"]):
                    confirm_scan(selected_scan_id, confirmed_label)
                    st.success(t["history_confirmation_saved"])
                    st.rerun()
                confirm_delete_scan = st.checkbox(
                    t["history_confirm_delete"],
                    key=f"confirm_scan_delete_{selected_scan_id}",
                )
                if st.button(
                    t["history_delete"],
                    key=f"delete_scan_{selected_scan_id}",
                    disabled=not confirm_delete_scan,
                ):
                    delete_scan(selected_scan_id)
                    st.rerun()
            except (OSError, ValueError, sqlite3.Error) as error:
                logger.exception("Scan confirmation could not be loaded or saved.")
                st.error(f"Scan confirmation could not be saved: {error}")
    except (OSError, sqlite3.Error, ValueError) as error:
        logger.exception("Scan history could not be loaded.")
        st.error(f"Scan history could not be loaded: {error}")


with st.container(key="mobile_bottom_nav"):
    st.radio(
        t["menu"],
        primary_routes,
        format_func=lambda route: route_titles[route],
        key="mobile_route",
        horizontal=True,
        label_visibility="collapsed",
        on_change=_select_mobile_route,
    )

# =====================================================
# 21. FOOTER
# =====================================================
st.markdown("---")
st.caption(t["footer"])
