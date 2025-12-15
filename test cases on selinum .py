

import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# -------------------------
# Fixture: Browser Setup
# -------------------------
@pytest.fixture
def driver():
    driver = webdriver.Chrome()  
    driver.maximize_window()
    driver.get("http://localhost:3000")  
    yield driver
    driver.quit()

# -------------------------
# TC-01: App Launch
# -------------------------
def test_TC01_app_launch(driver):
    assert "Health App" in driver.title

# -------------------------
# TC-02: Valid Registration
# -------------------------
def test_TC02_valid_registration(driver):
    driver.find_element(By.ID, "name").send_keys("Hania")
    driver.find_element(By.ID, "age").send_keys("22")
    driver.find_element(By.ID, "gender").send_keys("Female")
    driver.find_element(By.ID, "submit").click()
    assert "Profile Setup" in driver.page_source

# -------------------------
# TC-03: Invalid Age
# -------------------------
def test_TC03_invalid_age(driver):
    driver.find_element(By.ID, "age").send_keys("-5")
    driver.find_element(By.ID, "submit").click()
    assert "Invalid age" in driver.page_source

# -------------------------
# TC-04: Empty Fields
# -------------------------
def test_TC04_empty_fields(driver):
    driver.find_element(By.ID, "submit").click()
    assert "Required field" in driver.page_source

# -------------------------
# TC-05: Health Condition
# -------------------------
def test_TC05_health_condition(driver):
    driver.find_element(By.ID, "diabetes").click()
    driver.find_element(By.ID, "save").click()
    assert "Diabetes" in driver.page_source

# -------------------------
# TC-06: Allergy Selection
# -------------------------
def test_TC06_allergy_selection(driver):
    driver.find_element(By.ID, "nuts").click()
    driver.find_element(By.ID, "save").click()
    assert "Nuts" in driver.page_source

# -------------------------
# TC-07: Complete Profile
# -------------------------
def test_TC07_profile_submit(driver):
    driver.find_element(By.ID, "complete_profile").click()
    assert "Dashboard" in driver.page_source

# -------------------------
# TC-08: Weight Loss Goal
# -------------------------
def test_TC08_weight_loss_goal(driver):
    driver.find_element(By.ID, "weight_loss").click()
    driver.find_element(By.ID, "save_goal").click()
    assert "Goal saved" in driver.page_source

# -------------------------
# TC-09: Weight Gain Goal
# -------------------------
def test_TC09_weight_gain_goal(driver):
    driver.find_element(By.ID, "weight_gain").click()
    driver.find_element(By.ID, "save_goal").click()
    assert "Goal saved" in driver.page_source

# -------------------------
# TC-10: Unrealistic Target Weight
# -------------------------
def test_TC10_unrealistic_weight(driver):
    driver.find_element(By.ID, "target_weight").send_keys("20")
    driver.find_element(By.ID, "save_goal").click()
    assert "Safety warning" in driver.page_source

# -------------------------
# TC-11: No Goal Selected
# -------------------------
def test_TC11_no_goal_selected(driver):
    driver.find_element(By.ID, "save_goal").click()
    assert "Select a goal" in driver.page_source

# -------------------------
# TC-12: Metrics Display
# -------------------------
def test_TC12_metrics_display(driver):
    assert driver.find_element(By.ID, "bmi").is_displayed()
    assert driver.find_element(By.ID, "bmr").is_displayed()
    assert driver.find_element(By.ID, "tdee").is_displayed()

# -------------------------
# TC-13: Health Status Label
# -------------------------
def test_TC13_health_status(driver):
    assert driver.find_element(By.ID, "health_status").is_displayed()

# -------------------------
# TC-14: Generate Meal Plan
# -------------------------
def test_TC14_generate_meal_plan(driver):
    driver.find_element(By.ID, "generate_meal").click()
    time.sleep(2)
    assert "Breakfast" in driver.page_source

# -------------------------
# TC-15: Nutrition Breakdown
# -------------------------
def test_TC15_nutrition_breakdown(driver):
    assert "Calories" in driver.page_source
    assert "Protein" in driver.page_source

# -------------------------
# TC-16: Condition-Aware Meal
# -------------------------
def test_TC16_condition_aware_meal(driver):
    assert "Diabetes" in driver.page_source

# -------------------------
# TC-17: Upload Food Image
# -------------------------
def test_TC17_food_image_upload(driver):
    driver.find_element(By.ID, "meal_image").send_keys("C:/images/food.jpg")
    assert "Image uploaded" in driver.page_source

# -------------------------
# TC-18: Non-Food Image
# -------------------------
def test_TC18_non_food_image(driver):
    driver.find_element(By.ID, "meal_image").send_keys("C:/images/random.jpg")
    assert "No food detected" in driver.page_source

# -------------------------
# TC-19: Calorie Estimation
# -------------------------
def test_TC19_calorie_estimation(driver):
    driver.find_element(By.ID, "submit_image").click()
    assert "Estimated Calories" in driver.page_source

# -------------------------
# TC-21: Submit Without Image
# -------------------------
def test_TC21_submit_without_image(driver):
    driver.find_element(By.ID, "submit_image").click()
    assert "Macros not calculated" in driver.page_source

# -------------------------
# TC-22: Open Workout Screen
# -------------------------
def test_TC22_workout_screen(driver):
    driver.find_element(By.ID, "workout").click()
    assert "Workout" in driver.page_source

# -------------------------
# TC-28: Open Chatbot
# -------------------------
def test_TC28_chatbot_open(driver):
    driver.find_element(By.ID, "chatbot").click()
    assert "Ask me" in driver.page_source

# -------------------------
# TC-29: Chatbot Response
# -------------------------
def test_TC29_chatbot_response(driver):
    driver.find_element(By.ID, "chat_input").send_keys("Suggest diet")
    driver.find_element(By.ID, "send").click()
    assert "calories" in driver.page_source

# -------------------------
# TC-30: Empty Chat
# -------------------------
def test_TC30_empty_chat(driver):
    driver.find_element(By.ID, "send").click()
    assert "Enter a message" in driver.page_source

# -------------------------
# TC-31: Dashboard
# -------------------------
def test_TC31_dashboard(driver):
    assert "Progress" in driver.page_source

# -------------------------
# TC-32: Navigation
# -------------------------
def test_TC32_navigation(driver):
    driver.find_element(By.ID, "meal_tab").click()
    driver.find_element(By.ID, "dashboard_tab").click()
    assert True

# -------------------------
# TC-33: Stability (Refresh)
# -------------------------
def test_TC33_page_refresh(driver):
    driver.refresh()
    assert "Dashboard" in driver.page_source

# -------------------------
# TC-34: Responsiveness
# -------------------------
def test_TC34_responsive(driver):
    driver.set_window_size(375, 812)
    assert driver.find_element(By.ID, "menu").is_displayed()

# -------------------------
# TC-35: POC Completion
# -------------------------
def test_TC35_poc_completion(driver):
    assert "Personalized Plan" in driver.page_source
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# -------------------------
# Fixture: Browser Setup
# -------------------------
@pytest.fixture
def driver():
    driver = webdriver.Chrome()  # Make sure ChromeDriver is installed
    driver.maximize_window()
    driver.get("http://localhost:3000")  # Change URL to your app
    yield driver
    driver.quit()

# -------------------------
# TC-01: App Launch
# -------------------------
def test_TC01_app_launch(driver):
    assert "Health App" in driver.title

# -------------------------
# TC-02: Valid Registration
# -------------------------
def test_TC02_valid_registration(driver):
    driver.find_element(By.ID, "name").send_keys("Hania")
    driver.find_element(By.ID, "age").send_keys("22")
    driver.find_element(By.ID, "gender").send_keys("Female")
    driver.find_element(By.ID, "submit").click()
    assert "Profile Setup" in driver.page_source

# -------------------------
# TC-03: Invalid Age
# -------------------------
def test_TC03_invalid_age(driver):
    driver.find_element(By.ID, "age").send_keys("-5")
    driver.find_element(By.ID, "submit").click()
    assert "Invalid age" in driver.page_source

# -------------------------
# TC-04: Empty Fields
# -------------------------
def test_TC04_empty_fields(driver):
    driver.find_element(By.ID, "submit").click()
    assert "Required field" in driver.page_source

# -------------------------
# TC-05: Health Condition
# -------------------------
def test_TC05_health_condition(driver):
    driver.find_element(By.ID, "diabetes").click()
    driver.find_element(By.ID, "save").click()
    assert "Diabetes" in driver.page_source

# -------------------------
# TC-06: Allergy Selection
# -------------------------
def test_TC06_allergy_selection(driver):
    driver.find_element(By.ID, "nuts").click()
    driver.find_element(By.ID, "save").click()
    assert "Nuts" in driver.page_source

# -------------------------
# TC-07: Complete Profile
# -------------------------
def test_TC07_profile_submit(driver):
    driver.find_element(By.ID, "complete_profile").click()
    assert "Dashboard" in driver.page_source

# -------------------------
# TC-08: Weight Loss Goal
# -------------------------
def test_TC08_weight_loss_goal(driver):
    driver.find_element(By.ID, "weight_loss").click()
    driver.find_element(By.ID, "save_goal").click()
    assert "Goal saved" in driver.page_source

# -------------------------
# TC-09: Weight Gain Goal
# -------------------------
def test_TC09_weight_gain_goal(driver):
    driver.find_element(By.ID, "weight_gain").click()
    driver.find_element(By.ID, "save_goal").click()
    assert "Goal saved" in driver.page_source

# -------------------------
# TC-10: Unrealistic Target Weight
# -------------------------
def test_TC10_unrealistic_weight(driver):
    driver.find_element(By.ID, "target_weight").send_keys("20")
    driver.find_element(By.ID, "save_goal").click()
    assert "Safety warning" in driver.page_source

# -------------------------
# TC-11: No Goal Selected
# -------------------------
def test_TC11_no_goal_selected(driver):
    driver.find_element(By.ID, "save_goal").click()
    assert "Select a goal" in driver.page_source

# -------------------------
# TC-12: Metrics Display
# -------------------------
def test_TC12_metrics_display(driver):
    assert driver.find_element(By.ID, "bmi").is_displayed()
    assert driver.find_element(By.ID, "bmr").is_displayed()
    assert driver.find_element(By.ID, "tdee").is_displayed()

# -------------------------
# TC-13: Health Status Label
# -------------------------
def test_TC13_health_status(driver):
    assert driver.find_element(By.ID, "health_status").is_displayed()

# -------------------------
# TC-14: Generate Meal Plan
# -------------------------
def test_TC14_generate_meal_plan(driver):
    driver.find_element(By.ID, "generate_meal").click()
    time.sleep(2)
    assert "Breakfast" in driver.page_source

# -------------------------
# TC-15: Nutrition Breakdown
# -------------------------
def test_TC15_nutrition_breakdown(driver):
    assert "Calories" in driver.page_source
    assert "Protein" in driver.page_source

# -------------------------
# TC-16: Condition-Aware Meal
# -------------------------
def test_TC16_condition_aware_meal(driver):
    assert "Diabetes" in driver.page_source

# -------------------------
# TC-17: Upload Food Image
# -------------------------
def test_TC17_food_image_upload(driver):
    driver.find_element(By.ID, "meal_image").send_keys("C:/images/food.jpg")
    assert "Image uploaded" in driver.page_source

# -------------------------
# TC-18: Non-Food Image
# -------------------------
def test_TC18_non_food_image(driver):
    driver.find_element(By.ID, "meal_image").send_keys("C:/images/random.jpg")
    assert "No food detected" in driver.page_source

# -------------------------
# TC-19: Calorie Estimation
# -------------------------
def test_TC19_calorie_estimation(driver):
    driver.find_element(By.ID, "submit_image").click()
    assert "Estimated Calories" in driver.page_source

# -------------------------
# TC-21: Submit Without Image
# -------------------------
def test_TC21_submit_without_image(driver):
    driver.find_element(By.ID, "submit_image").click()
    assert "Macros not calculated" in driver.page_source

# -------------------------
# TC-22: Open Workout Screen
# -------------------------
def test_TC22_workout_screen(driver):
    driver.find_element(By.ID, "workout").click()
    assert "Workout" in driver.page_source

# -------------------------
# TC-28: Open Chatbot
# -------------------------
def test_TC28_chatbot_open(driver):
    driver.find_element(By.ID, "chatbot").click()
    assert "Ask me" in driver.page_source

# -------------------------
# TC-29: Chatbot Response
# -------------------------
def test_TC29_chatbot_response(driver):
    driver.find_element(By.ID, "chat_input").send_keys("Suggest diet")
    driver.find_element(By.ID, "send").click()
    assert "calories" in driver.page_source

# -------------------------
# TC-30: Empty Chat
# -------------------------
def test_TC30_empty_chat(driver):
    driver.find_element(By.ID, "send").click()
    assert "Enter a message" in driver.page_source

# -------------------------
# TC-31: Dashboard
# -------------------------
def test_TC31_dashboard(driver):
    assert "Progress" in driver.page_source

# -------------------------
# TC-32: Navigation
# -------------------------
def test_TC32_navigation(driver):
    driver.find_element(By.ID, "meal_tab").click()
    driver.find_element(By.ID, "dashboard_tab").click()
    assert True

# -------------------------
# TC-33: Stability (Refresh)
# -------------------------
def test_TC33_page_refresh(driver):
    driver.refresh()
    assert "Dashboard" in driver.page_source

# -------------------------
# TC-34: Responsiveness
# -------------------------
def test_TC34_responsive(driver):
    driver.set_window_size(375, 812)
    assert driver.find_element(By.ID, "menu").is_displayed()

# -------------------------
# TC-35: POC Completion
# -------------------------
def test_TC35_poc_completion(driver):
    assert "Personalized Plan" in driver.page_source
    
