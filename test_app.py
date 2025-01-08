import time  # noqa: D100

from selenium import webdriver  # type: ignore  # noqa: PGH003
from selenium.webdriver.common.by import By  # type: ignore  # noqa: PGH003
from selenium.webdriver.common.keys import Keys  # type: ignore  # noqa: PGH003


def test_end_to_end_streamlit() -> None:  # noqa: D103
    # Set up the WebDriver (ensure you have ChromeDriver installed)
    driver = webdriver.Chrome()

    # Open the Streamlit app
    driver.get("http://localhost:3000")  # Streamlit default localhost port

    input_field = driver.find_element(By.XPATH, "//textarea")
    input_field.send_keys("Fever, headache, cough")
    input_field.send_keys(Keys.RETURN)  # Simulating hitting 'Enter"

    # Wait for the prediction output
    time.sleep(8)

    # Check for predicted disease in the UI
    try:
        predicted_disease = driver.find_element(
            By.XPATH, '//*[contains(text(), "Predicted Disease")]',
        ).text
        predicted_disease = predicted_disease.split(":")[
            1
        ].strip()  # Extract the disease name

        # Validate the disease prediction
        assert (  # noqa: S101
            predicted_disease == "Flu"
        )

        # Check for prescription / medical advice in the UI
        prescription_advice = driver.find_element(
            By.XPATH, '//*[contains(text(), "Prescription / Medical Advice")]',
        ).text
        prescription_advice = prescription_advice.split(":")[
            1
        ].strip()  # Extract prescription advice

        # Validate prescription/advice message
        assert (  # noqa: S101
            prescription_advice == "Rest, fluids, and over-the-counter medications"
        )  # Adjust based on your action data

        print("Test Passed: Disease prediction and prescription displayed correctly.")  # noqa: T201

    except AssertionError as e:
        print("Test Failed: ", e)  # noqa: T201
    except Exception as e:  # noqa: BLE001
        print("Error during test execution: ", e)  # noqa: T201

    # Close the browser window
    driver.quit()


if __name__ == "__main__":
    # Run the test
    test_end_to_end_streamlit()
