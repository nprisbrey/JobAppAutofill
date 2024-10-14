import json
import requests
import subprocess
import re
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException


class JobApplicationAutofill:
    def __init__(self):
        self.driver = None
        self.context_file = ""
        self.context = ""
        self.current_application_context = []
        self.answer_history = {}
        self.ollama_model = "llama3.2"  # Default model
        self.config = self.load_config()

    def load_config(self):
        try:
            with open("config.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"context_file": "", "ollama_model": "mistral", "browser": ""}

    def save_config(self):
        with open("config.json", "w") as f:
            json.dump(
                {
                    "context_file": self.context_file,
                    "ollama_model": self.ollama_model,
                    "browser": self.config.get("browser", ""),
                },
                f,
            )

    def setup_browser(self):
        if not self.config.get("browser"):
            self.choose_browser()

        if self.config["browser"].lower() == "firefox":
            options = FirefoxOptions()
            options.add_argument("--start-maximized")
            self.driver = webdriver.Firefox(options=options)
        else:  # Default to Chrome
            options = ChromeOptions()
            options.add_argument("--start-maximized")
            self.driver = webdriver.Chrome(options=options)

        # Navigate to LinkedIn.com
        self.driver.get("https://www.linkedin.com")
        print("Navigated to LinkedIn.com")

    def choose_browser(self):
        while True:
            choice = input("Choose your browser (firefox/chrome): ").lower()
            if choice in ["firefox", "chrome"]:
                self.config["browser"] = choice
                self.save_config()
                break
            else:
                print("Invalid choice. Please enter 'firefox' or 'chrome'.")

    def load_context_file(self):
        if self.config["context_file"]:
            file_path = self.config["context_file"]
        else:
            file_path = input("Enter the path to your context file: ")
        try:
            with open(file_path, "r") as file:
                self.context = file.read()
            self.context_file = file_path
            print("Context file loaded successfully.")
            self.save_config()
        except FileNotFoundError:
            print("File not found. Please check the path and try again.")

    def new_application(self):
        self.current_application_context = []
        print("Starting a new application. Previous context cleared.")

    def switch_to_latest_tab(self):
        if self.driver.window_handles:
            self.driver.switch_to.window(self.driver.window_handles[-1])

    def get_form_html(self):
        try:
            self.switch_to_latest_tab()
            current_element = self.driver.switch_to.active_element
            form = current_element.find_element(By.XPATH, "ancestor::form")
            return form.get_attribute("outerHTML")
        except NoSuchElementException:
            print("No form found containing the current element. Using body instead.")
            return self.driver.find_element(By.TAG_NAME, "body").get_attribute(
                "outerHTML"
            )
        except Exception as e:
            print(f"Error getting form HTML: {e}")
            return None

    def extract_input_elements(self, html):
        pattern = r"<(?:input|textarea|select)[^>]*>"
        return re.findall(pattern, html, re.IGNORECASE)

    def query_ollama(self, prompt, element=None):
        url = "http://localhost:11434/api/generate"
        data = {"model": self.ollama_model, "prompt": prompt}
        try:
            with requests.post(url, json=data, stream=True) as response:
                response.raise_for_status()
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        json_response = json.loads(line)
                        if "response" in json_response:
                            chunk = json_response["response"]
                            full_response += chunk
                            if element:
                                element.send_keys(chunk)
                return full_response
        except requests.RequestException as e:
            print(f"Error querying Ollama: {e}")
            return "Error querying Ollama"

    def fill_all_fields(self):
        self.switch_to_latest_tab()
        form_html = self.get_form_html()
        if not form_html:
            return

        input_elements = self.extract_input_elements(form_html)

        for element_html in input_elements:
            element_type = re.search(r"<(\w+)", element_html).group(1)
            element_id = re.search(r'id=[\'"]([^\'"]*)[\'"]', element_html)
            element_id = element_id.group(1) if element_id else None
            element_name = re.search(r'name=[\'"]([^\'"]*)[\'"]', element_html)
            element_name = element_name.group(1) if element_name else None

            if not element_id and not element_name:
                continue

            try:
                if element_type == "input" and re.search(
                    r'type=[\'"]radio[\'"]', element_html
                ):
                    self.handle_radio_group(element_name)
                else:
                    if element_id:
                        selenium_element = self.driver.find_element(By.ID, element_id)
                    elif element_name:
                        selenium_element = self.driver.find_element(
                            By.NAME, element_name
                        )

                    # Check if the element is visible and enabled
                    if (
                        not selenium_element.is_displayed()
                        or not selenium_element.is_enabled()
                    ):
                        print(
                            f"Skipping hidden or disabled element: {element_id or element_name}"
                        )
                        continue

                    # Skip file inputs
                    if selenium_element.get_attribute("type") == "file":
                        print(
                            f"Skipping file upload field: {element_id or element_name}"
                        )
                        continue

                    label = self.get_field_label(selenium_element)
                    print(f"Now processing input with label '{label}'....")

                    prompt = self.create_prompt(
                        form_html, element_type, element_id or element_name, label
                    )
                    response = self.query_ollama(prompt)

                    self.fill_field(selenium_element, response, label)

            except Exception as e:
                print(f"Error processing element {element_id or element_name}: {e}")

    def handle_radio_group(self, name):
        if not name:
            return

        radio_group = self.driver.find_elements(By.NAME, name)
        if not radio_group:
            print(f"No radio buttons found for group: {name}")
            return

        # Get the label for the radio group
        group_label = self.get_field_label(radio_group[0])
        print(f"Now processing input with label '{group_label}'....")

        # Create prompt and get response
        prompt = self.create_prompt(self.get_form_html(), "radio", name, group_label)
        response = self.query_ollama(prompt)

        # Find the best matching radio button
        best_match = None
        best_match_score = float("inf")
        for radio in radio_group:
            radio_value = radio.get_attribute("value").lower()
            radio_label = self.get_field_label(radio).lower()

            score = min(
                self.levenshtein_distance(radio_value, response.lower()),
                self.levenshtein_distance(radio_label, response.lower()),
            )

            if score < best_match_score:
                best_match = radio
                best_match_score = score

        if best_match:
            best_match.click()
            print(
                f"Radio group '{group_label}' filled with: {best_match.get_attribute('value')}"
            )
        else:
            print(
                f"No suitable match found for radio group '{group_label}': {response}"
            )

    def fill_field(self, element, response, label):
        element_type = element.get_attribute("type")

        if element.tag_name == "select":
            self.handle_select(element, response, label)
        elif element_type == "checkbox":
            self.handle_checkbox(element, response, label)
        else:  # text, textarea, etc.
            self.handle_text_input(element, response, label)

    def handle_select(self, element, response, label):
        select = Select(element)
        options = [option.text.strip().lower() for option in select.options]
        response_lower = response.strip().lower()

        best_match = min(
            options, key=lambda x: self.levenshtein_distance(x, response_lower)
        )
        select.select_by_visible_text(best_match)
        print(f"Field '{label}' (select) filled with: {best_match}")

    def handle_radio(self, element, response, label):
        radio_group = self.driver.find_elements(By.NAME, element.get_attribute("name"))
        response_lower = response.strip().lower()

        for radio in radio_group:
            if radio.get_attribute("value").lower() == response_lower:
                radio.click()
                print(f"Field '{label}' (radio) filled with: {response}")
                return

        print(f"No matching radio button found for '{label}': {response}")

    def handle_checkbox(self, element, response, label):
        response_lower = response.strip().lower()
        if response_lower in ["yes", "true", "1", "on"]:
            if not element.is_selected():
                element.click()
            print(f"Field '{label}' (checkbox) checked")
        elif response_lower in ["no", "false", "0", "off"]:
            if element.is_selected():
                element.click()
            print(f"Field '{label}' (checkbox) unchecked")
        else:
            print(f"Invalid response for checkbox '{label}': {response}")

    def handle_text_input(self, element, response, label):
        element.clear()
        element.send_keys(response)
        print(f"Field '{label}' filled with: {response}")

    def levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def get_field_label(self, element):
        label_text = None

        # Method 1: Check for 'id' attribute and corresponding label
        element_id = element.get_attribute("id")
        if element_id:
            try:
                label = self.driver.find_element(
                    By.CSS_SELECTOR, f"label[for='{element_id}']"
                )
                label_text = label.text.strip()
                if label_text:
                    return label_text
            except NoSuchElementException:
                pass

        # Method 2: Check for a label that's a parent of the input
        try:
            label = element.find_element(By.XPATH, "ancestor::label")
            label_text = label.text.strip()
            if label_text:
                return label_text
        except NoSuchElementException:
            pass

        # Method 3: Look for the closest preceding label
        try:
            label = element.find_element(By.XPATH, "preceding::label[1]")
            label_text = label.text.strip()
            if label_text:
                return label_text
        except NoSuchElementException:
            pass

        # Method 4: Look for a label or div with class containing 'label' right before the input
        try:
            label = element.find_element(
                By.XPATH,
                "./preceding-sibling::*[self::label or contains(@class, 'label')][1]",
            )
            label_text = label.text.strip()
            if label_text:
                return label_text
        except NoSuchElementException:
            pass

        # Fallback methods
        label_text = (
            element.get_attribute("aria-label")
            or element.get_attribute("placeholder")
            or element.get_attribute("name")
            or "Unknown field"
        )

        return label_text

    def create_prompt(self, form_html, element_type, element_id, label):
        return (
            f"Job applicant information:\n{self.context}\n\n"
            f"Current form HTML:\n{form_html}\n\n"
            "You are an assistant to the job applicant with the information given above. You are using the information given about the job applicant to fill in a field in the form given above. "
            f"Respond with exactly what the job aplicant would say in response for the {element_type} field with ID '{element_id}' and label '{label}'. "
            "Consider the field's type and label, and the current state of the form. Note that your whole and complete response will be filled in as the input field's value, meaning only respond with exactly what the job applicant would say. Keep the answer concise and relevant to the field type and context.\n"
            f"Response to insert into {element_type} field with ID '{element_id}' and label '{label}': "
        )

    def change_answer(self, direction):
        self.switch_to_latest_tab()
        current_element = self.driver.switch_to.active_element
        if not current_element.is_displayed() or not current_element.is_enabled():
            print("Current element is not visible or enabled. Cannot change answer.")
            return

        label = self.get_field_label(current_element)
        print(f"Now processing input with label '{label}'....")
        if label in self.answer_history:
            current_value = current_element.get_attribute("value")
            current_index = (
                self.answer_history[label].index(current_value)
                if current_value in self.answer_history[label]
                else -1
            )

            if direction == "previous":
                new_index = max(0, current_index - 1)
                new_answer = self.answer_history[label][new_index]
            else:  # next
                if current_index < len(self.answer_history[label]) - 1:
                    new_index = current_index + 1
                    new_answer = self.answer_history[label][new_index]
                else:
                    # Generate a new answer
                    form_html = self.get_form_html()
                    element_id = current_element.get_attribute("id")
                    element_type = current_element.tag_name
                    prompt = self.create_prompt(
                        form_html, element_type, element_id, label
                    )
                    new_answer = self.query_ollama(prompt)
                    self.answer_history[label].append(new_answer)

            current_element.clear()
            current_element.send_keys(new_answer)
            print(f"Answer changed to: {new_answer}")
        else:
            print("No previous answers for this field.")

    def previous_answer(self):
        self.change_answer("previous")

    def next_answer(self):
        self.change_answer("next")

    def set_ollama_model(self):
        print()  # Add a new line for better readability
        new_model = input(
            f"Current Ollama model: {self.ollama_model}.\nEnter a new model name or press Enter to keep current: "
        )
        if new_model:
            self.ollama_model = new_model
            print(f"Setting Ollama model to: {self.ollama_model}")
            self.save_config()

        # Pull the Ollama model
        print(f"Pulling Ollama model: {self.ollama_model}")
        try:
            subprocess.run(["ollama", "pull", self.ollama_model], check=True)
            print(f"Successfully pulled Ollama model: {self.ollama_model}")
        except subprocess.CalledProcessError as e:
            print(f"Error pulling Ollama model: {e}")
            print("Reverting to previous model.")
            self.ollama_model = self.config["ollama_model"]
        except FileNotFoundError:
            print(
                "Error: 'ollama' command not found. Please ensure Ollama is installed and in your system PATH."
            )
            print("Reverting to previous model.")
            self.ollama_model = self.config["ollama_model"]
        except Exception as e:
            print(f"Uncaught error while pulling model with Ollama: {e}")

    def print_commands(self):
        print("\nAvailable commands:")
        print("n: Start a new application")
        print("f: Fill all visible fields in the current form")
        print("p: Use previous answer for the current visible field")
        print("x: Use next answer or generate a new one for the current visible field")
        print("q: Quit the program")
        print("h: Show this help message")

    def run(self):
        self.load_context_file()
        self.set_ollama_model()
        self.setup_browser()
        print(f"Using {self.config['browser'].capitalize()} browser.")
        self.print_commands()

        while True:
            try:
                command = input("\nEnter a command: ").lower()
                if command == "n":
                    self.new_application()
                elif command == "f":
                    self.fill_all_fields()
                elif command == "p":
                    self.previous_answer()
                elif command == "x":
                    self.next_answer()
                elif command == "h":
                    self.print_commands()
                elif command == "q":
                    print("Exiting program...")
                    break
                else:
                    print("Unknown command. Type 'h' for help.")
            except KeyboardInterrupt:
                print("\nExiting program...")
                break

        if self.driver:
            self.driver.quit()


if __name__ == "__main__":
    autofill = JobApplicationAutofill()
    autofill.run()
