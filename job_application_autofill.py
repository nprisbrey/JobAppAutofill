import json
import requests
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from pynput import keyboard


class JobApplicationAutofill:
    def __init__(self):
        self.driver = None
        self.context_file = ""
        self.context = ""
        self.current_application_context = []
        self.answer_history = {}
        self.ollama_model = "mistral"  # Default model
        self.config = self.load_config()
        self.keyboard_listener = None

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

    def setup_keyboard_shortcuts(self):
        def on_press(key):
            try:
                if key == keyboard.KeyCode.from_char("n") and self.check_modifiers(key):
                    self.new_application()
                elif key == keyboard.KeyCode.from_char("f") and self.check_modifiers(
                    key
                ):
                    self.fill_current_field()
                elif key == keyboard.Key.left and self.check_modifiers(key):
                    self.previous_answer()
                elif key == keyboard.Key.right and self.check_modifiers(key):
                    self.next_answer()
            except AttributeError:
                pass

        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()

    def check_modifiers(self, key):
        return (
            keyboard.Key.ctrl in self.keyboard_listener.pressed_keys
            and keyboard.Key.shift in self.keyboard_listener.pressed_keys
        )

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

    def get_field_prompt(self):
        try:
            current_element = self.driver.switch_to.active_element
            wait = WebDriverWait(self.driver, 10)
            label = wait.until(
                EC.presence_of_element_located((By.XPATH, "./preceding::label[1]"))
            )
            return label.text
        except (NoSuchElementException, TimeoutException):
            print(
                "Unable to find label for the current field. Using placeholder text if available."
            )
            try:
                placeholder = current_element.get_attribute("placeholder")
                return placeholder if placeholder else "Unknown field"
            except:
                return "Unknown field"

    def query_ollama(self, prompt):
        url = "http://localhost:11434/api/generate"
        data = {"model": self.ollama_model, "prompt": prompt}
        try:
            response = requests.post(url, json=data, timeout=30)
            response.raise_for_status()
            return response.json()["response"]
        except requests.RequestException as e:
            print(f"Error querying Ollama: {e}")
            return "Error querying Ollama"

    def fill_current_field(self):
        field_prompt = self.get_field_prompt()
        full_context = (
            f"Context from file:\n{self.context}\n\n"
            f"Previous questions and answers:\n"
            f"{'\n'.join([f'Q: {q}\nA: {a}' for q, a in self.current_application_context])}\n\n"
            f"New question: {field_prompt}\n"
            f"Please answer this question as if you were the job applicant. "
            f"Keep the answer concise and relevant to the question."
        )
        answer = self.query_ollama(full_context)

        self.current_application_context.append((field_prompt, answer))
        if field_prompt not in self.answer_history:
            self.answer_history[field_prompt] = []
        self.answer_history[field_prompt].append(answer)

        try:
            current_element = self.driver.switch_to.active_element
            current_element.clear()
            current_element.send_keys(answer)
            print(f"Field '{field_prompt}' filled with: {answer}")
        except Exception as e:
            print(f"Error filling field: {e}")

    def change_answer(self, direction):
        current_element = self.driver.switch_to.active_element
        field_prompt = self.get_field_prompt()
        if field_prompt in self.answer_history:
            current_value = current_element.get_attribute("value")
            current_index = (
                self.answer_history[field_prompt].index(current_value)
                if current_value in self.answer_history[field_prompt]
                else -1
            )
            if direction == "previous":
                new_index = max(0, current_index - 1)
            else:  # next
                new_index = min(
                    len(self.answer_history[field_prompt]) - 1, current_index + 1
                )
            new_answer = self.answer_history[field_prompt][new_index]
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
            f"Current Ollama model: {self.ollama_model}. Enter a new model name or press Enter to keep current: "
        )
        if new_model:
            self.ollama_model = new_model
            print(f"Ollama model set to: {self.ollama_model}")
            self.save_config()

    def run(self):
        self.load_context_file()
        self.set_ollama_model()
        self.setup_browser()
        self.setup_keyboard_shortcuts()
        print(f"Using {self.config['browser'].capitalize()} browser.")
        print("Program is running. Use the following shortcuts:")
        print("Ctrl+Shift+N: Start a new application")
        print("Ctrl+Shift+F: Fill the current field")
        print("Ctrl+Shift+Left Arrow: Use previous answer")
        print("Ctrl+Shift+Right Arrow: Use next answer")
        print("Press Ctrl+C to exit the program")
        try:
            # Keep the program running
            keyboard.Listener.join(self.keyboard_listener)
        except KeyboardInterrupt:
            print("Exiting program...")
        finally:
            if self.driver:
                self.driver.quit()
            if self.keyboard_listener:
                self.keyboard_listener.stop()


if __name__ == "__main__":
    autofill = JobApplicationAutofill()
    autofill.run()
