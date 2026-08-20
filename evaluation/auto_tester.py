import re
import subprocess
import os


class AutoTester:
    def __init__(self, temp_filename="test_temp_eval.py"):
        self.temp_filename = temp_filename

    def _extract_code(self, model_response):
        """Extracts Python code block from LLM markdown response."""
        # Search for code blocks starting with ```python and ending with ```
        pattern = r"```python(.*?)```"
        match = re.search(pattern, model_response, re.DOTALL)

        if match:
            return match.group(1).strip()

        # If the model output was truncated with no closing backticks
        pattern_incomplete = r"```python(.*)"
        match_incomplete = re.search(pattern_incomplete, model_response, re.DOTALL)
        if match_incomplete:
            return match_incomplete.group(1).strip()

        # Fallback: if the model omitted the 'python' identifier, extract generic code blocks
        pattern_fallback = r"```(.*?)```"
        match_fallback = re.search(pattern_fallback, model_response, re.DOTALL)
        if match_fallback:
            return match_fallback.group(1).strip()

        # If no code block is found, return the raw response (risk of syntax error)
        return model_response.strip()

    def run_test(self, source_code, model_response):
        """Executes the source code and generated unit tests together in a single file."""
        test_code = self._extract_code(model_response)

        # Concatenate source code and test code into a single file to resolve dependencies
        # In production, source code should be placed in a separate module;
        # however, for evaluation simplicity, it is prepended here.
        combined_code = f"{source_code}\n\n# --- GENERATED TEST BELOW ---\n{test_code}"

        # Save to temporary file
        with open(self.temp_filename, "w", encoding="utf-8") as f:
            f.write(combined_code)

        try:
            # Run pytest quietly
            result = subprocess.run(
                ["pytest", self.temp_filename, "-q", "--disable-warnings"],
                capture_output=True,
                text=True,
                timeout=10,  # Prevent test from hanging indefinitely in case of infinite loops
            )

            # pytest returns exit code 0 if all tests pass
            if result.returncode == 0:
                return True, "Passed"
            else:
                return (
                    False,
                    f"{result.stdout}\n\n--- EXECUTED CODE ---\n{combined_code}",
                )

        except subprocess.TimeoutExpired:
            return (
                False,
                f"Timeout: Unit test execution timed out.\n\n--- CODE ---\n{combined_code}",
            )
        except Exception as e:
            return False, f"System Error: {str(e)}\n\n--- CODE ---\n{combined_code}"
        finally:
            # Clean up temporary file
            if os.path.exists(self.temp_filename):
                os.remove(self.temp_filename)
