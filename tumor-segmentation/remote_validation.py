from __future__ import annotations

"""Remote validation helper for DMIAI tumor-segmentation challenge.

This module provides the `TumorSegmentationValidator` class that wraps all
Playwright set-up in a convenient API:

Example
-------
>>> from remote_validation import TumorSegmentationValidator
>>> validator = TumorSegmentationValidator(headless=False)
>>> try:
...     validator.navigate_to_site()  # ➜ log-in manually in the opened browser
...     validator.run_interactive_loop()  # queue as many attempts as you like
... finally:
...     validator.close()
"""

import json
import sys
import time
from typing import Optional

# Optional: Native beep on Windows
try:
    import winsound  # type: ignore
except ImportError:  # pragma: no cover – module only available on Windows
    winsound = None

from playwright.sync_api import Page, sync_playwright

__all__ = ["TumorSegmentationValidator"]


class TumorSegmentationValidator:
    """Interactive helper for queuing and monitoring validation attempts.

    The *browser* is launched once when the instance is created.  The user can
    then log in manually, after which individual validation attempts can be
    queued and monitored using :py:meth:`queue_validation_attempt` – or the
    convenience method :py:meth:`run_interactive_loop` which replicates the
    behaviour of the original script.
    """

    def __init__(self, *, headless: bool = False) -> None:
        self._playwright = sync_playwright().start()
        self.browser = self._playwright.chromium.launch(headless=headless)
        self.context = self.browser.new_context()
        self.page: Page = self.context.new_page()

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def navigate_to_site(self, url: str = "https://cases.dmiai.dk") -> None:
        """Open *url* and block until the user has logged in.

        Because the challenge platform relies on an external identity provider,
        the easiest solution is to let the user log in manually.  The method
        therefore navigates to *url*, prints instructions to the console and
        blocks until *Enter* is pressed.
        """

        self.page.goto(url)
        print("🔑 Please log in manually and navigate to the tumour-segmentation page in the browser window that just opened.")
        input("Press <Enter> when you are ready to start queuing validation attempts…")

    def queue_validation_attempt(self, *, poll_interval: float = 5.0, timeout: Optional[float] = None) -> Optional[float]:
        """Queue a validation attempt and wait for the resulting score.

        Parameters
        ----------
        poll_interval:
            Seconds between successive checks for the result.
        timeout:
            Maximum waiting time in *seconds*.  If *None* (default) wait
            indefinitely.

        Returns
        -------
        float | None
            The score if one could be extracted, *None* otherwise.
        """
        time.sleep(1)
        print("🚀 Queuing validation attempt…")
        self.page.click("button:has-text('Queue validation attempt')")
        # small delay to ensure the request has been registered server-side
        time.sleep(10)

        # Trigger the explicit check – this behaves the same way as pressing
        # the "Check attempt" button manually.
        self.page.click("button:has-text('Check attempt')")
        print("⏳ Waiting for result (checking every %.0f seconds)…" % poll_interval)

        time.sleep(10)
        start_time = time.time()
        while True:
            time.sleep(poll_interval)
            score = self._extract_score_from_code_block()
            if score is not None:
                print(f"✅ Validation score: {score}")
                self._ding()
                return score

            if timeout is not None and (time.time() - start_time) >= timeout:
                print("⚠️  Timeout reached without receiving a score.")
                return None

    def run_interactive_loop(self) -> None:
        """Prompt the user whether to queue more attempts in an infinite loop."""
        while True:
            user_input = (
                input("\n🟢 Ready to queue validation? Type 'y' to continue, 'q' to quit: ")
                .strip()
                .lower()
            )
            if user_input == "q":
                print("Exiting.")
                break
            if user_input != "y":
                # Ignore everything that is not an explicit "yes".
                continue

            self.queue_validation_attempt()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Shut down Playwright and close the browser."""
        try:
            self.browser.close()
        finally:
            # Always call stop() even if browser.close() raised
            self._playwright.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_score_from_code_block(self) -> Optional[float]:
        """Attempt to extract *score* from <code> blocks with class ``success``."""
        try:
            code_blocks = self.page.locator("pre > code.success")
            for i in range(code_blocks.count()):
                text = code_blocks.nth(i).inner_text()
                if '"score"' in text:
                    data = json.loads(text)
                    score = data.get("attempt", {}).get("score")
                    if score is not None:
                        return float(score)
        except Exception as exc:  # broad – want to ignore *any* scraping error
            print("⚠️  Error while parsing score:", exc)
        return None

    def _ding(self) -> None:
        """Emit an audible alert if the terminal/OS supports it."""
        if "pytest" in sys.modules:  # avoid noise during automated tests
            return
        if winsound is not None and hasattr(winsound, "Beep"):
            try:
                winsound.Beep(1000, 250)  # frequency, duration in ms
                winsound.Beep(250, 250)
                winsound.Beep(500, 250)
                winsound.Beep(750, 250)
                winsound.Beep(1000, 250)
                return
            except RuntimeError:
                # Fall back to ASCII bell below
                pass
        # Standard ASCII bell – works in many terminals
        print("\a", end="", flush=True)

    # ------------------------------------------------------------------
    # Python context-manager protocol for obvious ergonomics
    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401 (one-line docstring is fine)
        self.close()


# ----------------------------------------------------------------------
# CLI entry-point for backwards compatibility with the original script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    with TumorSegmentationValidator(headless=False) as validator:
        validator.navigate_to_site()
        validator.run_interactive_loop()
