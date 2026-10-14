# 90-second hackathon demo script

**0:00–0:15 — The problem.** “Many people communicate with gestures, while common digital interfaces expect speech or typing. SignBridge explores one small computer-vision interaction that could make a screen more accessible.”

**0:15–0:30 — Set expectations.** “This is a limited gesture-to-phrase prototype, not a sign-language translator. It uses MediaPipe’s built-in gesture categories and phrases we configure for this demo.”

**0:30–0:50 — Live recognition.** Show the mirrored camera and landmarks. Hold an open palm until HELLO appears; try a thumbs-up for YES. Point out category, confidence, recognition mode, and stability/cooldown behavior. GOODBYE wave and custom classes should only be demonstrated after you have calibrated/tested them locally.

**0:50–1:05 — Compose.** Add a second phrase. Show the current sentence and recent history; use Backspace to undo and C to clear.

**1:05–1:20 — Accessible output.** Press S to speak the complete sentence using local text-to-speech. Mention that speech runs separately from the video loop.

Optional custom-model segment: collect balanced examples for at least two distinct configured classes, train the Random Forest, and show its held-out validation metrics before enabling that model in a live demo. Do not imply that untrained phrases are recognized.

**1:20–1:30 — What comes next.** “A real translator needs a carefully defined sign-language vocabulary, temporal recognition where needed, appropriate data, and validation with sign-language users or qualified interpreters.”
