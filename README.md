# SignBridge: Real-Time Hand Gesture Communication System

SignBridge is a webcam-based computer-vision demo that recognizes hand gestures, maps accepted predictions to configurable phrases, builds a sentence, and speaks each accepted phrase aloud using offline text-to-speech. Press S to speak the complete sentence again.

> **Prototype limitation:** This is a limited gesture-to-phrase prototype, not a complete sign-language translator. Its demonstration mappings are not universal or authoritative translations.

## Problem and features

The project demonstrates how camera-based hand tracking can offer an alternate way to create short spoken or written phrases. It provides a mirrored live feed, hand landmarks, gesture/confidence feedback, stable prediction and duplicate suppression, a sentence and history, undo/clear, requested speech, and keyboard controls.

## Technology and architecture

- Python 3.13 64-bit tested in the current workspace; Python 3.11 64-bit is also supported; Windows 10/11 target.
- MediaPipe Tasks Gesture Recognizer, OpenCV, NumPy, pyttsx3, scikit-learn, and joblib.
- `main.py` coordinates the loop; `src/camera.py` opens the camera; `src/gesture_recognition.py` downloads/loads the Tasks model and draws landmarks; `src/sentence_manager.py` handles phrase state; `src/speech.py` serializes non-blocking speech; `src/ui.py` renders the HUD; `src/config.py` validates configuration.

Data flow: mirrored camera frame → RGB image → MediaPipe Tasks video recognizer → category/confidence/landmarks → threshold and stable-frame gate → phrase and sentence manager → HUD and optional speech worker.

## Requirements and setup

Use 64-bit Python 3.13 or 3.11. The current pins use MediaPipe 1.1.0, which is available for Python 3.13 and includes the modern Tasks API. Check your active version:

```powershell
python --version
```

Recommended quick launch: double-click `run_windows.bat`. It creates `.venv`, installs `requirements.txt`, and launches the app. It does not install Python or change system settings.

Manual PowerShell setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

If PowerShell blocks activation, use a temporary process-only policy in that PowerShell window: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`. It does not persist after the window closes. You may also launch `.venv\Scripts\python.exe main.py` without activation.

If you specifically use Python 3.11, replace the first command with `py -3.11 -m venv .venv`. If `.venv` already exists from another Python version or dependency installation failed partway, remove it and recreate it before reinstalling: `Remove-Item -Recurse -Force .venv`.

The MediaPipe model requires internet on first run unless it is already present at `models/gesture_recognizer.task`. Only the HTTPS URL configured in `config.json` is fetched; the model is reused on later runs. Camera access is needed for the interactive app. To validate configuration/imports without opening a camera, run `python main.py --check`.

## Use and controls

Show one hand clearly in the mirrored video and hold a gesture steady. The category, confidence, mapped phrase, approximate FPS, current sentence, recent history, camera, recognition, and speech status are shown in the window.

| Key | Action |
|---|---|
| Q or Esc | Exit and release resources |
| C | Clear sentence and history |
| S | Speak the complete current sentence |
| Backspace | Undo the most recent phrase |
| H | Show or hide help |
| L | Show or hide landmarks |

## Recognition modes and vocabulary

The UI labels every prediction as **Built-in**, **Custom static**, or **Motion**. Built-in and motion phrases work without a custom classifier. Custom static definitions are configured but remain unavailable until you collect samples and train a classifier. A configured phrase is not evidence that the physical pose has been learned.

| MediaPipe category | Example phrase |
|---|---|
| Open_Palm | HELLO |
| Closed_Fist | STOP |
| Thumb_Up | YES |
| Thumb_Down | NO |
| Victory | PEACE |
| Pointing_Up | I HAVE A QUESTION |
| ILoveYou | I LOVE YOU |
| Open_Palm side-to-side wave | GOODBYE (motion rule) |

These are demonstration mappings, not universal sign-language meanings. `config.json` groups `builtin_gestures`, `custom_static_gestures`, and `motion_gestures`, with unique output phrases. Custom pose hints are suggested user-defined demo poses; they are not asserted to be standard signs. Other targets in the config are PLEASE, THANK YOU, SORRY, HELP, WAIT, FINISHED, I DON'T UNDERSTAND, REPEAT PLEASE, LEFT, RIGHT, UP, DOWN, NEXT, GO BACK, CONFIRM, CANCEL, SELECT, SMALL, THREE, FOUR, GOOD LUCK, RELAX, OK, WATER, FOOD, REST, HAPPY, SAD, PAIN, FRIEND, and FAMILY. They require your own labeled data and a successfully trained model before the app recognizes them.

The static feature extractor subtracts the wrist and normalizes by hand scale; it preserves image orientation so directional classes remain distinct. The Open_Palm wave detector smooths wrist motion, scales lateral travel by palm width, requires multiple reversals within a time window, and waits for a finishing pause. It does not implement forward/back depth motions: ordinary monocular landmarks are not a dependable depth sensor. Tune `motion` thresholds only after testing with your camera/users. Camera mirror orientation changes left/right interpretation.

Every accepted phrase is spoken automatically by default. Set `speak_each_phrase` to `false` in `config.json` to return to visual-only phrase entry; **S** still queues the full current sentence for speech. Playback runs on one background worker, so phrases are serialized and video processing stays responsive. If speech is unavailable, check the speech status shown in the window and the audio troubleshooting notes.

### Collect and train custom poses

Choose one pose per custom class and keep its meaning/shape consistent. The configured `pose_hint` is a prompt; adjust the actual pose to be comfortable and visually distinct from every other class. Collect varied examples across positions, distances, lighting, and users. Aim for **100–300 samples per phrase class**, with reasonably balanced class counts. Also collect **100–300 `__OTHER__` rejection samples**, including the seven built-in poses and miscellaneous non-command poses; the trainer refuses fewer than 20 samples for any class and requires at least two phrase classes plus `__OTHER__`. This rejection class helps the custom classifier leave built-in/unknown poses alone. Its validation split is held out by sample, so multiple samples from one recording/session can make metrics optimistic.

```powershell
python collect_gesture_data.py
# Or choose a class explicitly:
python collect_gesture_data.py --label PLEASE
```

In the collector, press **Space** to save one detected-hand landmark vector; press **Q/Esc** to finish. It writes normalized coordinates and labels to `data/gesture_samples.csv`; it does not save video. Run the collector separately for each class, including `python collect_gesture_data.py --label __OTHER__`. The dataset is your local user data and is intentionally not included in the project ZIP.

```powershell
python train_gesture_model.py
```

The trainer validates CSV columns/labels, makes a stratified 75/25 train/validation split, trains a class-balanced Random Forest, prints a per-class precision/recall/F1 report and confusion matrix, then saves `models/custom_gesture_model.joblib`. The app loads that local artifact when present; no custom model is downloaded. Review confusion-matrix off-diagonal errors and especially per-class recall. If two classes are confused, collect more varied samples or redesign the poses to be more distinct. Validation accuracy alone does not establish performance with new people or environments. The model artifact uses joblib/pickle: only load one produced by this project or otherwise trusted by you.

### Adjust recognition

In `config.json`, `confidence_thresholds.builtin` and `.custom` are in the 0–1 range; higher values reject more uncertain predictions. `stability_frames` is the consecutive-frame gate. `cooldown_seconds` suppresses repeated identical phrases; an intervening phrase permits a deliberate repeat. The wave rule has a separate time window, lateral amplitude in palm widths, reversal count, ending pause, cooldown, and net-drift limit under `motion`. More conservative thresholds reduce accidental triggers but require a clearer gesture.

## Model details

The default model URL is the MediaPipe Gesture Recognizer float16 task model. The app saves it under `models/`, enforces HTTPS, uses a temporary file during download, and limits downloads to 100 MB. Keep `models/` out of source control; the model can be fetched when needed.

## Tests

Run unit tests without a webcam:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q main.py src tests collect_gesture_data.py train_gesture_model.py
```

The tests cover phrase addition, unknown gestures, duplicate/cooldown behavior, clear/undo, confidence, stability, mapping and invalid configuration, scale normalization, wave thresholds, missing model fallback, and local model loading. `python main.py --check` checks config and component imports without opening a camera.

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for camera permission/busy state, model download, recognition confidence, PowerShell, audio, and configuration help. See [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) for a 90-second presentation and [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) for the component summary.

## Limitations, ethics, and accessibility

The seven configured built-in category mappings and the synthetic unit tests have been checked in this development environment. A no-hand frame was sent through MediaPipe Tasks to validate the installed API. Gestures have not been evaluated against a live webcam here; custom classes have not been trained from real samples, and the wave rule has not been calibrated with users/camera footage. The pretrained recognizer is sensitive to angle, lighting, background, hand size, and individual variation. A phrase mapping does not capture grammar, facial expression, two-handed signs, motion, regional variation, or sign-language context. Real sign-language translation needs a defined language/vocabulary, appropriate datasets, temporal recognition where needed, and validation with sign-language users or qualified interpreters. This is not a certified accessibility or emergency communication system. Camera frames are processed locally and not saved by the app or collector; the MediaPipe task model is downloaded from the configured URL on first run.

## Future improvements

Potential next steps include selectable camera devices, configurable phrase editing in the UI, a richer temporal model trained and validated for a defined sign language, accessibility-focused user testing, and packaging after compatibility testing on supported Windows machines.
