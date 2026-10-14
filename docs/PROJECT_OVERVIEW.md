# Project overview

SignBridge is a real-time, single-hand gesture-to-phrase demonstration. OpenCV captures and mirrors webcam frames, MediaPipe Tasks Gesture Recognizer returns hand landmarks and a built-in category, an optional local Random Forest predicts user-trained static poses, and a temporal rule detects a deliberate open-palm lateral wave. A stability/cooldown gate accepts configured phrases, and a separate sentence manager tracks the sentence/history. A worker thread automatically speaks each accepted phrase; S can queue the complete sentence.

The interface uses an OpenCV resizable window and HUD to keep installation small. `config.json` groups built-in, custom static, and motion vocabulary and defines thresholds. `collect_gesture_data.py` saves only normalized landmark samples; `train_gesture_model.py` validates the CSV, evaluates a held-out split, and writes a local joblib classifier. Custom recognition is absent until that user training step succeeds. The pretrained MediaPipe task model is downloaded from the configured HTTPS URL the first time the recognizer launches.

This is not a complete sign-language translator. Real translation needs a defined language and vocabulary, suitable datasets, temporal modeling for signs that depend on motion, and validation with sign-language users or qualified interpreters.
