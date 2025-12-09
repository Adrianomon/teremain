 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/theremin_magico.py b/theremin_magico.py
new file mode 100644
index 0000000000000000000000000000000000000000..78fcdf8d263cb711a1169db50854eea1a8b8995a
--- /dev/null
+++ b/theremin_magico.py
@@ -0,0 +1,241 @@
+"""Theremín mágico – versión expresiva.
+
+Control gestual con manos: la derecha define tono (eje X) y la izquierda
+abre el volumen y la expresividad. Incluye guía de notas, suavizado de
+frecuencia, portamento y un motor de audio más musical.
+"""
+
+import math
+from dataclasses import dataclass
+from typing import Tuple
+
+import cv2
+import mediapipe as mp
+import numpy as np
+
+from theremin_engine import ThereminAudioEngine
+
+
+# ============================
+# CONFIGURACIÓN
+# ============================
+
+
+@dataclass
+class ThereminConfig:
+    freq_min: float = 440.0  # A4
+    freq_max: float = 7040.0  # A8
+    snap_cents: float = 16.0  # Cuánto se acerca al centro de nota si estás afinado
+    vibrato_depth_hz: float = 4.2
+    vibrato_rate_hz: float = 5.7
+    portamento_ms: float = 26.0
+    attack_ms: float = 12.0
+    release_ms: float = 95.0
+    noise_level: float = 0.003
+
+
+config = ThereminConfig()
+OCTAVE_RANGE = math.log2(config.freq_max / config.freq_min)
+NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
+
+ENGINE = ThereminAudioEngine(
+    portamento_ms=config.portamento_ms,
+    attack_ms=config.attack_ms,
+    release_ms=config.release_ms,
+    vibrato_hz=config.vibrato_rate_hz,
+    vibrato_depth_hz=config.vibrato_depth_hz,
+    noise_level=config.noise_level,
+)
+
+
+# ============================
+# UTILIDADES MUSICALES
+# ============================
+
+def freq_to_note(freq: float) -> Tuple[str, int, float]:
+    if freq <= 0:
+        return "-", 0, 0.0
+    semis = round(12 * math.log2(freq / 440.0))
+    freq_center = 440.0 * (2 ** (semitones_to_ratio(semi=semis)))
+    midi = 69 + semis
+    name = NOTE_NAMES[midi % 12]
+    octave = (midi // 12) - 1
+    return name, octave, freq_center
+
+
+def semitones_to_ratio(semi: float) -> float:
+    return semi / 12.0
+
+
+def apply_note_lock(freq: float) -> float:
+    """Acerca la frecuencia al centro de la nota cuando ya estás cerca."""
+    _, _, freq_center = freq_to_note(freq)
+    if freq_center <= 0:
+        return freq
+    cents = 1200 * math.log2(freq / freq_center)
+    if abs(cents) < config.snap_cents:
+        blend = 1 - abs(cents) / config.snap_cents
+        freq = freq_center * blend + freq * (1 - blend * 0.35)
+    return freq
+
+
+# ============================
+# DIBUJO GUIADO
+# ============================
+
+def draw_guide(frame, freq_now: float, volume: float) -> None:
+    h, w, _ = frame.shape
+    margin = 40
+    bar_y = h - 70
+    bar_width = w - 2 * margin
+
+    overlay = frame.copy()
+    cv2.rectangle(overlay, (0, bar_y - 50), (w, h), (12, 12, 12), -1)
+    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
+
+    cv2.line(frame, (margin, bar_y), (w - margin, bar_y), (180, 180, 180), 1)
+
+    log_min = math.log2(config.freq_min)
+    log_max = math.log2(config.freq_max)
+
+    note_name, octave, freq_center = freq_to_note(freq_now)
+    freq_center = np.clip(freq_center, config.freq_min, config.freq_max)
+
+    semitone = 0
+    while True:
+        freq_note = 440.0 * (2 ** semitones_to_ratio(semitone))
+        if freq_note < config.freq_min - 1:
+            semitone += 1
+            continue
+        if freq_note > config.freq_max + 1:
+            break
+
+        log_f = math.log2(freq_note)
+        x = int(margin + (log_f - log_min) / (log_max - log_min) * bar_width)
+
+        n_name, n_oct, _ = freq_to_note(freq_note)
+
+        is_current = abs(freq_note - freq_center) < 1.0
+        color = (60, 235, 255) if is_current else (200, 200, 200)
+
+        cv2.circle(frame, (x, bar_y), 5 if is_current else 2, color, -1)
+        if semitone % 2 == 0 or is_current:
+            cv2.putText(frame, f"{n_name}{n_oct}", (x - 15, bar_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
+
+        semitone += 1
+
+    cv2.putText(frame, f"{freq_now:.1f} Hz", (margin, bar_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (130, 255, 130), 2)
+    cv2.putText(frame, f"{note_name}{octave}", (margin + 170, bar_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 235, 255), 2)
+
+    # Barra de volumen
+    vol_height = int(120 * volume)
+    cv2.rectangle(frame, (w - margin - 25, bar_y - 10 - vol_height), (w - margin - 5, bar_y - 10), (120, 255, 140), -1)
+    cv2.rectangle(frame, (w - margin - 25, bar_y - 130), (w - margin - 5, bar_y - 10), (200, 200, 200), 1)
+
+
+# ============================
+# CONTROL DE MANOS
+# ============================
+
+def extract_pitch_frequency(hand_landmark, width: int) -> float:
+    x_norm = np.clip(hand_landmark.landmark[8].x, 0.0, 1.0)
+    freq = config.freq_min * (2 ** (OCTAVE_RANGE * x_norm))
+    return apply_note_lock(freq)
+
+
+def extract_volume(hand_landmark) -> float:
+    thumb = hand_landmark.landmark[4]
+    index = hand_landmark.landmark[8]
+    d = math.hypot(thumb.x - index.x, thumb.y - index.y) * 5.3
+    d = np.clip(d, 0.0, 1.0)
+    return math.sqrt(d)
+
+
+# ============================
+# LOOP PRINCIPAL
+# ============================
+
+def main() -> None:
+    mp_hands = mp.solutions.hands
+    hands = mp_hands.Hands(
+        max_num_hands=2,
+        min_detection_confidence=0.72,
+        min_tracking_confidence=0.72,
+        model_complexity=1,
+    )
+
+    cap = cv2.VideoCapture(0)
+    stream = ENGINE.start()
+
+    current_volume = 0.0
+
+    try:
+        while True:
+            ret, frame = cap.read()
+            if not ret:
+                continue
+
+            frame = cv2.flip(frame, 1)
+            h, w, _ = frame.shape
+
+            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
+            results = hands.process(rgb)
+
+            left_hand = None
+            right_hand = None
+
+            if results.multi_hand_landmarks:
+                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
+                    label = handedness.classification[0].label
+                    if label == "Right":
+                        right_hand = hand_landmarks
+                    else:
+                        left_hand = hand_landmarks
+
+            if right_hand:
+                freq = extract_pitch_frequency(right_hand, w)
+                ENGINE.set_frequency(freq)
+                pt = right_hand.landmark[8]
+                cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 12, (60, 235, 255), -1)
+            else:
+                # Mantén la última frecuencia para portamento, pero sin nuevas órdenes.
+                freq = ENGINE.current_freq
+
+            if left_hand:
+                volume = extract_volume(left_hand)
+                current_volume = 0.8 * current_volume + 0.2 * volume
+                ENGINE.set_volume(current_volume)
+                pt = left_hand.landmark[8]
+                cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 12, (120, 255, 140), -1)
+            else:
+                current_volume = 0.9 * current_volume
+                ENGINE.set_volume(current_volume)
+
+            draw_guide(frame, ENGINE.current_freq, current_volume)
+
+            cv2.putText(
+                frame,
+                "Derecha = tono / slides    Izquierda = volumen / color",
+                (15, 30),
+                cv2.FONT_HERSHEY_SIMPLEX,
+                0.65,
+                (235, 235, 235),
+                2,
+            )
+            cv2.putText(frame, "Escala logarítmica con snap suave a nota", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
+
+            cv2.imshow("THEREMIN MÁGICO", frame)
+
+            if cv2.waitKey(1) & 0xFF == ord("q"):
+                break
+
+    finally:
+        if stream is not None:
+            stream.stop()
+            stream.close()
+        cap.release()
+        cv2.destroyAllWindows()
+
+
+if __name__ == "__main__":
+    main()
 
EOF
)
