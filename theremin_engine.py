diff --git a/theremin_engine.py b/theremin_engine.py
new file mode 100644
index 0000000000000000000000000000000000000000..4491ac8744325ad75b080239dc8bd57e90a0ec47
--- /dev/null
+++ b/theremin_engine.py
@@ -0,0 +1,121 @@
+"""Motor de audio para el theremín mágico.
+
+Este motor es autocontenido y busca un sonido suave y expresivo:
+- Portamento configurable para transiciones fluidas.
+- Envolvente de ataque/caída para evitar clics.
+- Vibrato suave con profundidad ajustable.
+- Mezcla de fundamental, armónicos y un toque de ruido analógico.
+"""
+
+from __future__ import annotations
+
+import math
+from typing import Optional
+
+import numpy as np
+
+
+class ThereminAudioEngine:
+    """Motor de síntesis para controlar pitch y volumen en tiempo real.
+
+    Los parámetros están pensados para sonido "pro":
+    - portamento_ms controla la velocidad de deslizamiento entre notas.
+    - attack_ms y release_ms esculpen las transiciones de volumen.
+    - vibrato_hz y vibrato_depth_hz añaden vida al tono sin exagerarlo.
+    - noise_level añade un halo de ruido para un carácter más orgánico.
+    """
+
+    def __init__(
+        self,
+        *,
+        samplerate: int = 44_100,
+        portamento_ms: float = 35.0,
+        attack_ms: float = 12.0,
+        release_ms: float = 90.0,
+        vibrato_hz: float = 5.0,
+        vibrato_depth_hz: float = 2.5,
+        noise_level: float = 0.004,
+    ) -> None:
+        self.samplerate = samplerate
+        self.portamento_ms = portamento_ms
+        self.attack_ms = attack_ms
+        self.release_ms = release_ms
+        self.vibrato_hz = vibrato_hz
+        self.vibrato_depth_hz = vibrato_depth_hz
+        self.noise_level = noise_level
+
+        self.current_freq: float = 440.0
+        self.target_freq: float = 440.0
+        self.current_volume: float = 0.0
+        self.target_volume: float = 0.0
+
+        self._phase: float = 0.0
+        self._vibrato_phase: float = 0.0
+
+        self._portamento_coeff = self._ms_to_coeff(self.portamento_ms)
+        self._attack_coeff = self._ms_to_coeff(self.attack_ms)
+        self._release_coeff = self._ms_to_coeff(self.release_ms)
+
+    def _ms_to_coeff(self, ms: float) -> float:
+        return math.exp(-1.0 / (self.samplerate * (ms / 1000.0)))
+
+    def set_frequency(self, freq: float) -> None:
+        self.target_freq = max(0.0, float(freq))
+
+    def set_volume(self, volume: float) -> None:
+        self.target_volume = float(np.clip(volume, 0.0, 1.0))
+
+    # Callback compatible con sounddevice
+    def callback(self, outdata, frames: int, time, status) -> None:  # type: ignore[override]
+        # Vector de amortiguación para pitch y volumen
+        freq_series = np.empty(frames, dtype=np.float64)
+        vol_series = np.empty(frames, dtype=np.float64)
+
+        for i in range(frames):
+            self.current_freq += (self.target_freq - self.current_freq) * (1.0 - self._portamento_coeff)
+
+            coeff = self._attack_coeff if self.target_volume >= self.current_volume else self._release_coeff
+            self.current_volume += (self.target_volume - self.current_volume) * (1.0 - coeff)
+
+            freq_series[i] = self.current_freq
+            vol_series[i] = self.current_volume
+
+        # LFO de vibrato
+        vibrato_phase = self._vibrato_phase + 2 * np.pi * self.vibrato_hz * np.arange(frames) / self.samplerate
+        lfo = np.sin(vibrato_phase)
+        self._vibrato_phase = float((vibrato_phase[-1] + 2 * np.pi * self.vibrato_hz / self.samplerate) % (2 * np.pi))
+
+        freq_series = np.clip(freq_series + self.vibrato_depth_hz * lfo, 20.0, self.samplerate / 2 - 200)
+
+        phase_increment = 2 * np.pi * freq_series / self.samplerate
+        phase = self._phase + np.cumsum(phase_increment)
+        self._phase = float(phase[-1] % (2 * np.pi))
+
+        # Mezcla de fundamental + armónico + ruido
+        tone = 0.78 * np.sin(phase) + 0.18 * np.sin(2 * phase + 0.2) + 0.04 * np.sin(3 * phase + 0.5)
+        tone *= vol_series
+
+        noise = self.noise_level * np.random.randn(frames)
+        output = np.clip(tone + noise, -1.0, 1.0)
+
+        outdata[:, 0] = output.astype(np.float32)
+
+    # Helpers para quienes prefieran un contexto
+    def create_stream(self):
+        try:
+            import sounddevice as sd
+        except ModuleNotFoundError as exc:  # pragma: no cover - depende del entorno
+            raise RuntimeError("sounddevice no está instalado") from exc
+
+        return sd.OutputStream(
+            samplerate=self.samplerate,
+            channels=1,
+            callback=self.callback,
+            blocksize=0,
+            latency="low",
+        )
+
+    def start(self) -> Optional[object]:
+        stream = self.create_stream()
+        stream.start()
+        return stream
