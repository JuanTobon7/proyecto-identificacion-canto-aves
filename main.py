"""Interfaz principal del proyecto.

La app permite:
- Elegir un modelo entrenado desde /models
- Clasificar un audio WAV local
- Capturar audio en tiempo real desde el micrófono
- Visualizar el espectro en ambos modos
- Mostrar información e imagen de la especie predicha

Ejecución:
    python main.py
"""

from __future__ import annotations

import io
import json
import logging
import re
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import numpy as np
from PIL import Image, ImageTk, ImageOps  # type: ignore[import-not-found]
from scipy.io import wavfile
from tkinter import BOTH, DISABLED, END, NORMAL, LEFT, RIGHT, X, Y, StringVar, Tk, filedialog, messagebox, ttk
from tkinter import Text

try:
    import sounddevice as sd  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    sd = None

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # type: ignore[import-not-found]
from matplotlib.figure import Figure  # type: ignore[import-not-found]

from core.fft_filter import apply_hann_window, compute_fft


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATASET_INFO_PATH = BASE_DIR / "general_info_aves.json"
DEFAULT_AUDIO_DIR = BASE_DIR / "dataset_aves_soften"

BACKGROUND = "#0f172a"
PANEL = "#111827"
CARD = "#1f2937"
ACCENT = "#38bdf8"
TEXT = "#e5e7eb"
SUBTEXT = "#94a3b8"
WARN = "#f59e0b"
GOOD = "#22c55e"

FONT_UI_FAMILY = "Segoe UI"
FONT_UI_BOLD_FAMILY = "Segoe UI Semibold"
FONT_UI = (FONT_UI_FAMILY, 10)
FONT_UI_BOLD = (FONT_UI_BOLD_FAMILY, 10)
FONT_TITLE = (FONT_UI_BOLD_FAMILY, 22)
FONT_CARD_TITLE = (FONT_UI_BOLD_FAMILY, 12)
FONT_SUMMARY = (FONT_UI_BOLD_FAMILY, 18)

STYLE_FRAME_CARD = "Card.TFrame"
STYLE_FRAME_PANEL = "Panel.TFrame"
STYLE_LABEL_TITLE = "CardTitle.TLabel"
STYLE_LABEL_TEXT = "CardText.TLabel"
STYLE_BUTTON_ACCENT = "Accent.TButton"
STYLE_SPECIES_NAME = "SpeciesName.TLabel"
STYLE_SPECIES_SCIENTIFIC = "SpeciesScientific.TLabel"

TEXT_APP_TITLE = "Clasificador de aves"
TEXT_APP_SUBTITLE = "FFT, energías por banda, espectro en tiempo real y ficha de especie desde la raíz del proyecto."
TEXT_SECTION_CONFIG = "Configuración"
TEXT_SECTION_MODEL = "Modelo"
TEXT_SECTION_MODE = "Modo de entrada"
TEXT_SECTION_FILE = "Audio WAV"
TEXT_LIVE_MODE = "Tiempo real"
TEXT_SECTION_LIVE = TEXT_LIVE_MODE
TEXT_SECTION_STATUS = "Estado"
TEXT_SECTION_PREDICTION = "Predicción"
TEXT_SECTION_SPECIES_INFO = "Información de la especie"
TEXT_SPECIES_PLACEHOLDER = "Especie predicha"
TEXT_SCIENTIFIC_PLACEHOLDER = "Nombre científico"
TEXT_AUDIO_LOCAL = "Audio local"
TEXT_LIVE = TEXT_LIVE_MODE
TEXT_BROWSE_AUDIO = "Buscar audio"
TEXT_PREDICT_AUDIO = "Predecir audio"
TEXT_START_CAPTURE = "Iniciar captura"
TEXT_STOP_CAPTURE = "Detener captura"
TEXT_PREDICT_LIVE = "Predecir captura actual"
TEXT_PLACEHOLDER_IMAGE = "La foto aparecerá aquí"
TEXT_IMAGE_MISSING = "No se pudo cargar la imagen de la especie."
TEXT_INFO_PLACEHOLDER = "La información de la especie aparecerá aquí después de una predicción.\n"
TEXT_NO_SIGNAL = "Sin señal para mostrar"
TEXT_SPECTRUM = "Espectro FFT"
TEXT_NO_MODEL = "No hay un modelo cargado."
TEXT_NO_AUDIO_FIRST = "Selecciona un archivo WAV primero."
TEXT_AUDIO_NOT_FOUND = "No existe el archivo: "
TEXT_NO_AUDIO_CAPTURE = "No hay audio capturado para predecir."
TEXT_MODEL_LOADED = "Modelo cargado: "
TEXT_NO_PREDICTION = "Sin predicción"
TEXT_CONFIDENCE_DEFAULT = "Confianza: N/D"
TEXT_CONFIDENCE_ND = TEXT_CONFIDENCE_DEFAULT
TEXT_CONFIDENCE_PREFIX = "Confianza: "
TEXT_SOUNDDEVICE_MISSING = "sounddevice no está instalado; el modo en tiempo real quedó deshabilitado."
TEXT_STARTING_LIVE = "Capturando audio en tiempo real..."
TEXT_LIVE_STOPPED = "Captura en tiempo real detenida."
TEXT_LIVE_PREDICTED = "Predicción realizada sobre captura en tiempo real."
TEXT_FILE_PREDICTED = "Predicción realizada sobre "
TEXT_FILE_AUDIO_SELECTED = "Audio seleccionado: "
TEXT_MODEL_UNAVAILABLE = "Modelo no disponible"
TEXT_PREDICTION_ERROR = "Predicción"
TEXT_AUDIO_ERROR = "Audio"
TEXT_MODEL_ERROR = "Modelo"
TEXT_LIVE_ERROR = TEXT_LIVE_MODE
TEXT_STATUS_NO_LIVE = "No se pudo iniciar la captura en vivo."
TEXT_STATUS_NO_LOCAL = "No se pudo predecir el audio seleccionado."
TEXT_STATUS_NO_LIVE_PRED = "No se pudo predecir la captura en vivo."

LIVE_UPDATE_MS = 80
LIVE_SPECTRUM_WINDOW_SECONDS = 0.35
LIVE_MAX_FREQ_HZ = 12000.0
LIVE_MAX_POINTS = 1200


@dataclass
class ModelBundle:
    model_type: str
    model_path: Path
    metadata_path: Path
    model: Any
    model_kind: str
    species: List[str]
    frequency_bands: List[Tuple[float, float]]
    scoring_method: str
    species_profiles: List[Dict[str, Any]]
    accuracy: Optional[float] = None


@dataclass
class BirdInfo:
    species_key: str
    common_name_en: str
    common_name_es: str
    scientific_name: str
    family: str
    order: str
    description: str
    distribution: str
    habitat: List[str]
    vocalization_description: str
    vocalization_notes: str
    image_url: str


class LiveAudioBuffer:
    def __init__(self, sample_rate: int = 44100, seconds: int = 6):
        self.sample_rate = sample_rate
        self.seconds = seconds
        self.capacity = max(1, sample_rate * seconds)
        self.buffer = np.zeros(self.capacity, dtype=np.float32)
        self.write_index = 0
        self.filled = 0
        self.lock = threading.Lock()
        self.stream = None
        self.active = False

    def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice no está instalado en este entorno.")

        if self.active:
            return

        with self.lock:
            self.buffer.fill(0.0)
            self.write_index = 0
            self.filled = 0
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            blocksize=1024,
        )
        self.stream.start()
        self.active = True

    def _callback(self, indata, frames, time, status):  # pragma: no cover - callback driven by audio device
        if status:
            logger.warning("Audio en vivo: %s", status)
        chunk = np.asarray(indata[:, 0], dtype=np.float32)
        chunk_size = int(chunk.size)
        if chunk_size == 0:
            return
        with self.lock:
            if chunk_size >= self.capacity:
                self.buffer[:] = chunk[-self.capacity:]
                self.write_index = 0
                self.filled = self.capacity
                return

            end_index = self.write_index + chunk_size
            if end_index <= self.capacity:
                self.buffer[self.write_index:end_index] = chunk
            else:
                first = self.capacity - self.write_index
                self.buffer[self.write_index:] = chunk[:first]
                self.buffer[: end_index - self.capacity] = chunk[first:]

            self.write_index = end_index % self.capacity
            self.filled = min(self.capacity, self.filled + chunk_size)

    def stop(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None
        self.active = False

    def get_audio(self) -> np.ndarray:
        with self.lock:
            if self.filled == 0:
                return np.array([], dtype=np.float32)

            if self.filled < self.capacity:
                return self.buffer[:self.filled].copy()

            if self.write_index == 0:
                return self.buffer.copy()

            return np.concatenate((self.buffer[self.write_index:], self.buffer[:self.write_index])).astype(np.float32, copy=False)

    def get_recent_audio(self, max_samples: int) -> np.ndarray:
        sample_count = int(max_samples)
        if sample_count <= 0:
            return np.array([], dtype=np.float32)

        with self.lock:
            if self.filled == 0:
                return np.array([], dtype=np.float32)

            take = min(sample_count, self.filled)
            if self.filled < self.capacity:
                return self.buffer[self.filled - take:self.filled].copy()

            start = (self.write_index - take) % self.capacity
            if start < self.write_index:
                return self.buffer[start:self.write_index].copy()

            return np.concatenate((self.buffer[start:], self.buffer[:self.write_index])).astype(np.float32, copy=False)


def normalize_name(value: str) -> str:
    return value.strip().replace(" ", "_").replace("/", "_")


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(info.min), info.max))
        if scale > 0:
            audio = audio.astype(np.float32) / scale
        else:
            audio = audio.astype(np.float32)
    else:
        audio = audio.astype(np.float32)
    return audio


def load_bird_catalog(info_path: Path) -> Dict[str, BirdInfo]:
    if not info_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de metadatos: {info_path}")

    with info_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    catalog: Dict[str, BirdInfo] = {}
    for record in payload.get("aves", []):
        common_en = record.get("nombre_comun_ingles", "")
        common_es = record.get("nombre_comun_espanol", "")
        species_key = normalize_name(common_en)
        if not species_key:
            continue

        bird = BirdInfo(
            species_key=species_key,
            common_name_en=common_en,
            common_name_es=common_es,
            scientific_name=record.get("nombre_cientifico", ""),
            family=record.get("familia", ""),
            order=record.get("orden", ""),
            description=record.get("descripcion", ""),
            distribution=record.get("distribucion", ""),
            habitat=list(record.get("habitat", [])),
            vocalization_description=record.get("vocalizaciones", {}).get("descripcion", ""),
            vocalization_notes=record.get("vocalizaciones", {}).get("frecuencias_Hz", {}).get("notas", ""),
            image_url=record.get("img", ""),
        )

        catalog[species_key] = bird
        catalog[normalize_name(common_es)] = bird
        catalog[normalize_name(record.get("nombre_cientifico", ""))] = bird

    return catalog


def load_image_from_url(url: str, max_size: Tuple[int, int] = (360, 240)) -> Optional[ImageTk.PhotoImage]:
    if not url:
        return None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = response.read()
            content_type = response.headers.get_content_type()

        if content_type == "text/html":
            html = payload.decode("utf-8", errors="ignore")
            image_url = extract_image_url_from_html(url, html)
            if image_url is None:
                return None
            request = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=8) as response:
                payload = response.read()

        image = Image.open(io.BytesIO(payload)).convert("RGB")
        image = ImageOps.contain(image, max_size)
        return ImageTk.PhotoImage(image)
    except Exception as exc:  # pragma: no cover - external network dependent
        logger.warning("No se pudo cargar la imagen %s: %s", url, exc)
        return None


def extract_image_url_from_html(base_url: str, html: str) -> Optional[str]:
    patterns = [
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<img[^>]+src=["\']([^"\']+)["\']',
    ]

    for pattern in patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if match:
            return urljoin(base_url, match.group(1))
    return None


def load_model_bundle(model_type: str) -> ModelBundle:
    model_path = MODELS_DIR / model_type
    if model_path.suffix.lower() != ".json":
        model_path = MODELS_DIR / f"{model_type}.json"
    if not model_path.exists():
        legacy_path = MODELS_DIR / f"model_{model_type}.json"
        if legacy_path.exists():
            model_path = legacy_path
    metadata_path = model_path

    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")

    with model_path.open("r", encoding="utf-8") as handle:
        model = json.load(handle)

    model_kind = str(model.get("kind", ""))
    raw_bands = model.get("frequency_bands") or model.get("bands") or []
    frequency_bands = [tuple(float(value) for value in band) for band in raw_bands]
    species_profiles = list(model.get("species_profiles", []))
    species = [profile.get("species_key", "") for profile in species_profiles]

    return ModelBundle(
        model_type=model_type,
        model_path=model_path,
        metadata_path=metadata_path,
        model=model,
        model_kind=model_kind,
        species=species,
        frequency_bands=frequency_bands,
        scoring_method=str(model.get("scoring_method", "cosine")),
        species_profiles=species_profiles,
        accuracy=model.get("accuracy"),
    )


def discover_models() -> List[str]:
    if not MODELS_DIR.exists():
        return []

    available = []
    for model_file in sorted(MODELS_DIR.glob("*.json")):
        model_type = model_file.stem
        with model_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and (
            payload.get("kind") == "deterministic_spectral_template"
            or payload.get("kind") == "filterbank_energy_thresholds"
            or payload.get("species_profiles")
        ):
            available.append(model_type)
    return sorted(dict.fromkeys(available))


def extract_energy_vector(
    audio: np.ndarray,
    sample_rate: int,
    bands: List[Tuple[float, float]],
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    audio_mono = to_mono_float32(audio)
    if audio_mono.size == 0:
        return np.array([]), np.array([]), np.array([])

    windowed = apply_hann_window(audio_mono)
    freqs, magnitude = compute_fft(windowed, sample_rate)

    energy_vector: List[float] = []
    for low_hz, high_hz in bands:
        mask = (freqs >= low_hz) & (freqs < high_hz)
        band_energy = float(np.sum(magnitude[mask] ** 2)) if np.any(mask) else 0.0
        energy_vector.append(band_energy)

    vector = np.asarray(energy_vector, dtype=np.float32)
    if normalize:
        # Normalize with L2 to match training pipeline (training normalizes with L2)
        vector = normalize_feature_vector(vector)

    return freqs, magnitude, vector


def weighted_profile_vector(profile_vector: np.ndarray, std_vector: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    if profile_vector.size == 0:
        return profile_vector.astype(np.float32, copy=False)
    weights = 1.0 / np.maximum(std_vector, epsilon)
    return (profile_vector * weights).astype(np.float32, copy=False)


def euclidean_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    if vector_a.size == 0 or vector_b.size == 0:
        return 0.0
    return float(np.linalg.norm(vector_a - vector_b))


def normalize_feature_vector(vector: np.ndarray) -> np.ndarray:
    if vector.size == 0:
        return vector.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        return (vector / norm).astype(np.float32, copy=False)
    return vector.astype(np.float32, copy=False)


def spectral_centroid(freqs: np.ndarray, magnitude: np.ndarray) -> float:
    """Compute spectral centroid from FFT freqs and magnitude."""
    if freqs.size == 0 or magnitude.size == 0:
        return 0.0
    total_mag = float(np.sum(magnitude))
    if total_mag <= 0:
        return 0.0
    return float(np.sum(freqs * magnitude) / total_mag)


def compute_matching_distances(model_bundle: ModelBundle, feature_vector: np.ndarray) -> List[float]:
    normalized_vector = normalize_feature_vector(feature_vector)
    distances: List[float] = []

    for profile in model_bundle.species_profiles:
        profile_vector = np.asarray(profile.get("profile_vector", []), dtype=np.float32)
        std_vector = np.asarray(profile.get("std_energy_vector", []), dtype=np.float32)
        if model_bundle.scoring_method == "weighted":
            reference_vector = weighted_profile_vector(profile_vector, std_vector)
            sample_vector = weighted_profile_vector(normalized_vector, std_vector)
        else:
            reference_vector = profile_vector
            sample_vector = normalized_vector
        distances.append(euclidean_distance(sample_vector, reference_vector))

    return distances


def derive_rejection_threshold(profile: Dict[str, Any]) -> float:
    stored_threshold = profile.get("rejection_threshold")
    if stored_threshold is not None:
        return float(stored_threshold)

    std_vector = np.asarray(profile.get("std_energy_vector", []), dtype=np.float32)
    if std_vector.size == 0:
        return 0.75

    spread = float(np.mean(std_vector) + np.std(std_vector))
    sample_count = float(profile.get("sample_count", 1) or 1)
    sample_factor = min(max(sample_count / 1000.0, 0.6), 1.5)
    return float(max(0.15, spread * (2.0 / sample_factor) + 0.05))


def distances_to_confidence(distances: List[float], temperature: float = 0.35) -> Optional[float]:
    if not distances:
        return None
    values = np.asarray(distances, dtype=np.float32)
    logits = -values
    shifted = logits - np.max(logits)
    scaled = shifted / max(temperature, 1e-6)
    exp_scores = np.exp(scaled)
    total = float(np.sum(exp_scores))
    if total <= 0:
        return None
    probabilities = exp_scores / total
    return float(np.max(probabilities))


def predict_species(model_bundle: ModelBundle, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    is_filterbank_model = model_bundle.model_kind == "filterbank_energy_thresholds"
    freqs, magnitude, feature_vector = extract_energy_vector(
        audio,
        sample_rate,
        model_bundle.frequency_bands,
        normalize=not is_filterbank_model,
    )
    if feature_vector.size == 0:
        raise ValueError("No se pudieron extraer características del audio.")

    if not model_bundle.frequency_bands:
        raise ValueError("El modelo cargado no contiene bandas de frecuencia.")

    if is_filterbank_model:
        sample_vector = feature_vector.astype(np.float32, copy=False)
        distances: List[float] = []
        for profile in model_bundle.species_profiles:
            mean_vector = np.asarray(profile.get("mean_energy_vector", []), dtype=np.float32)
            std_vector = np.asarray(profile.get("std_energy_vector", []), dtype=np.float32)
            if mean_vector.size != sample_vector.size:
                distances.append(float("inf"))
                continue

            # Primary score follows the requested algorithm: sum of absolute differences.
            direct_score = float(np.sum(np.abs(sample_vector - mean_vector)))

            # Note-aware refinement: add a tiny penalty only when the sample exits mean ± std.
            lower = mean_vector - std_vector
            upper = mean_vector + std_vector
            below = np.maximum(lower - sample_vector, 0.0)
            above = np.maximum(sample_vector - upper, 0.0)
            interval_penalty = float(np.sum(below + above))

            distances.append(direct_score + 0.05 * interval_penalty)
    else:
        distances = compute_matching_distances(model_bundle, feature_vector)

    if not distances:
        raise ValueError("El modelo no contiene perfiles para comparar.")

    # Determine top-2 candidates
    sorted_indices = np.argsort(np.asarray(distances, dtype=np.float32))
    predicted_index = int(sorted_indices[0])
    second_index = int(sorted_indices[1]) if len(sorted_indices) > 1 else None
    predicted_profile = model_bundle.species_profiles[predicted_index]
    predicted_label = str(predicted_profile.get("species_key", ""))
    best_distance = float(distances[predicted_index])
    second_best_distance = float(distances[second_index]) if second_index is not None else float("inf")
    threshold = derive_rejection_threshold(predicted_profile)
    ambiguity_margin = float(model_bundle.model.get("matching", {}).get("ambiguity_margin", 0.05))
    confidence = distances_to_confidence(distances, float(model_bundle.model.get("matching", {}).get("temperature", 0.35)))

    # Basic rejection check
    ambig_limit = max(ambiguity_margin * max(threshold, 1e-6), 0.01)
    rejected = best_distance > threshold
    rejection_reason = None

    # If within ambiguity margin, attempt a lightweight tie-breaker using spectral centroid
    if not rejected and second_index is not None and (second_best_distance - best_distance) <= ambig_limit:
        # compute sample centroid
        sample_centroid = spectral_centroid(freqs, magnitude)
        # compute centroid per profile using mean_magnitude_vector if available, else fallback to band centers
        def profile_centroid(profile: Dict[str, Any]) -> float:
            mean_mag = np.asarray(profile.get("mean_magnitude_vector", []), dtype=np.float32)
            if mean_mag.size and np.sum(mean_mag) > 0:
                centers = np.asarray(model_bundle.model.get("feature_summary", {}).get("band_centers_hz", []), dtype=np.float32)
                if centers.size == mean_mag.size:
                    return float(np.sum(centers * mean_mag) / np.sum(mean_mag))
            prof_vec = np.asarray(profile.get("profile_vector", []), dtype=np.float32)
            if prof_vec.size:
                idx = int(np.argmax(prof_vec))
                bands = model_bundle.frequency_bands
                low, high = bands[idx]
                return float((low + high) / 2.0)
            return 0.0

        pred_centroid = profile_centroid(predicted_profile)
        second_profile = model_bundle.species_profiles[second_index]
        second_centroid = profile_centroid(second_profile)

        d_pred = abs(sample_centroid - pred_centroid)
        d_second = abs(sample_centroid - second_centroid)
        logger.debug("Top-2 distances: best=%.6f second=%.6f; centroid diffs: pred=%.2f vs second=%.2f (sample=%.2f)", best_distance, second_best_distance, d_pred, d_second, sample_centroid)

        # If centroid prefers the second candidate by a meaningful margin, swap
        if d_second + 1e-6 < d_pred:
            predicted_index = second_index
            predicted_profile = second_profile
            predicted_label = str(predicted_profile.get("species_key", ""))
            best_distance = float(distances[predicted_index])

    if not rejected:
        rejected = best_distance > threshold
        if rejected:
            rejection_reason = "Fuera de umbral"
    else:
        rejection_reason = "Fuera de umbral"

    return {
        "predicted_label": None if rejected else predicted_label,
        "predicted_index": predicted_index,
        "confidence": confidence,
        "feature_vector": feature_vector,
        "freqs": freqs,
        "magnitude": magnitude,
        "distances": distances,
        "best_distance": best_distance,
        "threshold": threshold,
        "rejected": rejected,
        "rejection_reason": rejection_reason,
        "model_kind": model_bundle.model_kind,
    }


class BirdClassifierApp(Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Clasificador de aves")
        self.geometry("1450x920")
        self.minsize(1280, 840)
        self.configure(bg=BACKGROUND)

        self.bird_catalog = load_bird_catalog(DATASET_INFO_PATH)
        self.available_models = discover_models()
        if not self.available_models:
            raise FileNotFoundError("No se encontraron modelos en la carpeta models/.")

        self.selected_model = StringVar(value=self.available_models[0])
        self.mode = StringVar(value="archivo")
        self.audio_path = StringVar(value="")
        self.status_text = StringVar(value="Selecciona un modelo y un audio, o inicia la captura en tiempo real.")
        self.prediction_text = StringVar(value="Sin predicción")
        self.confidence_text = StringVar(value=TEXT_CONFIDENCE_DEFAULT)
        self.model_text = StringVar(value="")

        self.current_model: Optional[ModelBundle] = None
        self.current_audio: Optional[np.ndarray] = None
        self.current_sample_rate: Optional[int] = None
        self.current_spectrum = (np.array([]), np.array([]))
        self.current_image = None
        self.live_buffer = LiveAudioBuffer()
        self.live_after_id: Optional[str] = None
        self._live_predict_running = False
        self._live_predict_lock = threading.Lock()
        self.live_predict_window_seconds = 1.0  # use last N seconds for each live prediction
        self._last_live_species_key: Optional[str] = None

        self._configure_style()
        self._build_layout()
        self._load_model_from_selection()
        self._refresh_mode_ui()
        self._set_placeholder_info()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TFrame", background=BACKGROUND)
        style.configure(STYLE_FRAME_CARD, background=CARD)
        style.configure(STYLE_FRAME_PANEL, background=PANEL)
        style.configure("TLabel", background=BACKGROUND, foreground=TEXT, font=FONT_UI)
        style.configure("Title.TLabel", background=BACKGROUND, foreground=TEXT, font=FONT_TITLE)
        style.configure("Subtitle.TLabel", background=BACKGROUND, foreground=SUBTEXT, font=FONT_UI)
        style.configure(STYLE_LABEL_TITLE, background=CARD, foreground=TEXT, font=FONT_CARD_TITLE)
        style.configure(STYLE_LABEL_TEXT, background=CARD, foreground=TEXT, font=FONT_UI)
        style.configure(STYLE_BUTTON_ACCENT, font=FONT_UI_BOLD)
        style.configure(STYLE_SPECIES_NAME, background=CARD, foreground=TEXT, font=FONT_SUMMARY)
        style.configure(STYLE_SPECIES_SCIENTIFIC, background=CARD, foreground=SUBTEXT, font=(FONT_UI_FAMILY, 10, "italic"))
        style.map(STYLE_BUTTON_ACCENT, foreground=[("active", TEXT)], background=[("active", ACCENT)])

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, padding=(24, 20, 24, 10))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text=TEXT_APP_TITLE, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text=TEXT_APP_SUBTITLE,
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        body = ttk.Frame(self, padding=(20, 0, 20, 20))
        body.grid(row=1, column=0, columnspan=2, sticky="nsew")
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.left_panel = ttk.Frame(body, style=STYLE_FRAME_PANEL, padding=18)
        self.left_panel.grid(row=0, column=0, sticky="nsw", padx=(0, 14))
        self.left_panel.columnconfigure(0, weight=1)

        self.right_panel = ttk.Frame(body, style="TFrame")
        self.right_panel.grid(row=0, column=1, sticky="nsew")
        self.right_panel.columnconfigure(0, weight=1)
        self.right_panel.rowconfigure(1, weight=1)
        self.right_panel.rowconfigure(2, weight=1)

        self._build_left_panel()
        self._build_right_panel()

    def _build_left_panel(self) -> None:
        ttk.Label(self.left_panel, text=TEXT_SECTION_CONFIG, style=STYLE_LABEL_TITLE).grid(row=0, column=0, sticky="w")

        model_box = ttk.Frame(self.left_panel, style=STYLE_FRAME_CARD, padding=14)
        model_box.grid(row=1, column=0, sticky="ew", pady=(12, 12))
        model_box.columnconfigure(0, weight=1)
        ttk.Label(model_box, text=TEXT_SECTION_MODEL, style=STYLE_LABEL_TITLE).grid(row=0, column=0, sticky="w")
        self.model_combo = ttk.Combobox(
            model_box,
            textvariable=self.selected_model,
            values=self.available_models,
            state="readonly",
            width=24,
        )
        self.model_combo.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.model_combo.bind("<<ComboboxSelected>>", lambda _event: self._load_model_from_selection())
        self.model_status_label = ttk.Label(model_box, textvariable=self.model_text, style=STYLE_LABEL_TEXT, wraplength=280)
        self.model_status_label.grid(row=2, column=0, sticky="w", pady=(10, 0))

        mode_box = ttk.Frame(self.left_panel, style=STYLE_FRAME_CARD, padding=14)
        mode_box.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        mode_box.columnconfigure(0, weight=1)
        ttk.Label(mode_box, text=TEXT_SECTION_MODE, style=STYLE_LABEL_TITLE).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(mode_box, text=TEXT_AUDIO_LOCAL, variable=self.mode, value="archivo", command=self._refresh_mode_ui).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Radiobutton(mode_box, text=TEXT_LIVE, variable=self.mode, value="tiempo_real", command=self._refresh_mode_ui).grid(row=2, column=0, sticky="w", pady=(4, 0))

        file_box = ttk.Frame(self.left_panel, style=STYLE_FRAME_CARD, padding=14)
        file_box.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        file_box.columnconfigure(0, weight=1)
        ttk.Label(file_box, text=TEXT_SECTION_FILE, style=STYLE_LABEL_TITLE).grid(row=0, column=0, sticky="w")
        self.file_entry = ttk.Entry(file_box, textvariable=self.audio_path, width=34)
        self.file_entry.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        ttk.Button(file_box, text=TEXT_BROWSE_AUDIO, command=self._browse_audio).grid(row=2, column=0, sticky="ew")
        self.predict_file_button = ttk.Button(file_box, text=TEXT_PREDICT_AUDIO, command=self.predict_selected_audio, style=STYLE_BUTTON_ACCENT)
        self.predict_file_button.grid(row=3, column=0, sticky="ew", pady=(8, 0))

        live_box = ttk.Frame(self.left_panel, style=STYLE_FRAME_CARD, padding=14)
        live_box.grid(row=4, column=0, sticky="ew", pady=(0, 12))
        live_box.columnconfigure(0, weight=1)
        ttk.Label(live_box, text=TEXT_SECTION_LIVE, style=STYLE_LABEL_TITLE).grid(row=0, column=0, sticky="w")
        self.start_live_button = ttk.Button(live_box, text=TEXT_START_CAPTURE, command=self.start_live_capture)
        self.start_live_button.grid(row=1, column=0, sticky="ew", pady=(8, 6))
        self.stop_live_button = ttk.Button(live_box, text=TEXT_STOP_CAPTURE, command=self.stop_live_capture, state=DISABLED)
        self.stop_live_button.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        self.predict_live_button = ttk.Button(live_box, text=TEXT_PREDICT_LIVE, command=self.predict_live_audio, state=DISABLED, style=STYLE_BUTTON_ACCENT)
        self.predict_live_button.grid(row=3, column=0, sticky="ew")

        status_box = ttk.Frame(self.left_panel, style=STYLE_FRAME_CARD, padding=14)
        status_box.grid(row=5, column=0, sticky="ew")
        status_box.columnconfigure(0, weight=1)
        ttk.Label(status_box, text=TEXT_SECTION_STATUS, style=STYLE_LABEL_TITLE).grid(row=0, column=0, sticky="w")
        self.status_label = ttk.Label(status_box, textvariable=self.status_text, style=STYLE_LABEL_TEXT, wraplength=280)
        self.status_label.grid(row=1, column=0, sticky="w", pady=(8, 0))

    def _build_right_panel(self) -> None:
        self.summary_card = ttk.Frame(self.right_panel, style=STYLE_FRAME_CARD, padding=18)
        self.summary_card.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        self.summary_card.columnconfigure(0, weight=1)
        self.summary_card.columnconfigure(1, weight=0)

        ttk.Label(self.summary_card, text=TEXT_SECTION_PREDICTION, style=STYLE_LABEL_TITLE).grid(row=0, column=0, sticky="w")
        self.prediction_label = ttk.Label(self.summary_card, textvariable=self.prediction_text, style=STYLE_LABEL_TEXT, font=FONT_SUMMARY)
        self.prediction_label.grid(row=1, column=0, sticky="w", pady=(8, 2))
        ttk.Label(self.summary_card, textvariable=self.confidence_text, style=STYLE_LABEL_TEXT).grid(row=2, column=0, sticky="w")

        self.figure = Figure(figsize=(8, 4), dpi=100, facecolor=CARD)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(CARD)
        self.ax.tick_params(colors=TEXT)
        self.ax.spines["bottom"].set_color(TEXT)
        self.ax.spines["top"].set_color(TEXT)
        self.ax.spines["left"].set_color(TEXT)
        self.ax.spines["right"].set_color(TEXT)
        self.ax.set_title(TEXT_SPECTRUM, color=TEXT)
        self.ax.set_xlabel("Frecuencia (Hz)", color=TEXT)
        self.ax.set_ylabel("Magnitud", color=TEXT)
        self.ax.grid(True, alpha=0.22)
        self.spectrum_line, = self.ax.plot([], [], color=ACCENT, linewidth=1.6)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_panel)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, sticky="nsew", pady=(0, 14))

        info_card = ttk.Frame(self.right_panel, style=STYLE_FRAME_CARD, padding=18)
        info_card.grid(row=2, column=0, sticky="nsew")
        info_card.columnconfigure(0, weight=1)
        info_card.columnconfigure(1, weight=1)
        info_card.rowconfigure(2, weight=1)

        ttk.Label(info_card, text=TEXT_SECTION_SPECIES_INFO, style=STYLE_LABEL_TITLE).grid(row=0, column=0, columnspan=2, sticky="w")

        species_header = ttk.Frame(info_card, style=STYLE_FRAME_CARD)
        species_header.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 4))
        species_header.columnconfigure(0, weight=1)
        self.species_name_label = ttk.Label(species_header, text=TEXT_SPECIES_PLACEHOLDER, style=STYLE_SPECIES_NAME)
        self.species_name_label.grid(row=0, column=0, sticky="w")
        self.species_scientific_label = ttk.Label(species_header, text=TEXT_SCIENTIFIC_PLACEHOLDER, style=STYLE_SPECIES_SCIENTIFIC)
        self.species_scientific_label.grid(row=1, column=0, sticky="w", pady=(2, 0))

        image_frame = ttk.Frame(info_card, style=STYLE_FRAME_CARD)
        image_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 16), pady=(12, 0))
        image_frame.columnconfigure(0, weight=1)
        self.image_label = ttk.Label(image_frame)
        self.image_label.grid(row=0, column=0, sticky="n", pady=(0, 8))
        self.image_caption = ttk.Label(image_frame, text=TEXT_PLACEHOLDER_IMAGE, style=STYLE_LABEL_TEXT, wraplength=340)
        self.image_caption.grid(row=1, column=0, sticky="n")

        text_frame = ttk.Frame(info_card, style=STYLE_FRAME_CARD)
        text_frame.grid(row=2, column=1, sticky="nsew", pady=(12, 0))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        self.info_text = Text(
            text_frame,
            height=14,
            wrap="word",
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=FONT_UI,
        )
        self.info_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.info_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.info_text.configure(yscrollcommand=scrollbar.set)
        self.info_text.configure(state=DISABLED)

    def _set_placeholder_info(self) -> None:
        self._update_info_panel(None)
        self._plot_spectrum(np.array([]), np.array([]), title=TEXT_SPECTRUM)

    def _refresh_mode_ui(self) -> None:
        is_file_mode = self.mode.get() == "archivo"
        self.file_entry.configure(state=NORMAL if is_file_mode else DISABLED)
        self.predict_file_button.configure(state=NORMAL if is_file_mode else DISABLED)

        live_state = NORMAL if not is_file_mode else DISABLED
        if sd is None:
            live_state = DISABLED
            self.status_text.set(TEXT_SOUNDDEVICE_MISSING)
        self.start_live_button.configure(state=live_state)
        if is_file_mode:
            self.stop_live_button.configure(state=DISABLED)
            self.predict_live_button.configure(state=DISABLED)
        else:
            self.predict_live_button.configure(state=live_state if self.live_buffer.active else DISABLED)
            self.stop_live_button.configure(state=live_state if self.live_buffer.active else DISABLED)

    def _load_model_from_selection(self) -> None:
        try:
            self.current_model = load_model_bundle(self.selected_model.get())
            species_count = len(self.current_model.species_profiles)
            self.model_text.set(
                f"{self.current_model.model_path.name} · estrategia: {self.current_model.scoring_method} · {species_count} especies"
            )
            self.status_text.set(f"{TEXT_MODEL_LOADED}{self.current_model.model_type}")
        except Exception as exc:
            self.current_model = None
            self.model_text.set(str(exc))
            self.status_text.set("No se pudo cargar el modelo seleccionado.")
            messagebox.showerror(TEXT_MODEL_UNAVAILABLE, str(exc))

    def _browse_audio(self) -> None:
        initial_dir = str(DEFAULT_AUDIO_DIR if DEFAULT_AUDIO_DIR.exists() else BASE_DIR)
        filename = filedialog.askopenfilename(
            title=TEXT_SECTION_FILE,
            initialdir=initial_dir,
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if filename:
            self.audio_path.set(filename)
            self.status_text.set(f"{TEXT_FILE_AUDIO_SELECTED}{Path(filename).name}")

    def _read_audio_file(self, path: Path) -> Tuple[np.ndarray, int]:
        sample_rate, audio = wavfile.read(str(path))
        return to_mono_float32(audio), sample_rate

    def _plot_spectrum(self, freqs: np.ndarray, magnitude: np.ndarray, title: str = "Espectro FFT") -> None:
        self.ax.clear()
        self.ax.set_facecolor(CARD)
        self.ax.tick_params(colors=TEXT)
        self.ax.spines["bottom"].set_color(TEXT)
        self.ax.spines["top"].set_color(TEXT)
        self.ax.spines["left"].set_color(TEXT)
        self.ax.spines["right"].set_color(TEXT)
        self.ax.set_title(title, color=TEXT)
        self.ax.set_xlabel("Frecuencia (Hz)", color=TEXT)
        self.ax.set_ylabel("Magnitud", color=TEXT)
        self.ax.grid(True, alpha=0.22)

        if freqs.size and magnitude.size:
            self.ax.plot(freqs, magnitude, color=ACCENT, linewidth=1.2)
            limit = min(float(np.max(freqs)), 20000.0)
            self.ax.set_xlim(0, limit if limit > 0 else 1)
        else:
            self.ax.text(0.5, 0.5, TEXT_NO_SIGNAL, transform=self.ax.transAxes, ha="center", va="center", color=SUBTEXT)

        self.canvas.draw_idle()

    def _update_info_panel(self, bird: Optional[BirdInfo]) -> None:
        self.info_text.configure(state=NORMAL)
        self.info_text.delete("1.0", END)

        if bird is None:
            self.species_name_label.configure(text=TEXT_SPECIES_PLACEHOLDER)
            self.species_scientific_label.configure(text=TEXT_SCIENTIFIC_PLACEHOLDER)
            self.info_text.insert(END, TEXT_INFO_PLACEHOLDER)
            self.image_label.configure(image="")
            self.image_label.image = None
            self.image_caption.configure(text=TEXT_PLACEHOLDER_IMAGE)
            self.info_text.configure(state=DISABLED)
            return

        self.species_name_label.configure(text=bird.common_name_en)
        self.species_scientific_label.configure(text=bird.scientific_name)

        lines = [
            f"Nombre común: {bird.common_name_en}",
            f"Nombre en español: {bird.common_name_es}",
            f"Nombre científico: {bird.scientific_name}",
            f"Familia: {bird.family}",
            f"Orden: {bird.order}",
            "",
            f"Descripción: {bird.description}",
            f"Distribución: {bird.distribution}",
            "",
            "Hábitat:",
        ]
        lines.extend([f"- {item}" for item in bird.habitat] if bird.habitat else ["- Sin datos"])
        lines.extend([
            "",
            f"Vocalización: {bird.vocalization_description}",
            f"Notas: {bird.vocalization_notes}",
        ])
        self.info_text.insert(END, "\n".join(lines))
        self.info_text.configure(state=DISABLED)

        image = load_image_from_url(bird.image_url)
        if image is not None:
            self.current_image = image
            self.image_label.configure(image=image)
            self.image_label.image = image
            self.image_caption.configure(text=f"Foto de {bird.common_name_en}")
        else:
            self.image_label.configure(image="")
            self.image_label.image = None
            self.image_caption.configure(text=TEXT_IMAGE_MISSING)

    def _show_prediction_result(self, result: Dict[str, Any]) -> None:
        predicted_label = result.get("predicted_label")
        confidence = result.get("confidence")

        if not predicted_label:
            self.prediction_text.set("No clasificable")
            self.confidence_text.set(TEXT_CONFIDENCE_DEFAULT)
            self._update_info_panel(None)
            freqs = result.get("freqs", np.array([]))
            magnitude = result.get("magnitude", np.array([]))
            self._plot_spectrum(freqs, magnitude, title=TEXT_LIVE)
            return

        bird = self.bird_catalog.get(normalize_name(predicted_label))
        if bird is None:
            bird = self.bird_catalog.get(predicted_label)

        display_name = bird.common_name_en if bird is not None else predicted_label.replace("_", " ")
        self.prediction_text.set(display_name)
        if confidence is not None:
            self.confidence_text.set(f"Confianza: {confidence:.2%}")
        else:
            self.confidence_text.set(TEXT_CONFIDENCE_DEFAULT)

        self._update_info_panel(bird)

        freqs = result.get("freqs", np.array([]))
        magnitude = result.get("magnitude", np.array([]))
        title = f"Espectro FFT - {display_name}"
        self._plot_spectrum(freqs, magnitude, title=title)

    def predict_selected_audio(self) -> None:
        if self.current_model is None:
            messagebox.showerror(TEXT_MODEL_ERROR, TEXT_NO_MODEL)
            return

        file_value = self.audio_path.get().strip()
        if not file_value:
            messagebox.showwarning(TEXT_AUDIO_ERROR, TEXT_NO_AUDIO_FIRST)
            return

        audio_file = Path(file_value)
        if not audio_file.exists():
            messagebox.showerror(TEXT_AUDIO_ERROR, f"{TEXT_AUDIO_NOT_FOUND}{audio_file}")
            return

        try:
            audio, sample_rate = self._read_audio_file(audio_file)
            self.current_audio = audio
            self.current_sample_rate = sample_rate
            result = predict_species(self.current_model, audio, sample_rate)
            self.status_text.set(f"{TEXT_FILE_PREDICTED}{audio_file.name}")
            self._show_prediction_result(result)
        except Exception as exc:
            logger.exception("Error procesando audio local")
            messagebox.showerror(TEXT_PREDICTION_ERROR, str(exc))
            self.status_text.set(TEXT_STATUS_NO_LOCAL)

    def start_live_capture(self) -> None:
        if self.current_model is None:
            messagebox.showerror(TEXT_MODEL_ERROR, TEXT_NO_MODEL)
            return
        if sd is None:
            messagebox.showerror(TEXT_LIVE_ERROR, "sounddevice no está instalado. Instálalo para usar el modo en tiempo real.")
            return
        if self.live_buffer.active:
            return

        try:
            self.live_buffer.start()
            self.status_text.set(TEXT_STARTING_LIVE)
            self.start_live_button.configure(state=DISABLED)
            self.stop_live_button.configure(state=NORMAL)
            self.predict_live_button.configure(state=NORMAL)
            self._schedule_live_update()
        except Exception as exc:
            messagebox.showerror(TEXT_LIVE_ERROR, str(exc))
            self.status_text.set(TEXT_STATUS_NO_LIVE)

    def _plot_live_spectrum_fast(self, freqs: np.ndarray, magnitude: np.ndarray) -> None:
        if freqs.size == 0 or magnitude.size == 0:
            self.spectrum_line.set_data([], [])
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.ax.set_title(TEXT_LIVE, color=TEXT)
            self.canvas.draw_idle()
            return

        mask = freqs <= LIVE_MAX_FREQ_HZ
        x = freqs[mask]
        y = magnitude[mask]
        if x.size == 0 or y.size == 0:
            self.spectrum_line.set_data([], [])
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.ax.set_title(TEXT_LIVE, color=TEXT)
            self.canvas.draw_idle()
            return

        if x.size > LIVE_MAX_POINTS:
            index = np.linspace(0, x.size - 1, LIVE_MAX_POINTS, dtype=np.int32)
            x = x[index]
            y = y[index]

        self.spectrum_line.set_data(x, y)
        x_max = float(x[-1]) if x.size else LIVE_MAX_FREQ_HZ
        y_max = float(np.max(y)) if y.size else 1.0
        self.ax.set_xlim(0, max(1000.0, min(LIVE_MAX_FREQ_HZ, x_max)))
        self.ax.set_ylim(0, max(1e-6, y_max * 1.05))
        self.ax.set_title(TEXT_LIVE, color=TEXT)
        self.canvas.draw_idle()

    def _live_window_lengths(self, sample_rate: int) -> Tuple[int, int]:
        live_window_len = int(LIVE_SPECTRUM_WINDOW_SECONDS * float(sample_rate))
        if live_window_len <= 0:
            live_window_len = sample_rate

        predict_window_len = int(self.live_predict_window_seconds * float(sample_rate))
        if predict_window_len <= 0:
            predict_window_len = sample_rate

        return live_window_len, predict_window_len

    def _try_start_live_prediction(self, audio: np.ndarray, predict_window_len: int, sample_rate: int) -> None:
        if self.current_model is None or self._live_predict_running:
            return

        audio_window = audio[-predict_window_len:].copy()
        worker = threading.Thread(target=self._do_live_predict, args=(audio_window, sample_rate), daemon=True)
        worker.start()

    def _predict_live_result(self, audio: np.ndarray, sample_rate: int) -> Optional[Dict[str, Any]]:
        model_snapshot = self.current_model
        if model_snapshot is None:
            return None
        try:
            return predict_species(model_snapshot, audio, sample_rate)
        except Exception as exc:
            logger.exception("Error during live prediction: %s", exc)
            return None

    def _schedule_live_result(self, result: Dict[str, Any]) -> None:
        try:
            self.after(0, lambda: self._handle_live_result(result))
        except Exception:
            logger.exception("Error scheduling UI update for live prediction")

    def _handle_live_rejected_result(self, result: Dict[str, Any]) -> None:
        self.status_text.set(f"{TEXT_LIVE_PREDICTED} - Rechazado: {result.get('rejection_reason')}")
        self.prediction_text.set("No clasificable")
        self.confidence_text.set(TEXT_CONFIDENCE_DEFAULT)

    def _resolve_bird_from_label(self, predicted_label: str) -> Optional[BirdInfo]:
        bird = self.bird_catalog.get(normalize_name(predicted_label))
        if bird is None:
            bird = self.bird_catalog.get(predicted_label)
        return bird

    def _handle_live_accepted_result(self, predicted_label: str, confidence: Optional[float]) -> None:
        self.status_text.set(TEXT_LIVE_PREDICTED)
        bird = self._resolve_bird_from_label(predicted_label)

        display_name = bird.common_name_en if bird is not None else str(predicted_label).replace("_", " ")
        self.prediction_text.set(display_name)
        if confidence is not None:
            self.confidence_text.set(f"Confianza: {confidence:.2%}")
        else:
            self.confidence_text.set(TEXT_CONFIDENCE_DEFAULT)

        # Refresh info panel only when species changes to avoid network/image overhead.
        new_key = normalize_name(predicted_label)
        if new_key != self._last_live_species_key:
            self._last_live_species_key = new_key
            self._update_info_panel(bird)

    def _process_live_audio_frame(self, audio: np.ndarray, sample_rate: int, live_window_len: int, predict_window_len: int) -> None:
        if audio.size <= 64:
            return

        audio_live = audio[-live_window_len:]
        freqs, magnitude, _ = extract_energy_vector(audio_live, sample_rate, self.current_model.frequency_bands if self.current_model else [])
        self.current_audio = audio
        self.current_sample_rate = sample_rate
        self._plot_live_spectrum_fast(freqs, magnitude)

        # Launch asynchronous prediction on the most recent window (avoid overlapping predictions)
        try:
            self._try_start_live_prediction(audio, predict_window_len, sample_rate)
        except Exception:
            logger.exception("Error iniciando predicción en background")

    def _release_live_predict_lock(self) -> None:
        self._live_predict_running = False
        try:
            self._live_predict_lock.release()
        except Exception:
            pass

    def _schedule_live_update(self) -> None:
        if not self.live_buffer.active:
            return

        sample_rate = int(self.live_buffer.sample_rate)
        live_window_len, predict_window_len = self._live_window_lengths(sample_rate)
        request_len = max(live_window_len, predict_window_len)
        audio = self.live_buffer.get_recent_audio(request_len)
        self._process_live_audio_frame(audio, sample_rate, live_window_len, predict_window_len)

        self.live_after_id = self.after(LIVE_UPDATE_MS, self._schedule_live_update)


    def _do_live_predict(self, audio: np.ndarray, sample_rate: int) -> None:
        # ensure only one background prediction runs at a time
        if not self._live_predict_lock.acquire(blocking=False):
            return
        self._live_predict_running = True
        try:
            result = self._predict_live_result(audio, sample_rate)
            if result is None:
                return
            self._schedule_live_result(result)
        finally:
            self._release_live_predict_lock()


    def _handle_live_result(self, result: Dict[str, Any]) -> None:
        # Keep live updates light: avoid reloading image/text on every frame.
        try:
            predicted_label = result.get("predicted_label")
            confidence = result.get("confidence")

            if result.get("rejected"):
                self._handle_live_rejected_result(result)
                return

            if predicted_label:
                self._handle_live_accepted_result(str(predicted_label), confidence)
        except Exception:
            logger.exception("Error al actualizar UI con resultado en vivo")

    def stop_live_capture(self) -> None:
        if self.live_after_id is not None:
            try:
                self.after_cancel(self.live_after_id)
            except Exception:
                pass
            self.live_after_id = None

        if self.live_buffer.active:
            self.live_buffer.stop()
            self.status_text.set(TEXT_LIVE_STOPPED)
        self._last_live_species_key = None
        self._refresh_mode_ui()

    def predict_live_audio(self) -> None:
        if self.current_model is None:
            messagebox.showerror("Modelo", "No hay un modelo cargado.")
            return

        audio = self.live_buffer.get_audio() if self.live_buffer.active else self.current_audio
        sample_rate = self.live_buffer.sample_rate if self.live_buffer.active else self.current_sample_rate
        if audio is None or sample_rate is None or audio.size == 0:
            messagebox.showwarning(TEXT_LIVE_ERROR, TEXT_NO_AUDIO_CAPTURE)
            return

        try:
            result = predict_species(self.current_model, audio, sample_rate)
            self.status_text.set(TEXT_LIVE_PREDICTED)
            self._show_prediction_result(result)
        except Exception as exc:
            logger.exception("Error al predecir audio en vivo")
            messagebox.showerror(TEXT_PREDICTION_ERROR, str(exc))
            self.status_text.set(TEXT_STATUS_NO_LIVE_PRED)

    def on_close(self) -> None:
        try:
            self.stop_live_capture()
        finally:
            self.destroy()


def main() -> None:
    app = BirdClassifierApp()
    app.mainloop()


if __name__ == "__main__":
    main()
