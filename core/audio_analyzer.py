import json
from pathlib import Path
from typing import Optional
from config.routes_path import RoutesPath
from config.frecuency_bands import FrequencyBands
from core.audio_converter import AudioConverter
from core.maths.fft import FFTProcessor
from core.remove_trashs_audios import RemoveTrashAudios
from core.models_managment import ModelsManagement
from core.dto.audio_stats import AudioStats
from config.frecuency_bands import FrequencyBands
from core.butterworth_controller import ButterworthController
from core.maths.filter_butterworth import FilterButterworth
from core.maths.energy_vector import EnergyVector
from core.maths.statistics import Statistics

import numpy as np
import soundfile as sf

from core.repo.birds_repo import BirdRepository

class AudioAnalyzer:
    """
    Analiza un corpus de audios normalizados y detecta outliers
    según la desviación estándar de múltiples métricas.
 
    Parámetros
    ----------
    normalized_dir  : carpeta con los audios ya normalizados.
    std_threshold   : número de desviaciones estándar para considerar outlier.
                      Valor típico: 2.5 – 3.5.
    silence_thresh  : umbral de amplitud (float32) para considerar silencio.
    butterworth_order     : orden del filtro Butterworth que se entrena/guarda.
    butterworth_cutoff_hz : frecuencia de corte del filtro guardado.
    trash_json      : ruta del JSON de basura (por defecto trash_audios.json).
    models_dir      : directorio donde se guardan los modelos.
    """
 
    def __init__(
        self,
        normalized_dir: str | Path | None = None,
        std_threshold: float = 3.0,
        silence_thresh: float = 0.01,
        butterworth_order: int = 4,
        min_duration_sec: float = 3.0,
        butterworth_cutoff_hz: float = 8000.0,
        trash_json: str | Path | None = None,
        models_dir: str | Path = "models",
    ):
        self.normalized_dir = Path(
            normalized_dir or RoutesPath.BANK_AUDIOS_NORMALIZED
        )
        self.std_threshold = std_threshold
        self.silence_thresh = silence_thresh
        self.butterworth_order = butterworth_order
        self.butterworth_cutoff_hz = butterworth_cutoff_hz
        self.min_duration_sec = min_duration_sec
        self._converter = AudioConverter()
        self._fft = FFTProcessor()
        self.models_manage = ModelsManagement(base_dir=models_dir)
        self.bird_repo = BirdRepository()
        self.trash = RemoveTrashAudios(json_path=trash_json)
        self.models = ModelsManagement(base_dir=models_dir)
 
        self._stats: list[AudioStats] = []
 
    # ------------------------------------------------------------------
    # Punto de entrada principal
    # ------------------------------------------------------------------
 
    def run(self) -> list[dict]:
        """
        Ejecuta el pipeline completo:
          1. Descubre archivos de audio en la carpeta normalizada.
          2. Analiza cada uno en detalle.
          3. Detecta outliers comparando con la distribución del corpus.
          4. Marca los outliers en RemoveTrashAudios.
          5. Entrena y guarda el modelo Butterworth con ModelsManagement.
 
        Retorna la lista de dicts con los stats de cada archivo.
        """
        print(f"\n{'='*60}")
        print(f"  AudioAnalyzer — corpus: {self.normalized_dir}")
        print(f"  Umbral outlier: ±{self.std_threshold}σ")
        print(f"{'='*60}\n")
        print(f"Usando carpeta normalizada: {self.normalized_dir}")
        if self.verify_normalized_audios():
            print("[WARN] Se detectaron audios no normalizados. Por favor, normalízalos antes de continuar.")
            return []
        print("Los audios parecen estar normalizados. Continuando con el análisis...")
        # Paso 1: análisis individual
        audio_files = self._discover_files()
        print("archivos", audio_files)
        print(f"Archivos encontrados: {len(audio_files)}")
        print("PASO 1 ANALIZAR CADA ARCHIVO INDIVIDUALMENTE...")
        for fpath in audio_files:
            print(f"Analizando {fpath.name}...")
            stats = self._analyze_file(fpath)
            self._stats.append(stats)

        print(f"\nAnálisis individual completado. Archivos analizados: {len(self._stats)}")
        print("PASO 2 DETECTAR OUTLIERS A NIVEL DE CORPUS...")
        # Paso 2: detección de outliers corpus-level
        self._detect_outliers()
 
        # Paso 3: marcar en trash
        outlier_count = 0
        for stats in self._stats:
            if stats.is_outlier:
                reasons = "; ".join(stats.outlier_reasons)
                self.trash.add(stats.file_path, reason=reasons)
                outlier_count += 1

        print("PASO 4 APLICAR FILTRO BUTTERWORTH A LOS AUDIOS LIMPIOS...")
        
        # Paso 4: suavizar_audio
        self.soft_audio()
 
        # Resumen final
        self._print_corpus_summary(outlier_count)
 
        return [s.to_dict() for s in self._stats]
 
    def _discover_files(self) -> list[Path]:
        """Busca archivos de audio en la carpeta normalizada."""
        audio_files = []
        for especie, paths in self.bird_repo.get_audios_by_species().items():
            audio_files.extend(paths)
        return audio_files
 
    # ------------------------------------------------------------------
    # Análisis de un único archivo
    # ------------------------------------------------------------------
 
    def _analyze_file(self, fpath: Path) -> AudioStats:
        stats = AudioStats(
            file_path=str(fpath.resolve()),
            file_name=fpath.name,
        )
        try:
            audio_raw, sr = sf.read(str(fpath), always_2d=False)
        except Exception as exc:
            print(f"  [ERROR] No se pudo leer {fpath.name}: {exc}")
            stats.outlier_reasons.append(f"read_error: {exc}")
            stats.is_outlier = True
            return stats
 
        stats.sample_rate = sr
        stats.channels = 1 if audio_raw.ndim == 1 else audio_raw.shape[1]
        stats.n_samples = len(audio_raw) if audio_raw.ndim == 1 else audio_raw.shape[0]
        stats.duration_sec = stats.n_samples / sr if sr > 0 else 0.0
        if stats.duration_sec < self.min_duration_sec:
            stats.is_outlier = True

            stats.outlier_reasons.append(
                f"duration_too_short="
                f"{stats.duration_sec:.2f}s "
                f"(min={self.min_duration_sec:.2f}s)"
            )

        # Convertir a mono float32
        y: np.ndarray = self._converter.to_mono_float32(audio_raw)
 
        # --- Amplitud ---
        stats.amp_mean = float(np.mean(y))
        stats.amp_std = float(np.std(y))
        stats.amp_max = float(np.max(y))
        stats.amp_min = float(np.min(y))
        rms = float(np.sqrt(np.mean(y ** 2)))
        stats.amp_rms = rms
        peak = max(abs(stats.amp_max), abs(stats.amp_min))
        if rms > 0:
            stats.amp_peak_to_rms_db = float(20 * np.log10(peak / rms))
 
        # --- Energía ---
        stats.energy = float(np.sum(y ** 2))
 
        # --- DC offset ---
        stats.dc_offset = stats.amp_mean
 
        # --- Clipping ---
        stats.clipping_samples = int(np.sum(np.abs(y) >= 0.9999))
 
        # --- Silencio ---
        silence_mask = np.abs(y) < self.silence_thresh
        stats.silence_ratio = float(np.mean(silence_mask))
        stats.max_silence_run_sec = self._max_run(silence_mask) / sr
 
        # --- FFT y frecuencias ---
        freqs, magnitude = self._fft.compute_fft(y, sr)
        if len(freqs) > 0:
            # Frecuencia dominante
            stats.dominant_freq_hz = float(freqs[np.argmax(magnitude)])
 
            # Energía por bandas
            band_tuples = [(lo, hi) for lo, hi, _ in FrequencyBands().get_bands()]
            band_labels = [lbl for _, _, lbl in FrequencyBands().get_bands()]
            energies = self._fft.compute_band_energies(y, sr, band_tuples)
            stats.band_energies = {
                lbl: float(e) for lbl, e in zip(band_labels, energies)
            }
 
        return stats
 
    # ------------------------------------------------------------------
    # Detección de outliers corpus-level
    # ------------------------------------------------------------------
 
    _OUTLIER_METRICS = [
        "amp_rms",
        "amp_std",
        "energy",
        "silence_ratio",
        "spectral_centroid",
        "spectral_bandwidth",
        "duration_sec",
    ]
 
    def _detect_outliers(self) -> None:
        """
        Para cada métrica calcula media y std del corpus.
        Archivos cuyo valor supere ±std_threshold·σ se marcan como outlier.
        """
        valid = [s for s in self._stats if not s.is_outlier]
        if len(valid) < 3:
            print("[WARN] Muy pocos audios válidos para calcular distribución.")
            return
 
        corpus_stats: dict[str, tuple[float, float]] = {}
 
        for metric in self._OUTLIER_METRICS:
            values = np.array([getattr(s, metric) for s in valid], dtype=np.float64)
            mu = float(np.mean(values))
            sigma = float(np.std(values))
            corpus_stats[metric] = (mu, sigma)
 
        for stats in self._stats:
            if stats.is_outlier:
                continue
            for metric, (mu, sigma) in corpus_stats.items():
                if sigma == 0:
                    continue
                value = getattr(stats, metric)
                z = abs(value - mu) / sigma
                if z > self.std_threshold:
                    stats.is_outlier = True
                    stats.outlier_reasons.append(
                        f"{metric}={value:.4f} (z={z:.2f}, μ={mu:.4f}, σ={sigma:.4f})"
                    )
 
    def _create_output_dir_if_has_audios(
        self,
        especie: str,
        audio_paths: list
    ) -> Path | None:
        """
        Crea el directorio de salida solo si la especie
        tiene audios asociados.

        Returns:
            Path | None: ruta creada o None si no hay audios.
        """

        if not audio_paths:
            print(
                f"[INFO] La especie '{especie}' "
                "no tiene audios. Se omite carpeta."
            )
            return None

        output_dir = (
            Path(RoutesPath.PROCESSED_AUDIOS)
            / especie
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        return output_dir
 
    # ------------------------------------------------------------------
    # Entrenamiento y persistencia del modelo
    # ------------------------------------------------------------------
 
    def soft_audio(self) -> None:
        """
        Process the audios normalized to apply
        a butterworth filter for each.
        """

        print(
            "\nObteniendo audios limpios "
            "para aplicar filtro Butterworth..."
        )

        audio_paths_by_especie = (
            self.bird_repo.get_audios_by_species()
        )

        print(
            f"Carpetas encontrados: "
            f"{len(audio_paths_by_especie)}"
        )

        if not audio_paths_by_especie:
            print(
                "[WARN] No se encontraron "
                "archivos de audio para entrenar "
                "el modelo."
            )
            return
    
        print("PASO 5 APLICAR FILTRO BUTTERWORTH A LOS AUDIOS LIMPIOS...")
        print(f"Total especies a procesar: {len(audio_paths_by_especie)}")
        for especie, audio_paths in (
            audio_paths_by_especie.items()
        ):

            # Crear carpeta SOLO si hay audios
            output_dir = (
                self._create_output_dir_if_has_audios(
                    especie,
                    audio_paths
                )
            )

            if output_dir is None:
                continue

            bird = self.bird_repo.get_by_species(
                especie
            )
            print(f"\nProcesando especie: {especie}")
            print()
            print(bird.__str__())
            print()
            for fpath in audio_paths:

                try:
                    # Leer audio
                    audio_raw, sr = sf.read(
                        str(fpath),
                        always_2d=False
                    )

                    # Convertir a mono float32
                    y = (
                        self._converter
                        .to_mono_float32(
                            audio_raw
                        )
                    )

                    # Obtener rango frecuencias
                    freqs = (
                        bird.vocalizaciones
                        .frecuencias_hz
                        .rango_principal
                    )

                    # Crear filtro
                    butterworth = (
                        ButterworthController(
                            order=(
                                self
                                .butterworth_order
                            ),
                            filter_type="band",
                            high_freq=freqs.max,
                            low_freq=freqs.min,
                        )
                    )
                    
                    # construir filtro (diagnósticos incluidos)
                    filter_model = butterworth.build(signal=y, sr=sr)
                    params = butterworth.last_params

                    print(f"  [DEBUG] Params: order={params.order}, low={params.low_freq}, high={params.high_freq}, auto={params.auto_detected}")

                    y_filtered = filter_model.apply_bandpass(
                        signal=y,
                        sr=sr,
                        low_freq=params.low_freq,
                        high_freq=params.high_freq,
                    )

                    # Diagnostics: check for NaN/Inf and amplitude
                    if not np.isfinite(y_filtered).all():
                        print(f"  [ERROR] filtered contains non-finite values for {fpath.name}")
                        continue

                    pre_peak = float(np.max(np.abs(y))) if y.size else 0.0
                    post_peak = float(np.max(np.abs(y_filtered))) if y_filtered.size else 0.0
                    print(f"  [DEBUG] pre_peak={pre_peak:.6f}, post_peak={post_peak:.6f}")

                    # Normalize output to avoid clipping when writing
                    peak = post_peak
                    if peak > 0:
                        y_out = (y_filtered / peak * 0.99).astype(np.float32)
                    else:
                        y_out = y_filtered.astype(np.float32)

                    output_path = output_dir / fpath.name

                    # Guardar audio como float WAV para evitar cuantización agresiva
                    sf.write(str(output_path), y_out, sr, subtype="FLOAT")

                except Exception as exc:
                    print(
                        f"  [ERROR] No se pudo procesar {fpath.name}: {exc}"
                    )
                    continue    
 
    def _print_corpus_summary(self, outlier_count: int) -> None:
        total = len(self._stats)
        clean = total - outlier_count
        print(f"\n{'='*60}")
        print(f"  RESUMEN DEL CORPUS")
        print(f"  Total archivos analizados : {total}")
        print(f"  Archivos limpios          : {clean}")
        print(f"  Outliers marcados         : {outlier_count}")
        if outlier_count:
            print(
                f"\n  Revisa trash_audios.json y ejecuta:\n"
                f"    analyzer.trash.delete_marked(dry_run=True)  # preview\n"
                f"    analyzer.trash.delete_marked()              # borrar"
            )
        print(f"{'='*60}\n")
 
    def verify_normalized_audios(self) -> bool:
        """
        True  -> hay que normalizar
        False -> ya están normalizados
        """

        normalized_dir = Path("dataset_aves_normalized")

        if not normalized_dir.exists():
            return True

        audio_files = list(normalized_dir.rglob("*.wav"))

        if len(audio_files) == 0:
            return True

        print("Usando carpeta normalizada:", normalized_dir)
        return False
        
 
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
 
    @staticmethod
    def _max_run(mask: np.ndarray) -> int:
        """Longitud máxima de una racha de True en un array booleano."""
        max_run = 0
        current = 0
        for val in mask:
            if val:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0
        return max_run