from pathlib import Path
from typing import Optional
from config.routes_path import RoutesPath
from config.frecuency_bands import FrequencyBands
from core.audio_converter import AudioConverter
from core.fft import FFTProcessor
from core.signals_processor import SignalsProcessor

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
 
        self._converter = AudioConverter()
        self._fft = FFTProcessor()
 
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
 
        audio_files = self._discover_files()
        if not audio_files:
            print("[WARN] No se encontraron archivos de audio.")
            return []
 
        print(f"Archivos encontrados: {len(audio_files)}\n")
 
        # Paso 1: análisis individual
        for fpath in audio_files:
            stats = self._analyze_file(fpath)
            self._stats.append(stats)
            self._print_file_summary(stats)
 
        # Paso 2: detección de outliers corpus-level
        self._detect_outliers()
 
        # Paso 3: marcar en trash
        outlier_count = 0
        for stats in self._stats:
            if stats.is_outlier:
                reasons = "; ".join(stats.outlier_reasons)
                self.trash.add(stats.file_path, reason=reasons)
                outlier_count += 1
 
        # Paso 4: entrenar y guardar modelo Butterworth
        self._train_and_save_model()
 
        # Resumen final
        self._print_corpus_summary(outlier_count)
 
        return [s.to_dict() for s in self._stats]
 
    # ------------------------------------------------------------------
    # Descubrimiento de archivos
    # ------------------------------------------------------------------
 
    def _discover_files(self) -> list[Path]:
        extensions = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}
        if not self.normalized_dir.exists():
            raise FileNotFoundError(
                f"Directorio no encontrado: {self.normalized_dir}"
            )
        files = sorted(
            p for p in self.normalized_dir.rglob("*")
            if p.suffix.lower() in extensions
        )
        return files
 
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
            # Centroide espectral
            total_mag = np.sum(magnitude)
            if total_mag > 0:
                stats.spectral_centroid = float(
                    np.sum(freqs * magnitude) / total_mag
                )
                # Ancho de banda espectral
                stats.spectral_bandwidth = float(
                    np.sqrt(
                        np.sum(((freqs - stats.spectral_centroid) ** 2) * magnitude)
                        / total_mag
                    )
                )
            # Frecuencia dominante
            stats.dominant_freq_hz = float(freqs[np.argmax(magnitude)])
 
            # Energía por bandas
            band_tuples = [(lo, hi) for lo, hi, _ in FREQUENCY_BANDS]
            band_labels = [lbl for _, _, lbl in FREQUENCY_BANDS]
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
 
        print("\n--- Distribución del corpus ---")
        for metric, (mu, sigma) in corpus_stats.items():
            print(f"  {metric:30s}  μ={mu:.4f}  σ={sigma:.4f}")
        print()
 
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
 
    # ------------------------------------------------------------------
    # Entrenamiento y persistencia del modelo
    # ------------------------------------------------------------------
 
    def _train_and_save_model(self) -> None:
        """Entrena un FilterButterworth sobre el corpus y lo guarda."""
        trainer = TrainModelButterworth(
            cutoff_freq=self.butterworth_cutoff_hz,
            order=self.butterworth_order,
        )
 
        # Para entrenar necesitamos al menos una señal representativa.
        # Usamos el audio "mediano" (por RMS) del corpus limpio.
        clean_stats = [s for s in self._stats if not s.is_outlier]
        representative_signal: Optional[np.ndarray] = None
 
        if clean_stats:
            median_rms = np.median([s.amp_rms for s in clean_stats])
            best = min(clean_stats, key=lambda s: abs(s.amp_rms - median_rms))
            try:
                audio_raw, sr = sf.read(best.file_path, always_2d=False)
                representative_signal = self._converter.to_mono_float32(audio_raw)
                representative_sr = sr
            except Exception:
                representative_sr = 16000
        else:
            representative_sr = 16000
 
        filter_model: FilterButterworth = trainer.train(
            signal=representative_signal
            if representative_signal is not None
            else np.zeros(16000, dtype=np.float32),
            sr=representative_sr,
        )
 
        model_name = (
            f"butterworth_lp_"
            f"order{self.butterworth_order}_"
            f"cutoff{int(self.butterworth_cutoff_hz)}hz"
        )
 
        saved_path = self.models.save(
            name=model_name,
            model=filter_model,
            metadata={
                "type": "FilterButterworth",
                "order": self.butterworth_order,
                "cutoff_freq_hz": self.butterworth_cutoff_hz,
                "trained_on": len(clean_stats),
                "corpus_dir": str(self.normalized_dir),
            },
            overwrite=True,
        )
        print(f"\n✓ Modelo guardado en: {saved_path}")
 
    # ------------------------------------------------------------------
    # Salida por consola
    # ------------------------------------------------------------------
 
    def _print_file_summary(self, stats: AudioStats) -> None:
        status = "⚠ OUTLIER" if stats.is_outlier else "✓"
        print(
            f"  [{status}] {stats.file_name:<35s} "
            f"dur={stats.duration_sec:.2f}s  "
            f"rms={stats.amp_rms:.4f}  "
            f"silence={stats.silence_ratio:.2%}  "
            f"centroid={stats.spectral_centroid:.0f}Hz"
        )
        if stats.outlier_reasons:
            for reason in stats.outlier_reasons:
                print(f"           → {reason}")
 
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