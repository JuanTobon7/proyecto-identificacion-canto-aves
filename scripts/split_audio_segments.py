from pathlib import Path
import librosa
import soundfile as sf
import logging
from config.routes_path import RoutesPath

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Ruta origen
DATASET_PATH = (
    Path(__file__).parent.parent
    / RoutesPath.BANK_AUDIOS_CRUDE
)

# Ruta destino
OUTPUT_PATH = (
    Path(__file__).parent.parent
    / RoutesPath.BANK_AUDIOS_NORMALIZED
)

SEGMENT_DURATION_MS = 5000


def split_audio(
    audio_path,
    output_dir,
    segment_duration=SEGMENT_DURATION_MS
):
    """
    Divide un audio en segmentos.
    """

    try:
        file_stem = audio_path.stem

        logger.info(
            f"Procesando: {audio_path.name}"
        )

        # Cargar audio
        y, sr = librosa.load(
            str(audio_path),
            sr=None
        )

        total_samples = len(y)

        segment_samples = int(
            sr * (segment_duration / 1000)
        )

        num_segments = (
            total_samples
            + segment_samples
            - 1
        ) // segment_samples

        total_duration_s = total_samples / sr

        logger.info(
            f"  Duración: "
            f"{total_duration_s:.2f}s "
            f"-> {num_segments} segmentos"
        )

        # Crear carpeta si no existe
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for segment_num in range(
            1,
            num_segments + 1
        ):
            start_sample = (
                (segment_num - 1)
                * segment_samples
            )

            end_sample = min(
                segment_num * segment_samples,
                total_samples
            )

            segment_audio = y[
                start_sample:end_sample
            ]

            segment_duration_s = (
                len(segment_audio) / sr
            )

            output_filename = (
                f"{file_stem}_{segment_num}.wav"
            )

            output_path = (
                output_dir
                / output_filename
            )

            sf.write(
                str(output_path),
                segment_audio,
                sr
            )

            logger.info(
                f"  ✓ {output_filename} "
                f"({segment_duration_s:.2f}s)"
            )

        return num_segments

    except Exception as e:
        logger.exception(
            f"Error procesando "
            f"{audio_path.name}: {e}"
        )

        return 0


def main():

    if not DATASET_PATH.exists():
        logger.error(
            f"Dataset path no existe: "
            f"{DATASET_PATH}"
        )
        return

    logger.info(
        "Iniciando división "
        "de audios"
    )

    logger.info(
        f"Origen: {DATASET_PATH}"
    )

    logger.info(
        f"Destino: {OUTPUT_PATH}"
    )

    total_audios = 0
    total_segmentos = 0

    for species_dir in sorted(
        DATASET_PATH.iterdir()
    ):

        if not species_dir.is_dir():
            continue

        species_name = species_dir.name

        logger.info(
            f"\n📁 Procesando: "
            f"{species_name}"
        )

        audio_files = sorted(
            species_dir.glob("*.mp3")
        )

        if not audio_files:
            logger.warning(
                f"No hay MP3 en "
                f"{species_name}"
            )
            continue

        logger.info(
            f"Encontrados: "
            f"{len(audio_files)}"
        )

        # Carpeta destino por especie
        species_output_dir = (
            OUTPUT_PATH
            / species_name
        )

        species_total = 0

        for audio_file in audio_files:

            num_segments = split_audio(
                audio_file,
                species_output_dir
            )

            species_total += num_segments
            total_audios += 1

        total_segmentos += species_total

        logger.info(
            f"✓ {species_name}: "
            f"{species_total} "
            f"segmentos"
        )

    logger.info("=" * 60)
    logger.info(
        f"Audios procesados: "
        f"{total_audios}"
    )
    logger.info(
        f"Segmentos generados: "
        f"{total_segmentos}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()