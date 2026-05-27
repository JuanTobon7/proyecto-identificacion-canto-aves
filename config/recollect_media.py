import os
import re
import time
import random
import requests
import pandas as pd
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==========================================================
# CONFIG
# ==========================================================

CSV_FILE = "birds/ebd_US-AL-125_202503_202503_smp_relMar-2025.txt"

OUTPUT_DIR = "dataset_aves"

# Separador (eBird usa tabulaciones)
CSV_SEPARATOR = "\t"

# Rate limit (responsable)
MIN_WAIT = 2.0
MAX_WAIT = 5.0

# Timeout
REQUEST_TIMEOUT = 60

# ==========================================================
# HELPERS
# ==========================================================

def clean_name(name: str) -> str:
    """
    Limpia nombres para carpetas/archivos.
    """
    name = str(name).strip()
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name)
    return name


def find_ml_column(columns):
    """
    Busca automáticamente la columna del ID ML.
    Para archivos eBird, nota que no hay IDs ML directos.
    Retorna None si no existe.
    """
    possible = [
        "ML Catalog Number",
        "CATALOG NUMBER",
        "MEDIA ASSET ID",
        "GLOBAL UNIQUE IDENTIFIER",
        "ASSET ID",
        "ML ID",
    ]

    for col in columns:
        if col in possible:
            return col

    # fallback: buscar columnas que tengan ml
    for col in columns:
        if "ml" in col.lower():
            return col

    return None

def find_species_column(columns):
    """
    Busca columna de nombre del ave.
    """
    possible = [
        "COMMON NAME",
        "Common Name",
        "common_name",
    ]

    for col in columns:
        if col in possible:
            return col

    raise Exception("No encontré columna de especie (COMMON NAME).")


def build_session():
    """
    Session con retries.
    """
    session = requests.Session()
    print("Construyendo sesión con retries...")
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update({
        "User-Agent": "AcademicBirdDataset/1.0 (Research Purpose)"
    })

    return session


def extract_ml_number(value):
    """
    Convierte:
    ML658458906 -> 658458906
    658458906 -> 658458906
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    match = re.search(r'(\d+)', text)

    if match:
        return match.group(1)

    return None


# ==========================================================
# DOWNLOAD
# ==========================================================

def download_audio(session, ml_id, species_folder):

    output_file = species_folder / f"ML{ml_id}.mp3"

    if output_file.exists():
        print(f"[SKIP] {output_file.name}")
        return
    print(f"Descargando ML{ml_id}...")
    url = (
        f"https://cdn.download.ams.birds.cornell.edu/api/v1/asset/{ml_id}"
    )

    try:
        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT,
            stream=True
        )

        if response.status_code == 404:
            print(f"[404] ML{ml_id}")
            return

        response.raise_for_status()

        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"[OK] {output_file}")

    except Exception as e:
        print(f"[ERROR] ML{ml_id}: {e}")


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("Leyendo archivo eBird...")

    df = pd.read_csv(CSV_FILE, sep=CSV_SEPARATOR, low_memory=False)

    ml_column = find_ml_column(df.columns)
    print(f"Columnas encontradas: {list(df.columns[:5])}")
    if ml_column is None:
        print("⚠️  No se encontró columna con IDs ML directo.")
        print("Este archivo eBird no contiene IDs de descargas de media.")
        print("Columnas disponibles:", list(df.columns[:5]))
        return

    species_column = find_species_column(df.columns)

    print(f"✓ Columna ML encontrada: {ml_column}")
    print(f"✓ Columna especie: {species_column}")
    session = build_session()
    print(f"Total registros: {len(df)}")
    total = len(df)

    for idx, row in df.iterrows():

        species = clean_name(
            row.get(species_column, "Unknown")
        )
        print(f"\nProcesando especie: {species} ({idx+1}/{total})")
        species_folder = Path(
            OUTPUT_DIR
        ) / species

        species_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        ml_id = extract_ml_number(
            row.get(ml_column)
        )
        print(f"ML ID extraído: {ml_id}")
        if not ml_id:
            continue

        print(f"\n[{idx+1}/{total}] ML{ml_id}")

        download_audio(
            session,
            ml_id,
            species_folder
        )

        # Rate limiting responsable
        sleep_time = random.uniform(
            MIN_WAIT,
            MAX_WAIT
        )

        print(
            f"Esperando {sleep_time:.2f}s..."
        )

        time.sleep(sleep_time)

    print("\nFinalizado.")


if __name__ == "__main__":
    main()