from config.routes_path import RoutesPath

import argparse

def main():
    parser = argparse.ArgumentParser(description="Análisis y limpieza de corpus de audio.")
    
    parser.add_argument(
        "--model",
        choices=["fir", "butterworth"],
        default="butterworth",
        help="Modelo de filtro a utilizar."
    )
    
    args = parser.parse_args()
    
if __name__ == "__main__":    
    main()
    
    
    