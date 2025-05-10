import sys
from pathlib import Path
from loader import ConfigLoader
from validator import NoiseConfig, ChannelConfig, SystemConfig

def main(config_path: str) -> None:
    """Main workflow: Load -> Validate -> Use config."""
    try:
        # 1. Загрузка конфига
        print(f"Loading config from {config_path}...")
        raw_config = ConfigLoader.load(config_path)
        
        # 2. Валидация
        print("Validating config...")
        config = SystemConfig(**raw_config)
        
        # 3. Использование
        print("\nConfig successfully loaded!")
        print(f"Bits: {config.generation_signal_configuration.count_bit}")
        print(f"Modulation: {config.modulation_scheme}")
        print(f"Noise enabled: {config.noise_configuration.noise_enabled}")
        print(f"SNR: {config.noise_configuration.signal_to_noise_ratio_db} dB")
        
        # Здесь можно добавить логику симулятора
        run_simulation(config)
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

def run_simulation(config: SystemConfig) -> None:
    """Пример использования валидированного конфига"""
    print("\nStarting simulation with params:")
    print(f"- Channel type: {config.channel_configuration.channel_type}")
    if config.noise_configuration.noise_enabled:
        print(f"- Noise level: {config.noise_configuration.signal_to_noise_ratio_db} dB")

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "settings.json"
    main(config_file)