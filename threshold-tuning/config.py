from dataclasses import dataclass,field

@dataclass
class ThresholdConfig:
    models_csvs:   list[str] = field(
        default_factory=lambda: [
            "outputs/checkpoints-q1.5b/test_results_elf.csv",
            "outputs/checkpoints-q1.5b/test_results_pe.csv",
            "outputs/checkpoints-q1.5b-lora-classic/test_results_elf.csv",
            "outputs/checkpoints-q1.5b-lora-classic/test_results_pe.csv",
            "outputs/checkpoints-q1.5b-lora-attention/test_results_elf.csv",
            "outputs/checkpoints-q1.5b-lora-attention/test_results_pe.csv",
            "outputs/checkpoints-q1.5b-lora-full/test_results_elf.csv",
            "outputs/checkpoints-q1.5b-lora-full/test_results_pe.csv",
            "outputs/checkpoints-q3b/test_results_elf.csv",
            "outputs/checkpoints-q3b/test_results_pe.csv",
            "outputs/checkpoints-q3b-lora-classic/test_results_elf.csv",
            "outputs/checkpoints-q3b-lora-classic/test_results_pe.csv",
            "outputs/checkpoints-q3b-lora-attention/test_results_elf.csv",
            "outputs/checkpoints-q3b-lora-attention/test_results_pe.csv",
            "outputs/checkpoints-q3b-lora-full/test_results_elf_1_9.csv",
            "outputs/checkpoints-q3b-lora-full/test_results_pe.csv",
        ]    
    )